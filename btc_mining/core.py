"""Gate BTC perpetual formula mining using the repository's AlphaGPT generator.

No keys, database or live trading. All times are UTC; end is exclusive.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import requests
import torch
from torch import nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_core.alphagpt import AlphaGPT, MTPHead

FEATURES = ['RET', 'MOM24', 'VOL_SHOCK', 'DEV24', 'PRESSURE', 'VOLATILITY24']
OPS = [('ADD', 2), ('SUB', 2), ('MUL', 2), ('DIV', 2), ('NEG', 1),
       ('ABS', 1), ('SIGN', 1), ('GATE', 3), ('JUMP', 1),
       ('DECAY', 1), ('DELAY1', 1), ('MAX3', 1)]
TOKENS = FEATURES + [x[0] for x in OPS] + ['EOS']
EOS = len(TOKENS) - 1
SECONDS = {'1h': 3600, '4h': 14400}


def recent_date(days=0):
    return str((pd.Timestamp.now(tz='UTC').normalize() - pd.Timedelta(days=days)).date())


@dataclass
class Config:
    contract: str = 'BTC_USDT'
    interval: str = '1h'
    start: str = field(default_factory=lambda: recent_date(170))
    end: str = field(default_factory=recent_date)
    steps: int = 200
    batch_size: int = 128
    max_len: int = 12
    seeds: tuple = (42, 43, 44)
    learning_rate: float = 0.0003
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    signal_threshold: float = 0.2
    norm_window: int = 168
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    top_k: int = 30
    min_entries: int = 10
    drawdown_penalty: float = 2.0
    complexity_penalty: float = 0.005
    api_base: str = 'https://api.gateio.ws/api/v4'
    charts: bool = True

    def validate(self):
        integer_fields = ['steps', 'batch_size', 'max_len', 'norm_window', 'top_k', 'min_entries']
        if any(isinstance(getattr(self, k), bool) or not isinstance(getattr(self, k), int) for k in integer_fields):
            raise ValueError('Training counts/window sizes must be integers')
        if any(not isinstance(s, int) or not 0 <= s < 2**32 for s in self.seeds):
            raise ValueError('Seeds must be integers in [0, 2**32)')
        if any(not np.isfinite(getattr(self,k)) for k in ['learning_rate','fee_bps','slippage_bps',
            'signal_threshold','train_fraction','validation_fraction','drawdown_penalty','complexity_penalty']):
            raise ValueError('All numeric parameters must be finite')
        if self.learning_rate <= 0 or min(self.drawdown_penalty, self.complexity_penalty) < 0:
            raise ValueError('Learning rate must be positive; penalties nonnegative')
        if self.interval not in SECONDS or not 2 <= self.max_len <= 12:
            raise ValueError('interval must be 1h/4h; max_len must be 2..12')
        if self.steps < 1 or self.batch_size < 2 or not self.seeds:
            raise ValueError('steps >= 1, batch_size >= 2 and seeds required')
        if self.norm_window < 24 or self.top_k < 1 or self.min_entries < 1:
            raise ValueError('norm_window >= 24, top_k >= 1, min_entries >= 1')
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError('Invalid chronological split fractions')
        if not 0 <= self.signal_threshold < 1 or min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError('Invalid threshold or trading costs')
        start, end = utc_seconds(self.start), utc_seconds(self.end)
        dt = SECONDS[self.interval]
        if start % dt or end % dt or start >= end:
            raise ValueError('Dates must be aligned UTC candle boundaries; start < end')
        if end > int(time.time()) // dt * dt:
            raise ValueError('end includes an unfinished/future candle')


def utc_seconds(value):
    ts = pd.Timestamp(value)
    ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
    return int(ts.timestamp())


def get_json(session, url, params, retries=5):
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=(10, 45))
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.RequestException(f'HTTP {resp.status_code}')
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f'Expected Gate list response, got {str(data)[:200]}')
            return data
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError('Unreachable')


def validate_candles(df, cfg):
    df = df.copy()
    required = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not set(required).issubset(df.columns):
        raise ValueError(f'CSV must contain {required}')
    for name in required:
        df[name] = pd.to_numeric(df[name], errors='raise')
    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError('NaN/inf in candles')
    if not (df.time == df.time.astype('int64')).all():
        raise ValueError('Candle times must be integer Unix seconds')
    df['time'] = df.time.astype('int64')
    df = df.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    start, end, dt = utc_seconds(cfg.start), utc_seconds(cfg.end), SECONDS[cfg.interval]
    df = df[(df.time >= start) & (df.time < end)].reset_index(drop=True)
    expected = np.arange(start, end, dt, dtype=np.int64)
    if not np.array_equal(df.time.to_numpy(), expected):
        missing = np.setdiff1d(expected, df.time.to_numpy())
        raise ValueError(f'Incomplete/misaligned candles: {len(missing)} missing; '
                         f'first missing: {missing[:5].tolist()}. No silent forward filling.')
    p = df[['open', 'high', 'low', 'close']]
    if (p <= 0).any().any() or (df.volume < 0).any():
        raise ValueError('Nonpositive price or negative quote volume')
    if ((df.high < p[['open', 'close', 'low']].max(axis=1)) |
            (df.low > p[['open', 'close', 'high']].min(axis=1))).any():
        raise ValueError('Invalid OHLC ordering')
    return df


def fetch_gate(cfg, data_dir):
    """Bounded nonoverlapping windows, strict complete candle grid, cached CSVs."""
    cfg.validate()
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps({k: getattr(cfg, k) for k in
        ['contract', 'interval', 'start', 'end', 'api_base']}, sort_keys=True).encode()).hexdigest()[:12]
    candle_path, funding_path = data_dir / f'candles_{key}.csv', data_dir / f'funding_{key}.csv'
    start, end, dt = utc_seconds(cfg.start), utc_seconds(cfg.end), SECONDS[cfg.interval]
    if not funding_path.exists() and start < int(time.time()) - 179 * 86400:
        raise ValueError('Gate funding API currently rejects history older than 180 days. '
                         'Use the default recent 170 days, or supply historical candle AND funding CSVs.')
    with requests.Session() as session:
        if candle_path.exists():
            candles = validate_candles(pd.read_csv(candle_path), cfg)
        else:
            rows = []
            for cursor in range(start, end, 1000 * dt):
                stop = min(cursor + 1000 * dt, end)
                batch = get_json(session, cfg.api_base + '/futures/usdt/candlesticks',
                    {'contract': cfg.contract, 'interval': cfg.interval, 'from': cursor, 'to': stop - 1})
                for x in batch:
                    rows.append({'time': x['t'], 'open': x['o'], 'high': x['h'],
                        'low': x['l'], 'close': x['c'], 'volume': x['sum']})
                print(f'Candles: {pd.Timestamp(stop, unit="s", tz="UTC")} ({len(rows)} rows)', flush=True)
                time.sleep(0.1)
            candles = validate_candles(pd.DataFrame(rows), cfg)
            candles.to_csv(candle_path, index=False)
        if funding_path.exists():
            funding = pd.read_csv(funding_path)
        else:
            rows = []
            # Seven-day windows contain <=168 records even at hourly settlement.
            for cursor in range(start, end, 7 * 86400):
                stop = min(cursor + 7 * 86400, end)
                batch = get_json(session, cfg.api_base + '/futures/usdt/funding_rate',
                    {'contract': cfg.contract, 'from': cursor, 'to': stop - 1, 'limit': 1000})
                if not batch:
                    raise ValueError(f'No funding history in UTC day {cursor}; refusing zero-cost fallback')
                rows.extend({'time': x['t'], 'rate': x['r']} for x in batch)
                print(f'Funding: {pd.Timestamp(stop, unit="s", tz="UTC")} ({len(rows)} rows)', flush=True)
                time.sleep(0.05)
            funding = pd.DataFrame(rows).drop_duplicates('time').sort_values('time')
            funding.to_csv(funding_path, index=False)
    funding = validate_funding(funding, cfg)
    return candles, funding, [candle_path, funding_path]


def validate_funding(funding, cfg):
    funding = funding.copy()
    for col in ['time', 'rate']:
        funding[col] = pd.to_numeric(funding[col], errors='raise')
    if funding.empty or not np.isfinite(funding[['time', 'rate']].to_numpy()).all():
        raise ValueError('Empty/nonfinite funding history')
    if not (funding.time == funding.time.astype('int64')).all():
        raise ValueError('Funding timestamps must be integer Unix seconds')
    funding['time'] = funding.time.astype('int64')
    funding = funding.drop_duplicates('time').sort_values('time').reset_index(drop=True)
    start, end = utc_seconds(cfg.start), utc_seconds(cfg.end)
    funding = funding[(funding.time >= start) & (funding.time < end)].reset_index(drop=True)
    stamps = np.r_[start, funding.time.to_numpy(), end]
    if funding.empty or np.diff(stamps).max() > 12 * 3600:
        raise ValueError('Funding coverage gap >12h; inspect source instead of assuming zero')
    return funding


def make_features(df, window=168):
    """All features use current/past CLOSED bars; no fitted full-series statistics."""
    c, v = df.close, df.volume
    ret = np.log(c / c.shift(1))
    raw = pd.DataFrame({
        'RET': ret, 'MOM24': np.log(c / c.shift(24)),
        'VOL_SHOCK': np.log((v + 1) / (v.rolling(24).mean() + 1)),
        'DEV24': c / c.rolling(24).mean() - 1,
        'PRESSURE': (c - df.open) / (df.high - df.low).clip(lower=1e-9),
        'VOLATILITY24': ret.rolling(24).std(ddof=0),
    })
    med = raw.rolling(window, min_periods=window).median()
    # Exact rolling MAD over the same window (not a second rolling approximation).
    mad = raw.rolling(window, min_periods=window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    feats = ((raw - med) / (mad + 1e-6)).clip(-5, 5)
    good = np.isfinite(feats.to_numpy()).all(axis=1)
    if not good.any():
        raise ValueError('Too few bars for causal feature warmup')
    warmup = int(np.flatnonzero(good)[0])
    if not good[warmup:].all():
        raise ValueError('Unexpected invalid features after warmup')
    return feats.fillna(0).to_numpy(dtype=np.float64).T, warmup


def delay(x, n):
    out = np.zeros_like(x)
    if n < len(x):
        out[n:] = x[:-n]
    return out


def execute(tokens, features):
    stack = []
    with np.errstate(all='ignore'):
        for token in tokens:
            if token == EOS:
                break
            if 0 <= token < len(FEATURES):
                stack.append(features[token])
                continue
            if not len(FEATURES) <= token < EOS:
                raise ValueError('Unknown token')
            name, arity = OPS[token - len(FEATURES)]
            if len(stack) < arity:
                raise ValueError('Formula stack underflow')
            args = stack[-arity:]
            del stack[-arity:]
            x = args[0]
            if name == 'ADD': out = x + args[1]
            elif name == 'SUB': out = x - args[1]
            elif name == 'MUL': out = x * args[1]
            elif name == 'DIV':
                y = args[1]
                out = x / np.where(np.abs(y) < 1e-6, np.where(y < 0, -1e-6, 1e-6), y)
            elif name == 'NEG': out = -x
            elif name == 'ABS': out = np.abs(x)
            elif name == 'SIGN': out = np.sign(x)
            elif name == 'GATE': out = np.where(x > 0, args[1], args[2])
            elif name == 'JUMP':
                s = pd.Series(x)
                z = (s - s.rolling(20).mean()) / (s.rolling(20).std(ddof=0) + 1e-6)
                out = np.maximum(z.fillna(0).to_numpy() - 3, 0)
            elif name == 'DECAY': out = x + 0.8 * delay(x, 1) + 0.6 * delay(x, 2)
            elif name == 'DELAY1': out = delay(x, 1)
            else: out = np.maximum(x, np.maximum(delay(x, 1), delay(x, 2)))
            stack.append(np.clip(np.nan_to_num(out, nan=0, posinf=1e6, neginf=-1e6), -1e6, 1e6))
    if len(stack) != 1:
        raise ValueError('Formula must leave exactly one result')
    return stack[0]


def decode(tokens):
    stack = []
    for t in tokens:
        if t == EOS: break
        if t < len(FEATURES): stack.append(FEATURES[t])
        else:
            name, arity = OPS[t - len(FEATURES)]
            if len(stack) < arity: raise ValueError('Invalid formula')
            args = stack[-arity:]
            del stack[-arity:]
            stack.append(f'{name}({", ".join(args)})')
    if len(stack) != 1: raise ValueError('Invalid formula')
    return stack[0]


class Backtester:
    def __init__(self, df, funding, cfg, warmup):
        self.cfg = cfg
        self.df = df
        self.periods_per_year = 365.25 * 86400 / SECONDS[cfg.interval]
        self.ret = np.zeros(len(df))
        self.ret[:-2] = df.open.to_numpy()[2:] / df.open.to_numpy()[1:-1] - 1
        self.funding = np.zeros(len(df))
        times = df.time.to_numpy()
        # signal[t] owns (open[t+1], open[t+2]], including settlement at exit.
        for stamp, rate in funding[['time', 'rate']].itertuples(index=False, name=None):
            i = int(np.searchsorted(times, stamp, side='left')) - 2
            if 0 <= i < len(df) - 2:
                self.funding[i] += rate
        a, b = int(len(df) * cfg.train_fraction), int(len(df) * (cfg.train_fraction + cfg.validation_fraction))
        # Purge target horizons crossing each boundary; restart flat for each segment.
        self.splits = {'train': (warmup, a - 2), 'validation': (a, b - 2), 'test': (b, len(df) - 2)}
        if any(stop - start < 100 for start, stop in self.splits.values()):
            raise ValueError('Each split needs >=100 tradable bars after warmup/purge')

    def evaluate(self, signal, split, length=0, positions=False):
        start, stop = self.splits[split]
        s = np.tanh(signal[start:stop])
        p = np.where(s > self.cfg.signal_threshold, 1.0,
                     np.where(s < -self.cfg.signal_threshold, -1.0, 0.0))
        return self.evaluate_positions(p, start, stop, length, positions)

    def evaluate_positions(self, p, start, stop, length=0, include_series=False):
        prev = np.r_[0.0, p[:-1]]
        # Target exposure is reset to +/-1 equity at each open. Include resizing
        # of an unchanged direction after price/funding changes, as well as flips.
        r, f = self.ret[start:stop], self.funding[start:stop]
        cost_rate = (self.cfg.fee_bps + self.cfg.slippage_bps) / 1e4
        turnover = np.abs(p - prev)
        entries = int(np.count_nonzero((p != 0) & (p != prev)))
        costs = turnover * cost_rate
        net = p * r - p * f - costs
        # Charge approximate maintenance rebalancing on same-direction bars.
        same = (p[1:] != 0) & (p[1:] == p[:-1])
        resize = np.abs(p[:-1] * (1 + r[:-1]) / (1 + net[:-1]) - p[1:])
        costs[1:] += np.where(same, resize * cost_rate, 0)
        net = p * r - p * f - costs
        net[-1] -= abs(p[-1]) * cost_rate  # liquidate at segment end
        if not np.isfinite(net).all() or (net <= -1).any():
            raise ValueError('Nonfinite return or bankruptcy in simplified 1x model')
        equity = np.cumprod(1 + net)
        peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
        dd = float(np.max(1 - equity / peaks))
        sd = float(net.std(ddof=0))
        sharpe = float(net.mean() / sd * np.sqrt(self.periods_per_year)) if sd > 1e-12 else 0.0
        reward = float(np.clip(sharpe, -10, 10) - self.cfg.drawdown_penalty * dd
                       - self.cfg.complexity_penalty * length)
        eligible = entries >= self.cfg.min_entries
        result = {'score': reward if eligible else -20.0, 'eligible': bool(eligible),
            'total_return': float(equity[-1] - 1), 'sharpe': sharpe,
            'max_drawdown': dd, 'entries': entries,
            'long_fraction': float((p > 0).mean()), 'short_fraction': float((p < 0).mean()),
            'bars': len(p)}
        if include_series:
            result['series'] = pd.DataFrame({'signal_time': self.df.time.iloc[start:stop].to_numpy(),
                'entry_time': self.df.time.iloc[start + 1:stop + 1].to_numpy(),
                'exit_time': self.df.time.iloc[start + 2:stop + 2].to_numpy(),
                'position': p, 'market_return': r, 'funding_rate': f,
                'net_return': net, 'equity': equity})
        return result


def make_generator(device):
    # Keep the actual repository's looped AlphaGPT architecture; replace token heads
    # for BTC's 6 features + 12 operators + explicit EOS. LoRD remains disabled.
    model = AlphaGPT().to(device)
    model.token_emb = nn.Embedding(len(TOKENS), model.d_model).to(device)
    model.mtp_head = MTPHead(model.d_model, len(TOKENS), num_tasks=3).to(device)
    model.vocab = TOKENS
    model.vocab_size = len(TOKENS)
    return model


def sample_formulas(model, batch_size, max_len, device):
    inp = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    depth = torch.zeros(batch_size, dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    arities = torch.tensor([0] * len(FEATURES) + [a for _, a in OPS] + [0], device=device)
    sums = torch.zeros(batch_size, device=device)
    seqs = []
    for step in range(max_len):
        logits, _, _ = model(inp)
        after = depth[:, None] + torch.where(arities[None, :] == 0, 1, 1 - arities[None, :])
        valid = depth[:, None] >= arities[None, :]
        remaining = max_len - step - 1
        valid &= (after >= 1) & (after <= 1 + 2 * remaining)
        valid[:, EOS] = depth == 1
        valid[done] = False
        valid[done, EOS] = True
        dist = Categorical(logits=logits.masked_fill(~valid, -torch.inf))
        action = dist.sample()
        sums = sums + dist.log_prob(action) * (~done)
        seqs.append(action)
        new_done = done | (action == EOS)
        depth = torch.where(new_done, depth, depth + torch.where(action < len(FEATURES), 1, 1 - arities[action]))
        done = new_done
        inp = torch.cat([inp, action[:, None]], dim=1)
    tokens = torch.stack(seqs, dim=1).detach().cpu().tolist()
    tokens = [tuple(s[:s.index(EOS)] if EOS in s else s) for s in tokens]
    return tokens, sums


def search(features, bt, cfg, progress=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu': torch.set_num_threads(min(4, torch.get_num_threads()))
    cache, history = {}, []
    # Seed the archive with interpretable single-feature baselines; learned
    # candidates must compete against them under the same costs and selection.
    for i in range(len(FEATURES)):
        for formula in [(i,), (i, len(FEATURES) + 4)]:
            signal = execute(formula, features)
            cache[formula] = bt.evaluate(signal, 'train', len(formula))
    for seed in cfg.seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        model = make_generator(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
        for step in range(cfg.steps):
            seqs, logp = sample_formulas(model, cfg.batch_size, cfg.max_len, device)
            values = []
            for formula in seqs:
                if formula not in cache:
                    signal = execute(formula, features)
                    cache[formula] = bt.evaluate(signal, 'train', len(formula))
                values.append(cache[formula]['score'])
            rewards = torch.tensor(values, dtype=torch.float32, device=device)
            adv = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-6)
            loss = -(logp * adv.detach()).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append({'seed': seed, 'step': step, 'mean_reward': float(rewards.mean()),
                            'best_train_score': max(x['score'] for x in cache.values()),
                            'unique_formulas': len(cache)})
            if progress:
                progress((len(history) / (cfg.steps * len(cfg.seeds))),
                         f'Seed {seed} · step {step+1}/{cfg.steps} · {len(cache)} formulas')
            if step % 10 == 0 or step == cfg.steps - 1:
                print(f'seed={seed} step={step+1}/{cfg.steps} formulas={len(cache)} '
                      f'best_train={history[-1]["best_train_score"]:.3f}', flush=True)
        del model, optimizer
        if device.type == 'cuda': torch.cuda.empty_cache()
    ranked = sorted((f for f in cache if cache[f]['eligible']),
                    key=lambda f: (-cache[f]['score'], len(f), f))[:cfg.top_k]
    if not ranked: raise RuntimeError('No eligible training formula; inspect data/search configuration')
    candidates = []
    for formula in ranked:
        val = bt.evaluate(execute(formula, features), 'validation', len(formula))
        if val['eligible']:
            candidates.append({'tokens': list(formula), 'formula': decode(formula),
                               'train': cache[formula], 'validation': val})
    if not candidates: raise RuntimeError('No eligible validation formula; no best formula exported')
    candidates.sort(key=lambda x: (-x['validation']['score'], len(x['tokens']), x['tokens']))
    return candidates[0], candidates, history, device.type


def run(cfg, output_dir='btc_results', candles_csv=None, funding_csv=None, progress=None):
    cfg.validate()
    started = time.time()
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    # Each result directory is a frozen run; never mix old success files with a failed retry.
    if (out / 'best_btc_formula.json').exists():
        raise ValueError('Output already contains a completed run. Choose a new --output directory.')
    (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    if progress: progress(0, 'Fetching and validating Gate candles / funding')
    if candles_csv or funding_csv:
        if not candles_csv or not funding_csv: raise ValueError('Both candle and funding CSV paths required')
        df = validate_candles(pd.read_csv(candles_csv), cfg)
        funding = validate_funding(pd.read_csv(funding_csv), cfg)
        inputs = [Path(candles_csv), Path(funding_csv)]
    else:
        df, funding, inputs = fetch_gate(cfg, out / 'data')
    features, warmup = make_features(df, cfg.norm_window)
    warmup += cfg.max_len * 20  # conservative nested temporal-operator warmup
    bt = Backtester(df, funding, cfg, warmup)
    best, candidates, history, device = search(features, bt, cfg, progress=progress)
    # The test split is accessed only after selection has finished.
    tested = bt.evaluate(execute(best['tokens'], features), 'test', len(best['tokens']), positions=True)
    series = tested.pop('series')
    best['test'] = tested
    baselines = {}
    baseline_series = {}
    a, b = bt.splits['test']
    for name, p in [('long_1x', np.ones(b-a)), ('short_1x', -np.ones(b-a)), ('cash', np.zeros(b-a))]:
        evaluated = bt.evaluate_positions(p, a, b, include_series=True)
        baseline_series[name] = evaluated.pop('series')['equity'].to_numpy()
        baselines[name] = evaluated
    best.update({'schema_version': 1, 'market': 'Gate USDT perpetual',
        'contract': cfg.contract, 'interval': cfg.interval, 'selection': 'train top-k, then validation score; test untouched',
        'config': asdict(cfg), 'vocabulary': TOKENS,
        'feature_contract': 'causal OHLCV, rolling median / exact rolling MAD, clipped [-5,5]',
        'operator_contract': 'BTC VM v1; JUMP uses trailing 20-bar zscore; DIV sign-preserving epsilon; outputs clipped +/-1e6',
        'signal_rule': f'tanh(formula) > {cfg.signal_threshold}: long; < -{cfg.signal_threshold}: short; otherwise cash',
        'execution_rule': 'closed bar t -> open[t+1] entry -> open[t+2] mark/exit; 1x target notional; funding at exit belongs to incumbent position',
        'limitations': ['hourly OHLC fills and approximate notional resizing, no order book/mark-price liquidation model',
                        'validation selection is susceptible to multiple testing; test is descriptive, not proof of profitability',
                        'BTC vocabulary is incompatible with the original Solana strategy runner'],
        'test_baselines': baselines,
        'quality_flag': 'positive_test_return' if tested['total_return'] > 0 else 'test_nonpositive_do_not_treat_as_profitable',
        'splits': {k: {'signal_start_utc': str(pd.Timestamp(df.time.iloc[a], unit='s', tz='UTC')),
                      'last_signal_utc': str(pd.Timestamp(df.time.iloc[b-1], unit='s', tz='UTC')),
                      'bars': b-a} for k, (a,b) in bt.splits.items()}})
    (out / 'best_btc_formula.json').write_text(json.dumps(best, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    summary = '\n'.join([f'Formula: {best["formula"]}', f'Tokens: {best["tokens"]}', best['signal_rule'],
        best['execution_rule'], f'Validation: {best["validation"]}', f'Test: {best["test"]}',
        f'Quality: {best["quality_flag"]}', 'Best means best observed validation candidate, not a global optimum.'])
    (out / 'best_btc_formula.txt').write_text(summary, encoding='utf-8')
    pd.DataFrame([{'formula': x['formula'], 'tokens': json.dumps(x['tokens']),
                  **{f'train_{k}': v for k,v in x['train'].items()},
                  **{f'validation_{k}': v for k,v in x['validation'].items()}} for x in candidates]).to_csv(out / 'candidates.csv', index=False)
    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    series.to_csv(out / 'test_equity.csv', index=False)
    pd.DataFrame({'exit_time': series.exit_time, **baseline_series}).to_csv(out / 'test_baselines.csv', index=False)
    pd.DataFrame(features.T, columns=FEATURES).assign(time=df.time.to_numpy()).to_csv(out / 'features.csv', index=False)
    # Snapshot inputs even when supplied by upload, so downloaded runs are reproducible.
    data = out / 'data'; data.mkdir(exist_ok=True)
    df.to_csv(data / 'candles.csv', index=False)
    funding.to_csv(data / 'funding.csv', index=False)
    inputs = [data / 'candles.csv', data / 'funding.csv']
    manifest = {'config': asdict(cfg), 'device': device, 'runtime_seconds': time.time()-started,
        'python': sys.version, 'versions': {p: importlib.metadata.version(p) for p in ['torch', 'numpy', 'pandas', 'requests']},
        'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__)] + [Path(__file__).resolve().parents[1] / 'model_core' / n
                for n in ['alphagpt.py','config.py','vocab.py','ops.py']]}}
    (out / 'reproduction.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    if cfg.charts:
        if progress: progress(1, 'Exporting figures and interactive report')
        from .charts import render_results
        render_results(out)
        manifest['versions'].update({p:importlib.metadata.version(p) for p in ['matplotlib','plotly','pillow']})
        for chart_source in [Path(__file__).with_name('charts.py')] + list(Path(__file__).with_name('figure_tools').glob('*.py')):
            manifest['code_sha256'][str(chart_source)] = hashlib.sha256(chart_source.read_bytes()).hexdigest()
        manifest['runtime_seconds'] = time.time()-started
        (out/'reproduction.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(summary, flush=True)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON Config dictionary')
    parser.add_argument('--output', default='btc_results')
    parser.add_argument('--candles-csv'); parser.add_argument('--funding-csv')
    parser.add_argument('--steps', '--epochs', type=int, help='Policy-gradient steps per seed (not data epochs)')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seeds', help='Comma-separated integer seeds, e.g. 42,43,44')
    parser.add_argument('--start'); parser.add_argument('--end')
    parser.add_argument('--interval', choices=SECONDS)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--max-len', type=int)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--min-entries', type=int)
    parser.add_argument('--norm-window', type=int)
    parser.add_argument('--fee-bps', type=float); parser.add_argument('--slippage-bps', type=float)
    parser.add_argument('--signal-threshold', type=float)
    parser.add_argument('--train-fraction', type=float); parser.add_argument('--validation-fraction', type=float)
    parser.add_argument('--no-charts', action='store_true')
    parser.add_argument('--render-only', action='store_true', help='Regenerate charts for an existing output; no search or reselection')
    args = parser.parse_args()
    if args.render_only:
        from .charts import render_results
        render_results(args.output)
        manifest_path = Path(args.output)/'reproduction.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            sources = [Path(__file__).with_name('charts.py')] + list(Path(__file__).with_name('figure_tools').glob('*.py'))
            manifest['render_metadata'] = {
                'versions': {p:importlib.metadata.version(p) for p in ['matplotlib','plotly','pillow']},
                'code_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
            manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
        return
    cfg = Config(**json.loads(Path(args.config).read_text(encoding='utf-8'))) if args.config else Config()
    for name in ['steps','batch_size','start','end','interval','learning_rate','max_len','top_k',
                 'min_entries','norm_window','fee_bps','slippage_bps','signal_threshold',
                 'train_fraction','validation_fraction']:
        if getattr(args, name) is not None: setattr(cfg, name, getattr(args, name))
    if args.seeds: cfg.seeds = tuple(int(s.strip()) for s in args.seeds.split(','))
    if args.no_charts: cfg.charts = False
    run(cfg, args.output, args.candles_csv, args.funding_csv)


if __name__ == '__main__': main()
