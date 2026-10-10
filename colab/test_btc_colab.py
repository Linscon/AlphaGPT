"""Offline correctness and tiny end-to-end checks; synthetic results are not BTC evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from btc_colab import (Config, FEATURES, OPS, TOKENS, EOS, Backtester, decode,
                       execute, fetch_gate, make_features, make_generator, run,
                       sample_formulas, utc_seconds, validate_candles, validate_funding)


def fixture(n=1800):
    rng = np.random.default_rng(7)
    times = utc_seconds('2025-01-01') + np.arange(n) * 3600
    price = 90000 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, n)))
    close = price * np.exp(rng.normal(0, 0.001, n))
    df = pd.DataFrame({'time': times, 'open': price,
        'high': np.maximum(price, close) * 1.002, 'low': np.minimum(price, close) * 0.998,
        'close': close, 'volume': rng.lognormal(16, 0.5, n)})
    fund = pd.DataFrame({'time': times[::8], 'rate': 0.0001})
    cfg = Config(start='2025-01-01', end=str(pd.Timestamp(times[-1]+3600, unit='s', tz='UTC')),
                 steps=2, batch_size=8, seeds=(42,), norm_window=48, min_entries=1, top_k=10, charts=False)
    return df, fund, cfg


class CorrectnessTests(unittest.TestCase):
    def test_config_rejects_invalid_training_parameters(self):
        _, _, cfg = fixture()
        for name, value in [('steps', 1.5), ('batch_size', 1), ('learning_rate', 0),
                            ('fee_bps', float('nan')), ('train_fraction', 0.9)]:
            from dataclasses import replace
            with self.assertRaises(ValueError): replace(cfg, **{name:value}).validate()

    def test_completed_output_cannot_mix_two_runs(self):
        df, fund, cfg = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'best_btc_formula.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'completed run'):
                run(cfg,root)

    def test_future_changes_do_not_change_past_features_or_formulas(self):
        df, _, cfg = fixture()
        altered = df.copy()
        altered.loc[900:, ['open','high','low','close']] *= 3
        altered.loc[900:, 'volume'] *= 5
        a, _ = make_features(df, cfg.norm_window)
        b, _ = make_features(altered, cfg.norm_window)
        np.testing.assert_array_equal(a[:, :900], b[:, :900])
        for i, (_, arity) in enumerate(OPS):
            formula = [0, 1, 2][:arity] + [len(FEATURES)+i]
            np.testing.assert_array_equal(execute(formula, a)[:900], execute(formula, b)[:900])

    def test_prefix_matches_full_history(self):
        df, _, cfg = fixture()
        all_f, _ = make_features(df, cfg.norm_window)
        short_f, _ = make_features(df.iloc[:900], cfg.norm_window)
        np.testing.assert_array_equal(all_f[:, :900], short_f)

    def test_grid_rejects_missing_or_invalid_candles(self):
        df, _, cfg = fixture()
        self.assertEqual(len(validate_candles(df, cfg)), len(df))
        with self.assertRaises(ValueError): validate_candles(df.drop(index=3), cfg)
        df.loc[3, 'low'] = df.loc[3, 'high'] * 2
        with self.assertRaises(ValueError): validate_candles(df, cfg)

    def test_funding_coverage(self):
        _, funding, cfg = fixture()
        self.assertEqual(len(validate_funding(funding, cfg)), len(funding))
        with self.assertRaises(ValueError): validate_funding(funding.iloc[10:], cfg)

    def test_stack_and_protected_division(self):
        f = np.zeros((6, 30)); f[0] = 2; f[1] = 3
        self.assertEqual(decode([0,1,7]), 'SUB(RET, MOM24)')
        np.testing.assert_array_equal(execute([0,1,7], f), -np.ones(30))
        self.assertTrue(np.isfinite(execute([0,2,9], f)).all())
        with self.assertRaises(ValueError): execute([6], f)
        with self.assertRaises(ValueError): execute([0,1], f)

    def test_sampler_always_emits_valid_formulas_and_gradients(self):
        torch.set_num_threads(2); torch.manual_seed(5)
        model = make_generator('cpu')
        for length in [2, 5, 12]:
            tokens, logp = sample_formulas(model, 32, length, 'cpu')
            self.assertTrue(torch.isfinite(logp).all())
            for formula in tokens:
                self.assertTrue(np.isfinite(execute(formula, np.ones((6, 30)))).all())
                decode(formula)
            model.zero_grad(); logp.mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_execution_purge_funding_and_costs(self):
        df, funding, cfg = fixture()
        _, warmup = make_features(df, cfg.norm_window)
        bt = Backtester(df, funding, cfg, warmup)
        i = 100
        self.assertAlmostEqual(bt.ret[i], df.open.iloc[i+2]/df.open.iloc[i+1]-1)
        # A settlement at time index 104 is charged to signal index 102.
        self.assertAlmostEqual(bt.funding[102], 0.0001)
        self.assertEqual(bt.funding[103], 0)
        a, b = bt.splits['train']; c, d = bt.splits['validation']
        self.assertLess(b-1+2, c)
        a, b = bt.splits['test']
        bt.ret[:] = 0; bt.funding[:] = 0
        p = np.ones(b-a)
        result = bt.evaluate_positions(p, a, b, include_series=True)
        net = result['series'].net_return.to_numpy()
        fee = (cfg.fee_bps+cfg.slippage_bps)/10000
        self.assertAlmostEqual(net[0], -fee)
        self.assertLess(net[-1], -fee+1e-9)
        np.testing.assert_array_equal(bt.evaluate_positions(np.zeros(b-a), a,b, include_series=True)['series'].net_return, 0)
        bt.funding[a:b] = 0.001
        cfg.fee_bps = cfg.slippage_bps = 0
        long = bt.evaluate_positions(p,a,b,include_series=True)['series'].net_return
        short = bt.evaluate_positions(-p,a,b,include_series=True)['series'].net_return
        np.testing.assert_allclose(long, -0.001)
        np.testing.assert_allclose(short, 0.001)

    def test_gate_pagination_and_quote_volume_mapping(self):
        df, fund, cfg = fixture()
        calls = []
        def fake(session, url, params):
            calls.append((url, params))
            source = df if url.endswith('candlesticks') else fund
            part = source[(source.time >= params['from']) & (source.time <= params['to'])]
            if url.endswith('candlesticks'):
                self.assertNotIn('limit', params)
                self.assertLessEqual((params['to']-params['from'])//3600+1, 1000)
                return [{'t': int(x.time), 'o': str(x.open), 'h': str(x.high), 'l': str(x.low),
                         'c': str(x.close), 'sum': str(x.volume), 'v': '999'} for x in part.itertuples()]
            return [{'t':int(x.time), 'r':str(x.rate)} for x in part.itertuples()]
        with tempfile.TemporaryDirectory() as tmp, patch('btc_colab.get_json', fake), patch('btc_colab.time.sleep'), patch('btc_colab.time.time', return_value=utc_seconds(cfg.end)+3600):
            got, rates, _ = fetch_gate(cfg, tmp)
            np.testing.assert_allclose(got.volume, df.volume)
            self.assertEqual(len(rates), len(fund))
            count = len(calls)
            fetch_gate(cfg, tmp)
            self.assertEqual(len(calls), count)

    def test_end_to_end_export_and_test_is_not_used_for_selection(self):
        df, fund, cfg = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df.to_csv(root/'candles.csv', index=False); fund.to_csv(root/'funding.csv', index=False)
            events = []
            original = Backtester.evaluate
            def record(bt, signal, split, *args, **kwargs):
                events.append(split)
                return original(bt, signal, split, *args, **kwargs)
            with patch.object(Backtester, 'evaluate', record):
                best = run(cfg, root/'results', root/'candles.csv', root/'funding.csv')
            self.assertEqual(events[-1], 'test')
            self.assertEqual(events.count('test'), 1)
            saved = json.loads((root/'results/best_btc_formula.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['formula'], decode(saved['tokens']))
            self.assertEqual(best['test']['bars'], len(pd.read_csv(root/'results/test_equity.csv')))
            self.assertTrue((root/'results/reproduction.json').exists())


if __name__ == '__main__': unittest.main()
