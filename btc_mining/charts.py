"""Nine evidence-backed figures and a portable interactive HTML report."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .figure_tools.setup_style import setup_style
from .figure_tools.export_figure import export_figure
from .figure_tools.visual_qa import audit_layout

BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'


def load_results(directory):
    root = Path(directory)
    return (json.loads((root/'best_btc_formula.json').read_text(encoding='utf-8')),
            pd.read_csv(root/'test_equity.csv'), pd.read_csv(root/'test_baselines.csv'),
            pd.read_csv(root/'training_history.csv'), pd.read_csv(root/'candidates.csv'))


def interactive_figures(directory):
    best, series, baselines, history, _ = load_results(directory)
    dates = pd.to_datetime(series.exit_time, unit='s', utc=True)
    equity = go.Figure()
    for name, values, color, dash in [('Strategy', series.equity, BLUE, 'solid'),
        ('Long 1x', baselines.long_1x, ORANGE, 'dash'), ('Short 1x', baselines.short_1x, GREEN, 'dot')]:
        equity.add_trace(go.Scatter(x=dates, y=values, name=name, line=dict(color=color, dash=dash)))
    equity.add_hline(y=1, line_dash='dot', line_color='#888888')
    equity.update_layout(title='Untouched test equity · after costs and funding', yaxis_title='Equity (initial = 1)')
    peaks = np.maximum.accumulate(np.r_[1.0, series.equity])[1:]
    drawdown = go.Figure(go.Scatter(x=dates, y=1-series.equity/peaks, fill='tozeroy',
                                  line_color=ORANGE, name='Drawdown'))
    drawdown.update_layout(title='Test drawdown from running peak', yaxis_title='Drawdown', yaxis_tickformat='.1%')
    training = go.Figure()
    for seed, frame in history.groupby('seed', sort=False):
        training.add_trace(go.Scatter(x=frame.step+1, y=frame.mean_reward, name=f'Seed {seed} · mean reward'))
    training.update_layout(title='Training reward by seed', xaxis_title='Policy update step', yaxis_title='Mean train score')
    for fig in [equity, drawdown, training]:
        fig.update_layout(template='plotly_white', height=380, margin=dict(l=50,r=20,t=55,b=45),
                          legend=dict(orientation='h', y=-0.2))
    return equity, drawdown, training


def render_results(directory):
    root = Path(directory); figures = root/'figures'; figures.mkdir(exist_ok=True)
    previews = figures/'previews'; previews.mkdir(exist_ok=True)
    # Keep raster-only QA previews separate from the nine PNG/SVG deliverables.
    for previous in figures.glob('*_grayscale.png'):
        previous.replace(previews/previous.name)
    best, series, baselines, history, candidates = load_results(root)
    candles = pd.read_csv(root/'data/candles.csv')
    features = pd.read_csv(root/'features.csv')
    warmup = int(np.searchsorted(candles.time, int(pd.Timestamp(best['splits']['train']['signal_start_utc']).timestamp())))
    setup_style(journal='general', lang='en', use_sciplots=False)
    plt.rcParams.update({'font.size':10, 'axes.titlesize':12, 'legend.fontsize':9})
    dates = pd.to_datetime(candles.time, unit='s', utc=True)
    test_dates = pd.to_datetime(series.exit_time, unit='s', utc=True)
    contracts, pngs = [], []

    def chart(name, claim, draw, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(8,4), layout='constrained')
        draw(ax)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.set_title(claim, loc='left', pad=12)
        ax.spines[['top','right']].set_visible(False)
        if xlabel == 'UTC':
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        issues = audit_layout(fig)
        if issues: raise RuntimeError(f'Figure layout audit: {name}: {issues}')
        export_figure(fig, str(figures/name), formats=['svg','png'], size_inches=(8,4), dpi=300,
                      grayscale_preview=False, tight=False)
        from PIL import Image
        with Image.open(figures/f'{name}.png') as im:
            im.convert('L').save(previews/f'{name}_grayscale.png', dpi=(300,300))
        plt.close(fig)
        pngs.append(str(figures/f'{name}.png'))
        contracts.append({'file':name, 'question':'q1', 'claim':claim, 'size_inches':[8,4],
                          'backend':'matplotlib', 'dpi':300, 'layout':'single quantitative panel',
                          'x':xlabel, 'y':ylabel, 'uncertainty':'descriptive observed series; no confidence intervals'})

    chart('raw_q1_price', 'BTC perpetual price and chronological splits', lambda ax: (
        ax.plot(dates, candles.close, color=BLUE, linewidth=0.9),
        [ax.axvline(pd.Timestamp(best['splits'][s]['signal_start_utc']), color=ORANGE, linestyle='--',
                    label=s.title()) for s in ['validation','test']], ax.legend()), 'UTC', 'Close (USDT)')
    chart('raw_q1_volume', 'Distribution of hourly USDT turnover',
          lambda ax: ax.hist(np.log10(candles.volume.clip(lower=1)), bins=45, color=BLUE),
          'log10(hourly turnover / USDT)', 'Hours')
    names = ['RET','MOM24','VOL_SHOCK','DEV24','PRESSURE','VOLATILITY24']
    chart('raw_q1_features', 'Causal normalized feature distributions',
          lambda ax: ax.boxplot([features[n].iloc[warmup:] for n in names], tick_labels=names,
                                showfliers=False), 'Feature (warmup excluded)', 'Normalized value')
    chart('process_q1_reward', 'Mean training reward by independent seed', lambda ax: (
        [ax.plot(f.step+1, f.mean_reward, label=str(s)) for s,f in history.groupby('seed',sort=False)],
        ax.legend(title='Seed')), 'Policy update step', 'Mean train score')
    chart('process_q1_archive', 'Best observed training score in search archive',
          lambda ax: ax.plot(np.arange(len(history))+1, history.best_train_score, color=BLUE),
          'Total policy update step', 'Best train score')
    chart('process_q1_candidates', 'Training shortlist compared on validation only', lambda ax: (
        ax.scatter(candidates.train_score, candidates.validation_score, color=BLUE),
        ax.scatter(candidates.iloc[0].train_score, candidates.iloc[0].validation_score,
                   marker='*', s=140, color=ORANGE, label='Selected'), ax.legend()),
          'Train score', 'Validation score')
    chart('result_q1_equity', 'Test equity after fees, slippage and funding', lambda ax: (
        ax.plot(test_dates, series.equity, color=BLUE, label='Strategy'),
        ax.plot(test_dates, baselines.long_1x, color=ORANGE, linestyle='--',label='Long 1x'),
        ax.plot(test_dates, baselines.short_1x, color=GREEN, linestyle=':',label='Short 1x'),
        ax.axhline(1, color='#777777', linewidth=0.7), ax.legend()), 'UTC', 'Equity (initial = 1)')
    peaks = np.maximum.accumulate(np.r_[1.0, series.equity])[1:]
    chart('result_q1_drawdown', 'Test drawdown includes the initial equity peak',
          lambda ax: ax.fill_between(test_dates, 0, 100*(1-series.equity/peaks), color=ORANGE, alpha=0.6),
          'UTC', 'Drawdown (%)')
    chart('result_q1_positions', 'Test exposure: long, cash or short', lambda ax: (
        ax.step(pd.to_datetime(series.entry_time,unit='s',utc=True), series.position,where='post',color=BLUE),
        ax.set_yticks([-1,0,1], ['Short','Cash','Long'])), 'UTC', 'Direction (target 1x)')
    (figures/'contracts.json').write_text(json.dumps(contracts,indent=2),encoding='utf-8')
    plots = interactive_figures(root)
    report = '<!doctype html><html><meta charset="utf-8"><title>AlphaGPT BTC report</title><body style="max-width:1100px;margin:auto;font-family:Arial">'
    report += '<h1>AlphaGPT BTC · research results</h1><pre>'+html.escape(best['formula'])+'</pre>'
    report += '<p>Selection uses validation only. Test return: '+f'{best["test"]["total_return"]:.2%}'+'. Best observed candidate does not guarantee profitability.</p>'
    for i, fig in enumerate(plots): report += fig.to_html(full_html=False, include_plotlyjs=True if i==0 else False)
    report += ''.join(f'<img src="figures/{Path(p).name}" alt="{Path(p).stem}" style="width:100%">' for p in pngs)
    (root/'report.html').write_text(report+'</body></html>',encoding='utf-8')
    return pngs
