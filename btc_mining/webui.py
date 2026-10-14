"""Gradio controls for the same BTC research pipeline used by the CLI."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import uuid

import gradio as gr

from .core import Config, run
from .charts import interactive_figures


def ui_config(start, end, steps, batch, seeds, lr, max_len, top_k, min_entries,
              fee, slip, threshold, norm, train, validation):
    import math
    for value in [steps,batch,max_len,top_k,min_entries,norm]:
        if not math.isfinite(float(value)) or not float(value).is_integer():
            raise ValueError('训练步数、批大小、窗口与候选数量必须为整数')
    cfg = Config(start=start.strip(),end=end.strip(),steps=int(steps),batch_size=int(batch),
        seeds=tuple(int(s.strip()) for s in seeds.split(',')),learning_rate=float(lr),
        max_len=int(max_len),top_k=int(top_k),min_entries=int(min_entries),fee_bps=float(fee),
        slippage_bps=float(slip),signal_threshold=float(threshold),norm_window=int(norm),
        train_fraction=float(train),validation_fraction=float(validation))
    cfg.validate()
    return cfg


def train_ui(*values, progress=gr.Progress()):
    *params, candle_csv, funding_csv = values
    try:
        cfg = ui_config(*params)
        output = Path('btc_runs') / uuid.uuid4().hex[:12]
        best = run(cfg, output, candle_csv, funding_csv,
                   progress=lambda p, text: progress(p, desc=text))
        rows = ['| Split | Net return | Sharpe | Max drawdown | Entries |', '|---|---:|---:|---:|---:|']
        for name in ['train','validation','test']:
            x=best[name]
            rows.append(f'| {name} | {x["total_return"]:.2%} | {x["sharpe"]:.2f} | {x["max_drawdown"]:.2%} | {x["entries"]} |')
        status = '测试收益为负，未通过样本外盈利验证。' if best['test']['total_return'] <= 0 else '测试收益为正；这是本次搜索结果，不保证未来收益。'
        plots = interactive_figures(output)
        gallery = [(str(p), p.stem) for p in sorted((output/'figures').glob('*.png')) if 'grayscale' not in p.name]
        archive = shutil.make_archive(str(output.resolve()),'zip',output)
        return best['formula'], '\n'.join(rows), status, *plots, gallery, [archive, str(output/'best_btc_formula.json')]
    except (ValueError, RuntimeError, OSError) as exc:
        raise gr.Error(str(exc)) from exc


def build_ui():
    defaults = Config()
    with gr.Blocks(title='AlphaGPT BTC Factor Lab', theme=gr.themes.Soft(primary_hue='blue')) as app:
        gr.Markdown('# AlphaGPT BTC Factor Lab\nGate BTC_USDT 永续 · 1h · 多空因子挖掘 · 1 倍目标敞口')
        gr.Markdown('训练集搜索 → 验证集选择 → 测试集评估。公式、图表与数据可下载；所有日期为 UTC。')
        with gr.Row():
            with gr.Column(scale=1):
                start=gr.Textbox(label='开始日期 UTC',value=defaults.start)
                end=gr.Textbox(label='结束日期 UTC（不包含）',value=defaults.end)
                steps=gr.Slider(1,2000,value=200,step=1,label='每个种子的训练步数',info='策略梯度更新次数')
                batch=gr.Dropdown([8,16,32,64,128,256,512],value=128,label='每步公式数量')
                seeds=gr.Textbox(value='42,43,44',label='随机种子',info='用英文逗号分隔')
                with gr.Accordion('搜索与交易成本',open=False):
                    lr=gr.Number(value=0.0003,label='学习率')
                    max_len=gr.Slider(2,12,value=12,step=1,label='最大公式长度')
                    top_k=gr.Number(value=30,precision=0,label='训练集候选数 Top K')
                    min_entries=gr.Number(value=10,precision=0,label='最少入场次数')
                    fee=gr.Number(value=5,label='单边手续费 bps')
                    slip=gr.Number(value=2,label='单边滑点 bps')
                    threshold=gr.Slider(0,0.95,value=0.2,label='多空信号阈值')
                    norm=gr.Number(value=168,precision=0,label='滚动归一化窗口')
                    train=gr.Slider(0.3,0.8,value=0.6,step=0.05,label='训练集比例')
                    validation=gr.Slider(0.1,0.4,value=0.2,step=0.05,label='验证集比例')
                with gr.Accordion('导入历史 CSV（可选）',open=False):
                    gr.Markdown('超过 Gate 180 天资金费率范围时，同时上传 K 线和资金费率。列格式见 README。')
                    candle=gr.File(label='K 线 CSV',file_types=['.csv'],type='filepath')
                    funding=gr.File(label='资金费率 CSV',file_types=['.csv'],type='filepath')
                button=gr.Button('开始因子挖掘',variant='primary')
            with gr.Column(scale=3):
                status=gr.Markdown('等待运行。建议先用 20 步、一个种子检查环境。')
                formula=gr.Textbox(label='验证集最佳公式',lines=3,interactive=False)
                metrics=gr.Markdown()
                with gr.Tabs():
                    with gr.Tab('测试净值'): equity=gr.Plot()
                    with gr.Tab('测试回撤'): dd=gr.Plot()
                    with gr.Tab('训练过程'): training=gr.Plot()
                    with gr.Tab('全部图表'): gallery=gr.Gallery(columns=2,label='原始数据 / 搜索 / 测试结果')
                downloads=gr.Files(label='下载完整结果 ZIP 与公式 JSON')
        button.click(train_ui,inputs=[start,end,steps,batch,seeds,lr,max_len,top_k,min_entries,fee,slip,
                    threshold,norm,train,validation,candle,funding],
                    outputs=[formula,metrics,status,equity,dd,training,gallery,downloads],concurrency_limit=1,api_name=False)
    return app.queue(default_concurrency_limit=1,max_size=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--share',action='store_true',help='Create a temporary public Gradio link (for Colab)')
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    build_ui().launch(share=args.share,server_port=args.port)


if __name__=='__main__': main()
