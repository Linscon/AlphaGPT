"""Build a self-contained notebook from authoritative local sources (no git clone)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = {str(p.relative_to(ROOT)).replace('\\','/'): p.read_text(encoding='utf-8') for p in
             [ROOT/'model_core'/n for n in ['alphagpt.py','config.py','vocab.py','ops.py']] +
             [ROOT/'colab/btc_colab.py', ROOT/'colab/test_btc_colab.py', ROOT/'LICENSE'] +
             list((ROOT/'btc_mining').rglob('*.py'))}
    files = {k:v for k,v in files.items() if '.venv' not in k and '__pycache__' not in k}
    cells = []
    def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s): cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[], 'source':s.splitlines(True)})
    md('''# AlphaGPT × Gate BTC：最佳公式挖掘

依次运行所有单元。建议在「运行时 → 更改运行时类型」选择 T4 GPU。
本 Notebook 内嵌当前仓库的 AlphaGPT 生成器和 BTC 适配代码，无需克隆仓库。

**市场**：Gate BTC_USDT USDT 永续，1 小时，做多/做空/空仓，1 倍目标名义敞口。
公开行情与历史资金费率无需 API Key；没有交易或私钥代码。

最佳 = 训练集排名前 K 个公式中，验证集净收益风险评分最高的候选。
测试集只在选定公式后评估一次；负收益也如实导出，不宣称找到盈利策略。

Gate 接口文档：https://www.gate.com/docs/developers/apiv4/en/futures/
''')
    code('''%pip -q install "numpy>=1.26,<3" "pandas>=2.2,<3" "requests>=2.32,<3" "matplotlib>=3.9,<4" "plotly>=5.24,<7" "pillow>=10,<13"
import torch
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.is_available())
# 保留 Colab 自带的 GPU 版 PyTorch；不要安装仓库全部 Solana/数据库依赖。
''')
    md('## 1. 写入内嵌源码（可展开查看）')
    code('from pathlib import Path\nimport json, sys\nROOT = Path("/content/alphagpt_btc")\nROOT.mkdir(parents=True, exist_ok=True)\nFILES = '+repr(files)+'\nfor name, source in FILES.items():\n    path = ROOT / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(source, encoding="utf-8")\nsys.path.insert(0, str(ROOT / "colab"))\nsys.path.insert(0, str(ROOT))\nprint("Sources ready:", ROOT)\n')
    md('## 2. 离线正确性检查\n合成数据只用于测试，不会被当作 BTC 结果。')
    code('import subprocess\nsubprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "colab"), "-p", "test_btc_colab.py"], cwd=ROOT, check=True)\n')
    md('''## 3. 配置本次搜索

默认获取最近 170 天已结束的 K 线，日期和数据哈希会记录在结果中。
实测 Gate 资金费率接口拒绝超过 180 天的历史；更长范围需自行提供真实历史 CSV。
至少保留足够历史，使训练、验证、测试各有 100 根以上可交易 K 线。
默认 3 个种子，每个 200 步；先用 `steps=20, seeds=(42,)` 检查运行，再恢复默认正式搜索。
手续费与滑点是假设值，请按自己的账户和成交方式调整。重复看测试集再调参会破坏其独立性。
''')
    code('''from btc_colab import Config, run
cfg = Config(
    contract="BTC_USDT", interval="1h",
    # start/end 默认最近170天（UTC）；也可填写180天以内的固定日期。
    steps=200, batch_size=128, seeds=(42, 43, 44),
    fee_bps=5.0, slippage_bps=2.0,
    signal_threshold=0.2, top_k=30,
)
OUTPUT = ROOT / "btc_results"
cfg.validate()
print(cfg)
''')
    md('''## 4. 获取 Gate 数据并搜索

完整性检查失败会停止；不会用零资金费率或前向填充价格掩盖数据问题。
若 Colab 网络无法访问 Gate，可上传同市场的两个 CSV：
`time,open,high,low,close,volume`（volume 为 USDT 成交额）和 `time,rate`（真实资金费率记录）。
`time` 都为 Unix 秒，再传给 `run(..., candles_csv=..., funding_csv=...)`。
''')
    code('best = run(cfg, output_dir=OUTPUT)\n')
    md('## 5. 查看与下载最佳公式\n公式 JSON 同时包含特征、算子、信号、成本、时间切分与测试结果。BTC token 不能直接交给原 Solana runner。')
    code('''import pandas as pd
print((OUTPUT / "best_btc_formula.txt").read_text(encoding="utf-8"))
display(pd.DataFrame({k: best[k] for k in ["train", "validation", "test"]}).T)
display(pd.read_csv(OUTPUT / "candidates.csv").head(10))
''')
    code('''import shutil
from google.colab import files
archive = shutil.make_archive(str(ROOT / "btc_formula_results"), "zip", OUTPUT)
files.download(archive)
# 压缩包包括最佳公式 JSON/TXT、候选排名、训练日志、测试收益轨迹、原始数据与复现信息。
''')
    doc = {'cells':cells,'metadata':{'colab':{'name':'AlphaGPT_BTC_Gate.ipynb'},
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python'}, 'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
    for i, cell in enumerate(cells): cell['id'] = f'btc-{i:02d}'
    path = ROOT/'colab/AlphaGPT_BTC_Gate.ipynb'
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding='utf-8')
    print(path)


if __name__ == '__main__': build()
