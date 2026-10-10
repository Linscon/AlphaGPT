"""A small notebook that clones the user's fork and starts CLI or WebUI."""
import json
from pathlib import Path

def build():
    cells=[]
    def add(kind,s):
        c={'cell_type':kind,'id':f'fork-{len(cells)}','metadata':{},'source':s.splitlines(True)}
        if kind=='code': c.update(execution_count=None,outputs=[])
        cells.append(c)
    add('markdown','''# AlphaGPT BTC · Fork → Colab → WebUI
在运行时设置中选择 T4 GPU。默认克隆 Linscon/AlphaGPT 的 BTC 开发分支。
公开 Gate BTC_USDT 永续行情，无需 API Key。WebUI 临时分享链接的访问者可使用此研究界面。
''')
    add('code','''import subprocess, sys
from pathlib import Path
REPO = "https://github.com/Linscon/AlphaGPT.git"
BRANCH = "codex/btc-1h-mining"
ROOT = Path("/content/AlphaGPT")
if not ROOT.exists():
    subprocess.run(["git","clone","--branch",BRANCH,"--single-branch",REPO,str(ROOT)],check=True)
else:
    print("Using existing checkout; verify its branch and commit:")
subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True)
subprocess.run([sys.executable,"-m","pip","install","-q","-r",str(ROOT/"requirements-btc.txt")],check=True)
sys.path.insert(0,str(ROOT))
import os, torch
os.chdir(ROOT)
print("GPU available:",torch.cuda.is_available())
''')
    add('markdown','''## 选择运行方式
`MODE="webui"` 打开参数面板，或改为 `"cli"` 直接搜索。
WebUI 可以自定义步数、公式批大小、随机种子、日期、学习率、阈值、交易成本及验证比例。
''')
    add('code','''MODE = "webui"
STEPS = 200
BATCH_SIZE = 128
SEEDS = "42,43,44"
if MODE == "webui":
    from btc_mining.webui import build_ui
    app = build_ui()
    app.launch(share=True, inline=True)
elif MODE == "cli":
    subprocess.run([sys.executable,"mine_btc.py","--steps",str(STEPS),"--batch-size",str(BATCH_SIZE),
                    "--seeds",SEEDS,"--output","btc_results"],check=True)
else:
    raise ValueError("MODE must be webui or cli")
''')
    add('markdown','''## CLI 结果查看与下载（仅 CLI 模式）
WebUI 内已有图表、公式和下载按钮。测试净值为负也会如实显示；最佳不等于盈利。
''')
    add('code','''if MODE == "cli":
    from IPython.display import display, HTML
    from btc_mining.charts import interactive_figures
    for figure in interactive_figures("btc_results"): display(figure)
    print(Path("btc_results/best_btc_formula.txt").read_text())
    import shutil
    from google.colab import files
    files.download(shutil.make_archive("btc_results", "zip", "btc_results"))
''')
    doc={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},
        'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':'AlphaGPT_BTC_Fork.ipynb'}}}
    path=Path(__file__).parent/'AlphaGPT_BTC_Fork.ipynb'
    path.write_text(json.dumps(doc,indent=1,ensure_ascii=False),encoding='utf-8')
    print(path)

if __name__=='__main__': build()
