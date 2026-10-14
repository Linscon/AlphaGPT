"""Build the ready-to-run Fork/Colab notebook (no local paths or saved outputs)."""
import json
from pathlib import Path


def build():
    cells = []
    def add(kind, source):
        cell = {'cell_type':kind, 'id':f'btc-colab-{len(cells):02d}',
                'metadata':{}, 'source':source.splitlines(True)}
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add('markdown', '''# AlphaGPT · BTC 1h 多空因子挖掘
直接依次执行下面的单元格。推荐先在 **运行时 → 更改运行时类型 → T4 GPU** 中选择 GPU；CPU 也能运行。

- 默认打开 WebUI，在面板里设置训练步数并点击「开始因子挖掘」。
- 或把第 4 步的 `MODE` 改成 `"cli"`，用 Notebook 参数直接搜索。
- 数据：Gate BTC_USDT USDT 永续合约，1 小时；公共行情无需 API Key。
- 输出：验证集选出的最佳公式、独立测试结果、9 张图、离线交互报告和 ZIP。
- 「最佳」是本次候选中的验证集最佳。测试收益可能为负，结果会如实显示。

WebUI 使用 Gradio 临时公开链接，访问者可操作界面；Colab 会话结束后服务停止。此项目仅研究，不会下单。
''')
    add('markdown', '## 1. 克隆你的 Fork\n重复执行会检查仓库和分支，并仅快进更新；不会覆盖本地修改。')
    add('code', '''import os, sys, subprocess
from pathlib import Path
REPO = "https://github.com/Linscon/AlphaGPT.git"
BRANCH = "codex/btc-1h-mining"
ROOT = Path("/content/AlphaGPT")
def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
if not ROOT.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    "--single-branch", REPO, str(ROOT)], check=True)
else:
    if not (ROOT / ".git").is_dir():
        raise RuntimeError(f"{ROOT} 已存在但不是 Git 仓库，请修改 ROOT 为新的目录。")
    if git("remote", "get-url", "origin").removesuffix(".git") != REPO.removesuffix(".git"):
        raise RuntimeError("ROOT 中是其他仓库，请修改 ROOT 为新的目录。")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("ROOT 中不是 BTC 分支，请修改 ROOT 为新的目录。")
    if git("status", "--porcelain", "--untracked-files=no"):
        print("检测到本地修改，保留现有版本。")
    else:
        subprocess.run(["git", "fetch", "origin", BRANCH], cwd=ROOT, check=True)
        subprocess.run(["git", "merge", "--ff-only", "FETCH_HEAD"], cwd=ROOT, check=True)
os.chdir(ROOT)
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
print("工作目录:", ROOT, "提交:", git("rev-parse", "--short", "HEAD"))
''')
    add('markdown', '## 2. 安装依赖\n保留 Colab 自带的 GPU PyTorch。如果安装后提示必须重启会话，重启后从第 1 步重新执行。')
    add('code', '''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(ROOT / "requirements-btc.txt")], check=True)
print("依赖安装完成。")
''')
    add('markdown', '## 3. 检查 GPU 和 Gate 网络\n若 Gate 在当前 Colab 区域不可访问，可在 WebUI 上传行情与资金费率 CSV；格式见仓库 BTC_README.md。')
    add('code', '''import torch, requests, importlib.metadata
print("PyTorch:", torch.__version__, "GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU 型号:", torch.cuda.get_device_name(0))
    x = torch.ones((8, 8), device="cuda")
    print("GPU 运算:", (x @ x)[0, 0].item())
else:
    print("当前使用 CPU。建议先跑 20 步、batch 32、一个种子。")
for package in ["numpy", "pandas", "gradio", "matplotlib", "plotly"]:
    print(package, importlib.metadata.version(package))
try:
    response = requests.get("https://api.gateio.ws/api/v4/futures/usdt/contracts/BTC_USDT", timeout=30)
    response.raise_for_status()
    print("Gate 连通:", response.json()["name"])
except (requests.RequestException, ValueError, KeyError) as error:
    print("Gate 网络检查未通过:", str(error))
    print("可尝试重新连接 Colab，或使用 WebUI 的 CSV 上传入口。")
''')
    add('markdown', '''## 4. 选择模式和参数
默认 `MODE="webui"`：在 **第 5 步的 WebUI 面板** 修改参数，下面的搜索参数仅用于 CLI 模式。

CLI 默认先进行 20 步、一个种子的快速运行。检查通过后可设 `STEPS=200`、`BATCH_SIZE=128`、`SEEDS="42,43,44"`。
步数指每个种子的策略梯度更新次数。日期为 UTC，结束日期不含；默认最近 170 天以适配资金费率接口历史范围。
''')
    add('code', '''from datetime import datetime, timedelta, timezone
import uuid
MODE = "webui"  # "webui" 或 "cli"
STEPS = 20
BATCH_SIZE = 32
SEEDS = "42"
END = datetime.now(timezone.utc).date().isoformat()
START = (datetime.now(timezone.utc).date() - timedelta(days=170)).isoformat()
FEE_BPS = 5.0       # 单边手续费；1 bps = 0.01%
SLIPPAGE_BPS = 2.0  # 单边滑点
SIGNAL_THRESHOLD = 0.2
LEARNING_RATE = 0.0003
MIN_ENTRIES = 10
OUTPUT = ROOT / "btc_runs" / ("colab_" + uuid.uuid4().hex[:12])
if MODE not in {"webui", "cli"}: raise ValueError("MODE 只能是 webui 或 cli")
print("模式:", MODE, "CLI 数据范围:", START, "至", END)
print("CLI 结果目录:", OUTPUT)
''')
    add('markdown', '''## 5. 开始运行
**WebUI**：等待面板或 `gradio.live` 链接出现，再点击「开始因子挖掘」。重复运行本单元会关闭上一次面板。

**CLI**：等待搜索完成，继续第 6 步。每次重新执行第 4 步会创建新的结果目录，保留旧结果。
若没有符合最低入场次数的候选，增加搜索步数/种子，或在明确接受更少交易样本时调低 `MIN_ENTRIES`。
''')
    add('code', '''if MODE == "webui":
    if "app" in globals(): app.close()
    from btc_mining.webui import build_ui
    app = build_ui()
    app.launch(share=True, inline=True, prevent_thread_lock=True)
else:
    subprocess.run([sys.executable, str(ROOT / "mine_btc.py"),
        "--steps", str(STEPS), "--batch-size", str(BATCH_SIZE), "--seeds", SEEDS,
        "--start", START, "--end", END, "--interval", "1h",
        "--fee-bps", str(FEE_BPS), "--slippage-bps", str(SLIPPAGE_BPS),
        "--signal-threshold", str(SIGNAL_THRESHOLD), "--learning-rate", str(LEARNING_RATE),
        "--min-entries", str(MIN_ENTRIES), "--output", str(OUTPUT)], cwd=ROOT, check=True)
''')
    add('markdown', '## 6. 查看结果和下载\nWebUI 内已有公式、图表和下载按钮。CLI 模式在此显示交互图表、9 张图片并下载完整 ZIP。')
    add('code', '''if MODE == "cli":
    from IPython.display import display, Image
    from btc_mining.charts import interactive_figures
    print((OUTPUT / "best_btc_formula.txt").read_text(encoding="utf-8"))
    for figure in interactive_figures(OUTPUT): display(figure)
    for path in sorted((OUTPUT / "figures").glob("*.png")):
        display(Image(filename=str(path), width=900))
    import shutil
    from google.colab import files
    archive = shutil.make_archive(str(OUTPUT), "zip", str(OUTPUT))
    files.download(archive)
else:
    print("请在上方 WebUI 点击下载完整结果 ZIP 或公式 JSON。")
''')
    add('markdown', '''### 图表生成失败时恢复（可选）
CLI 已保存公式和 CSV 后，可单独重画图，无须重新搜索。仅在需要时运行下面代码，再执行第 6 步。
''')
    add('code', '''# subprocess.run([sys.executable, str(ROOT / "mine_btc.py"),
#                 "--render-only", "--output", str(OUTPUT)], cwd=ROOT, check=True)
''')
    notebook = {'cells':cells, 'nbformat':4, 'nbformat_minor':5,
        'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},
                    'language_info':{'name':'python'}, 'accelerator':'GPU',
                    'colab':{'name':'AlphaGPT_BTC_Fork.ipynb'}}}
    path = Path(__file__).with_name('AlphaGPT_BTC_Fork.ipynb')
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
    print(path)


if __name__ == '__main__': build()
