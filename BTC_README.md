# BTC 1h 多空因子挖掘

## Fork 与 Colab

你的 Fork：https://github.com/Linscon/AlphaGPT 。BTC 改造在 `codex/btc-1h-mining` 分支。
把 `colab/AlphaGPT_BTC_Fork.ipynb` 上传到 Google Colab，选择 T4 GPU，依次运行单元。
它会克隆该分支、安装专用依赖并打开 WebUI。若合并到 main，Notebook 中 BRANCH 也可改为 main。

手动克隆和运行：

```python
!git clone -b codex/btc-1h-mining --single-branch https://github.com/Linscon/AlphaGPT.git
%cd AlphaGPT
%pip install -r requirements-btc.txt
!python webui.py --share
```

Gradio 在 Colab 生成临时分享链接，访问者可以操作此界面，Colab 会话终止后服务停止。
公开接口不需要 API Key；项目不会下单。使用说明：https://www.gradio.app/guides/sharing-your-app 。

## 命令行参数

```bash
python mine_btc.py --steps 200 --batch-size 128 --seeds 42,43,44 --output btc_results
python mine_btc.py --steps 20 --batch-size 32 --seeds 42 --output quick_run
python webui.py                 # 本地 localhost:7860
python webui.py --share         # Colab 临时分享
python mine_btc.py --render-only --output btc_results  # 仅重新生成图表
```

训练“轮数”对应每个种子的策略梯度更新步数；每步生成 batch-size 个公式，不是遍历行情数据的 epoch。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--steps` / `--epochs` | 200 | 每种子更新步数 |
| `--batch-size` | 128 | 每步候选公式数 |
| `--seeds` | 42,43,44 | 独立随机初始化 |
| `--start`, `--end` | 最近 170 天 | UTC；结束排他 |
| `--learning-rate` | 0.0003 | AdamW 学习率 |
| `--max-len` | 12 | 公式最大 token 数 |
| `--top-k` | 30 | 训练集进入验证集的候选数量 |
| `--min-entries` | 10 | 训练/验证最少入场次数 |
| `--norm-window` | 168 | 因果归一化窗口 |
| `--fee-bps`, `--slippage-bps` | 5、2 | 单边成本假设 |
| `--signal-threshold` | 0.2 | tanh 因子多空阈值 |
| `--train-fraction`, `--validation-fraction` | 0.6、0.2 | 时间切分比例 |
| `--config` | 无 | JSON 配置；命令行参数覆盖配置值 |
| `--candles-csv`, `--funding-csv` | 无 | 同时提供历史数据 |
| `--no-charts` | 关闭 | 跳过图表，只导出研究结果 |

内置 WebUI 固定 BTC_USDT/1h，暴露主要搜索参数、成本、时间切分和 CSV 上传。
完成的输出目录不会被覆盖；重新运行请使用新的 `--output`。WebUI 自动为每次运行创建独立目录，训练任务串行执行。

## 数据、信号和评估

六个 BTC 因子是 RET、MOM24、VOL_SHOCK、DEV24、PRESSURE、VOLATILITY24，定义详见 [因子契约](colab/README.md#模型与公式定义)。
复用原仓库 AlphaGPT 循环 Transformer，增加后缀栈约束和 EOS。
只使用当前与过去闭合 K 线；滚动 median/MAD 与 JUMP 都不使用未来数据。

`tanh(formula[t]) > threshold` 做多；小于负阈值做空；其余空仓。
闭合 t 信号对应 `open[t+1] → open[t+2]`，目标名义敞口 1 倍，计真实资金费率及换仓费用。
仓位重置与交易费用是近似模型，不模拟盘口、标记价格强平与 ADL。

训练奖励不使用验证/测试；训练前 K 个合格候选按验证评分选一个，最后只评估一次测试。
排名分数为净收益年化 Sharpe（裁剪 ±10）减 `2*最大回撤` 和 `0.005*公式长度`。
输出有固定做多/做空/空仓基准；测试亏损也如实保留，重复依据同一测试期调参会损坏其独立性。

Gate 资金费率历史实测只支持最近 180 天，默认留出余量取 170 天。
更长范围同时上传：K 线 `time,open,high,low,close,volume`（USDT 成交额），资金费率 `time,rate`。
time 为 Unix 秒。缺失行情不会前向填充，缺失资金费率不会假设为零。

## 输出与图表

`best_btc_formula.json` / `.txt` 保存公式、token、信号规则、切分、成本与评估。
`candidates.csv`、`training_history.csv`、`test_equity.csv`、`test_baselines.csv` 保留检查明细。
`config.json`、`reproduction.json` 和 `data/` 保存运行参数、哈希、依赖版本与输入快照。

自动生成 9 幅 300 DPI PNG 和可编辑 SVG（另有灰度预览）：

- 原始数据：BTC 价格与切分、小时成交额分布、归一化因子分布。
- 搜索过程：每种子平均奖励、档案最佳训练分数、训练/验证候选散点。
- 测试结果：策略与多空基准净值、回撤、多空持仓方向。

`report.html` 是含交互净值、回撤、训练曲线的离线报告。WebUI 直接显示交互图表、全部图像和下载按钮。
BTC token 与原 Solana runner 不兼容，只通过本模块复算公式。

测试：`python -m unittest discover -s colab -p test_btc_colab.py`。
本地无 GPU 时安装 CPU PyTorch，再安装 requirements-btc.txt；Colab 保留自带 GPU PyTorch。
