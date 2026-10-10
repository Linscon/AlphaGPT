# 在 Google Colab 挖掘 BTC 最佳公式

打开 `AlphaGPT_BTC_Gate.ipynb`：在 https://colab.research.google.com/ 选择「文件 → 上传笔记本」，上传该文件，选择 T4 GPU，依次运行全部单元。

Notebook 自带本仓库的 AlphaGPT 源码和 BTC 适配程序，不需要 Git 克隆、数据库、API Key 或 Solana 依赖。没有下单功能。默认读取 Gate 的 `BTC_USDT` USDT 永续合约，一小时 K 线，允许做多、做空、空仓。

## 数据范围与成本

- 默认最近 170 天，UTC 当日零点为排他的结束时间，全部为已结束 K 线。
- 2026-10-10 实测 Gate 资金费率 API 拒绝超过 180 天的请求。更长时间需导入同市场的真实历史 K 线和资金费率 CSV，不会默认为零资金费率。
- K 线 `sum` 为 USDT 成交额，不能把合约张数 `v` 当成它。
- 默认单边手续费 5 bps、滑点 2 bps；这是可配置假设，不是对当前账户实际费率的断言。
- 使用实际资金费率记录；正费率多头支付、空头收取。资金费率只有成本用途，不作为提前可见特征。
- 缺失或不对齐的 K 线会报错；资金费率覆盖间隙超过 12 小时会报错。此检查不能证明所有资金结算记录均无缺失，请保留原始数据核查。

官方接口：https://www.gate.com/docs/developers/apiv4/en/futures/

## 模型与公式定义

复用 `model_core/alphagpt.py` 中原始 AlphaGPT 的循环 Transformer、RMSNorm、SwiGLU 和输出路由。只更换输入 embedding 与输出头，以支持 BTC 词表和 EOS。
原模型的 QK-Norm 没有接入前向路径，本适配不宣称修复或验证它；不启用 LoRD。

闭合 K 线 t 的基础因子为：

| token | 归一化前定义 |
|---|---|
| RET | `log(close[t]/close[t-1])` |
| MOM24 | `log(close[t]/close[t-24])` |
| VOL_SHOCK | `log((volume[t]+1)/(mean(volume[t-23:t])+1))` |
| DEV24 | `close[t]/mean(close[t-23:t])-1` |
| PRESSURE | `(close[t]-open[t])/max(high[t]-low[t],1e-9)` |
| VOLATILITY24 | 最近 24 根 RET 的总体标准差 |

所有因子均用最近 168 根（含当前闭合 K 线）的 median 与该窗口内精确 MAD 标准化，加 `1e-6` 后裁剪到 `[-5,5]`。前导特征暖机与嵌套时序算子暖机不计入训练收益。

保留 12 个算子名；`JUMP` 改为只用当前与过去 20 根的 zscore 超过 3 的部分。`DIV` 使用保留符号的 epsilon 保护；每个算子结果裁剪至 ±1e6。生成时使用后缀公式栈深约束和 EOS，最终只保留单个栈结果。

公式信号规则：

```text
tanh(formula[t]) > 0.2  → 下一根开盘做多
tanh(formula[t]) < -0.2 → 下一根开盘做空
其余                    → 下一根开盘空仓
```

这是研究用的 1 倍目标名义敞口模型。信号 t 对应 `open[t+1] → open[t+2]` 收益，允许下一时点续持或换仓。资金费率在结束边界归属于原有持仓。换方向收费双边，每个独立区间从空仓开始并在末端计平仓成本。同方向名义敞口重置费用作近似处理，换仓及末端费用按目标敞口估计；不是精确持仓数量账本，也不模拟盘口成交、标记价格强平或 ADL。

## 怎样定义“最佳”

按时间划分 60% 训练、20% 验证、20% 测试，并剔除跨边界的两根收益目标。

1. 默认三个种子，每种子 200 步，每步 128 个公式。
2. 策略梯度奖励只来自训练集；候选缓存避免重复回测。
3. 从训练集取前 30 个合格公式，按验证集评分选一个。
4. 选定后仅评估一次测试集，同时报告固定做多、固定做空和空仓基准。

评分为：`clip(净收益序列年化 Sharpe,-10,10) - 2*最大回撤 - 0.005*公式长度`。
不足 10 次入场的训练/验证候选不合格。Sharpe 按每年 365.25 天的小时数年化，均值/总体标准差，假设无风险收益为零。它只是排序统计量，不代表独立小时收益假设成立。

“最佳”仅指本次搜索得到的验证集最佳候选；不保证全局最优或盈利。重复查看测试集后调整参数，会使测试集失去独立性。负测试收益如实保留，公式 JSON 会给出质量标记。

## 导出内容

| 文件 | 内容 |
|---|---|
| `best_btc_formula.json` | 公式、token、词表、信号与执行规则、参数、时间切分及评估 |
| `best_btc_formula.txt` | 可直接阅读的公式与结果摘要 |
| `candidates.csv` | 候选训练/验证排名 |
| `training_history.csv` | 每个种子的训练记录 |
| `test_equity.csv` | 测试时点、方向、资金费率、净收益与权益轨迹 |
| `reproduction.json` | 数据/代码哈希、依赖、随机种子与运行时 |
| `data/*.csv` | 使用的 Gate 原始标准化数据 |

**BTC token 与原 Solana 特征语义不同，不能把此 JSON 直接交给原 `StrategyRunner`。** 复算时使用本目录的 `execute(tokens, features)` 和同一特征/信号契约。

本地运行：

```bash
python colab/btc_colab.py --config colab/smoke_config.json --output colab/btc_smoke_results
python -m unittest discover -s colab -p test_btc_colab.py
python colab/build_notebook.py
```

`smoke_config.json` 是本次真实小规模验证配置（每种子 30 步、每步 32 个），不是 Colab 默认完整搜索。
结果需与对应配置一起解释；离线单元测试使用合成数据，不能作为 BTC 收益证据。

## 本次小规模真实运行

数据范围：2026-04-23 至 2026-10-10 UTC（结束时间排他）；4,080 根小时 K 线、510 条资金费率。
共搜索 2,583 个不同公式。验证集选出的候选为：

```text
SIGN(GATE(SUB(JUMP(VOLATILITY24), VOLATILITY24), NEG(JUMP(DEV24)), SIGN(MAX3(MOM24))))
```

| 区间 | 净收益 | Sharpe | 最大回撤 |
|---|---:|---:|---:|
| 验证集 | +15.69% | 5.09 | 6.96% |
| 测试集 | -11.03% | -5.43 | 11.29% |

测试期固定做多基准为 +3.00%。该候选明显未通过样本外盈利验证，不应作为盈利策略推荐。
导出的 JSON/TXT 与明细位于 `btc_smoke_results/`；这是小规模运行的验证集最佳候选，不是默认完整 Colab 搜索的结果。
后续继续搜索时应保留新的独立测试区间，不能按已知测试期表现反复调参。

软件改编遵循仓库 Apache-2.0 许可。原始模型源码内嵌于 Notebook，许可证也随 Notebook 引导单元写入。
