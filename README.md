# AlphaGPT

## BTC 1h 多空因子挖掘（Colab / WebUI）

仓库新增 Gate BTC_USDT 永续合约研究入口，支持自定义训练步数、批大小、种子、成本与数据日期，输出最佳公式、9 张 PNG/SVG 图和交互式报告。

```bash
pip install -r requirements-btc.txt
python mine_btc.py --steps 200 --batch-size 128 --seeds 42,43,44
# 参数面板（Colab 中添加 --share）
python webui.py
```

本地还需安装 PyTorch；Colab 使用自带 GPU 版 PyTorch。详见 [BTC 使用说明](BTC_README.md)。
Colab 可打开 [Fork 克隆 Notebook](colab/AlphaGPT_BTC_Fork.ipynb)。
采用按时间划分的训练/验证/测试与真实历史资金费率；“最佳公式”指本次搜索的验证集最佳候选，不承诺盈利。


![](showcase.png)

*在 AlphaGPT 开源后，社区曾使用开源版本管理了 1.66 亿人民币规模资金。*

[时隔数月后，项目作者再次受到了人身安全威胁。](https://github.com/imbue-bit/fuck_all_tian)

也许金融真的是一个意外死亡率很高的行业🤔

---

如果您觉得 AlphaGPT 很有用，可以通过以下渠道捐助作者：

![](helpme.jpg)

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=imbue-bit/AlphaGPT&type=date&legend=top-left)](https://www.star-history.com/#imbue-bit/AlphaGPT&type=date&legend=top-left)
