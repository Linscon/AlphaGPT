# BTC version validation

Validated locally on CPU, 2026-10-10. Colab GPU execution has not been verified.

- 11 unit tests passed (python -m unittest discover -s colab -p test_btc_colab.py).
- Gradio 5.50 launched locally: HTTP /config 200, 42 components.
- Real Gate cached BTC_USDT 1h input: 4080 candles, 510 funding observations; UTC 2026-04-23 through 2026-10-10 exclusive.
- 30 steps x 3 seeds x 32 batch: 2583 unique formulas; selected validation winner returned -11.0253% on untouched test, maximum drawdown 11.2899%. This small search did not deliver a profitable strategy.
- A fresh 3-step, single-seed pipeline smoke run produced formula, detailed CSVs, input snapshots, complete reproduction manifest, nine figures and interactive HTML. Minimum entries was reduced for this environment check; its long-only output does not establish trading advantage.
- --render-only regenerates saved charts without search or reselection; rendering versions and source hashes are recorded separately from original training provenance.
- Nine PNGs at 2400 x 1200 (300 DPI), nine editable SVGs; raw/process/result each have three charts. Raster grayscale QA previews use a separate subdirectory.
- Both strict figure audits passed; independent source/output review and visual inspection found no trading-logic or chart blockers.

Colab GPU, network access to Gate and Gradio public sharing still require verification in the user's Colab environment. Search outcomes depend on the actual configuration and data.
