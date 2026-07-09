# Linear Regression vs MACD in VN-Index Context

This repository compares linear-regression forecasting models against the MACD 12/26/9 trading rule on VN-Index data.

## Contents

- `linear_regression_backtest.py`: builds features, trains linear models, runs the Ridge strategy backtest, and exports visual reports.
- `compare_ridge_vs_macd12269.py`: compares the best Ridge model against MACD 12/26/9 on the same test period.
- `stress_test_volatile_years.py`: retrains Ridge year by year and stress tests the most volatile VN-Index years against MACD.
- `READ.md`: detailed Vietnamese stress-test report with tables, charts, and interpretation.
- `data.csv`: VN-Index OHLCV input data.

## Main Reports

- Linear-regression report: `outputs_linear_regression_backtest/report.html`
- Ridge vs MACD report: `outputs_model_vs_macd12269/ridge_vs_macd12269_report.html`
- Stress-test report: `READ.md`

## Reproduce

Use the `eda` conda environment:

```bash
/home/namngyh/miniconda3/envs/eda/bin/python linear_regression_backtest.py
/home/namngyh/miniconda3/envs/eda/bin/python compare_ridge_vs_macd12269.py
/home/namngyh/miniconda3/envs/eda/bin/python stress_test_volatile_years.py
```

The CSV parser reconstructs OHLCV values that are split by thousands separators in the source file.
