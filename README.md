# Pairs Trading Strategy

This project implements a cointegration-based pairs trading strategy across S&P 500 equities. It discovers cointegrated pairs, generates mean-reversion signals from spread z-scores, and backtests the strategy on train, validation, and test periods.

## Project Overview

The strategy follows these steps:

1. Build a universe from S&P 500 constituents.
2. Download historical price data using `yfinance`.
3. Clean the data and split it into train, validation, and test periods.
4. Discover cointegrated pairs within the same industry.
5. Generate z-score based trading signals.
6. Backtest the strategy with transaction costs.
7. Tune parameters using a validation set.
8. Evaluate final performance on the test set.

## Pipeline

```bash
pip install -r requirements.txt

python src/data.py
python src/clean.py
python src/discovery.py
python src/signals.py
python src/backtest.py
python src/validation.py
```

## Current Result

| Period | Sharpe | Total Return |
| --- | ---: | ---: |
| Train | 2.07 | 18.09% |
| Validation | 2.40 | 3.79% |
| Test | 0.78 | 7.65% |

The strategy works well in sample but decays out of sample. The test result remains positive, but the return is not high enough to justify real-world use without further improvements.

## Report

See `report/pairs_trading_report.ipynb` for the full analysis.
