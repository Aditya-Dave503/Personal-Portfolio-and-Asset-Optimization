# Phase 2 — Feature Engineering Report

## What was generated

- Equity and mutual-fund feature rows: **139,086**
- Equity symbols: **50**
- Mutual-fund schemes: **8**
- Isolated gold rows (not eligible for training): **2,511**
- Government-security yield feature rows: **2,448**

## Features

- Lagged daily returns: 1, 5 and 21 observations.
- Compounded trailing returns: 5, 21, 63, 126 and 252 observations.
- Annualized rolling volatility: 21, 63, 126 and 252 observations.
- Drawdown from the trailing 252-observation high.
- 10-year government-security yield level and 5/21-observation yield-change aggregates.

## Targets

Forward compounded returns for 21, 63, 126 and 252 trading-day horizons. Targets start at the next observation after the feature date; target columns must never be used as model inputs.

## Training-ready row counts

| Horizon | Rows | Assets | Output |
|---:|---:|---:|---|
| 21 trading days | 123,252 | 58 | `training_dataset_21d.csv.gz` |
| 63 trading days | 120,816 | 58 | `training_dataset_63d.csv.gz` |
| 126 trading days | 117,162 | 58 | `training_dataset_126d.csv.gz` |
| 252 trading days | 109,854 | 58 | `training_dataset_252d.csv.gz` |

## Data and modelling cautions

- Gold source/instrument/market/currency are unverified; isolated and marked ineligible for training.
- Government-security yield changes are not bond investment returns.
- Current Nifty 50 constituents create survivorship bias.
- Historical features and targets are not a guarantee of future performance.
- Use chronological walk-forward validation; do not randomly split rows across time.
- Rows across assets on the same date are not independent samples.
- Mutual funds can overlap with Nifty 50 holdings; account for look-through exposure in portfolio optimization.

## Next phase

Phase 3 should establish a baseline and backtesting protocol before selecting a predictive model. Use chronological walk-forward splits, fit preprocessing only on each training window, and compare any ML model against a historical-return baseline.
