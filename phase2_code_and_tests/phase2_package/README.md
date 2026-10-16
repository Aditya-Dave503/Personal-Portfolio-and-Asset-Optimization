# Phase 2: Feature Engineering

This package builds leakage-aware feature tables from the Phase 1 preliminary dataset.

## Inputs
Place these Phase 1 CSV files in one directory:

- `equity_daily_returns_nifty50.csv`
- `mutual_fund_nav_returns.csv`
- `gold_data_unvalidated.csv`
- `india_10y_gsec_yield_changes.csv`

## Install
Use the project's virtual environment and install the data-processing dependencies:

```powershell
python -m pip install pandas numpy pytest
```

## Run
From the folder containing `features/` and `tests/`:

```powershell
python -m features.feature_engineering --input-dir "PATH_TO_PHASE1_CSV_FOLDER" --output-dir "outputs/phase2"
```

For the uploaded ZIP, extract the four CSVs into a folder first, then pass that folder as `--input-dir`.

## Outputs

- `asset_features.csv.gz`: equity and mutual-fund features, targets, and lagged G-Sec yield features.
- `training_dataset_21d.csv.gz`, `training_dataset_63d.csv.gz`, `training_dataset_126d.csv.gz`, `training_dataset_252d.csv.gz`: rows with complete features and the corresponding forward-return target.
- `gold_features_unvalidated.csv.gz`: separate gold feature table. It is explicitly not eligible for model training until the source, instrument and units are verified.
- `gsec_yield_features.csv`: yield-level/change features only; not bond returns.
- `feature_manifest.json`: feature/target definitions, row counts and limitations.
- `phase2_report.md`: human-readable report.

## Feature timing and leakage controls
Features at date `t` use price/NAV/market data available through `t`. Forward targets compound returns from the next available observation through the specified horizon. The forward-return target columns must never be included as input features. Use chronological walk-forward validation, not a random row split.

Daily observation horizons are approximate: 21 (~1 month), 63 (~quarter), 126 (~half-year), and 252 (~year) trading observations. The data is not a guarantee of future performance. Current Nifty 50 membership creates survivorship bias, and funds may overlap with stocks.
