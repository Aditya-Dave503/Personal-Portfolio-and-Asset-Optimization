"""Phase 2 feature engineering for the personal asset management project.

Creates leakage-aware historical features and forward-return targets from the
Phase 1 preliminary dataset. Gold is deliberately written to a separate,
non-training table until its instrument, source, market and currency are verified.
Government-security yield changes are explanatory features, never bond returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126, 252)  # trading-day horizons: ~1m, 1q, 6m, 1y
ROLLING_WINDOWS = (21, 63, 126, 252)

EQUITY_FILE = "equity_daily_returns_nifty50.csv"
MF_FILE = "mutual_fund_nav_returns.csv"
GOLD_FILE = "gold_data_unvalidated.csv"
YIELD_FILE = "india_10y_gsec_yield_changes.csv"

BASE_FEATURES = [
    "daily_return",
    "return_lag_1d",
    "return_lag_5d",
    "return_lag_21d",
    "trailing_return_5d",
    "trailing_return_21d",
    "trailing_return_63d",
    "trailing_return_126d",
    "trailing_return_252d",
    "volatility_21d",
    "volatility_63d",
    "volatility_126d",
    "volatility_252d",
    "drawdown_252d",
]
MACRO_FEATURES = ["gsec_yield_pct", "gsec_yield_change_bps", "gsec_yield_change_5d_bps", "gsec_yield_change_21d_bps"]
TARGET_COLUMNS = [f"future_return_{h}d" for h in HORIZONS]


def _read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df


def _rolling_return(returns: pd.Series, window: int) -> pd.Series:
    # Compound simple returns through rolling log sums; requires a complete window.
    safe = returns.where(returns > -1.0)
    return np.expm1(np.log1p(safe).rolling(window, min_periods=window).sum())


def _forward_return_targets(returns: pd.Series, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """For row t, compound returns from t+1 through t+h, never including t."""
    values = returns.astype(float).to_numpy()
    # Invalid or impossible simple returns should not silently become usable labels.
    valid = np.isfinite(values) & (values > -1.0)
    log_returns = np.full(values.shape, np.nan, dtype=float)
    log_returns[valid] = np.log1p(values[valid])
    reverse_log = pd.Series(log_returns[::-1])
    reverse_valid = pd.Series(valid[::-1].astype(float))
    out = {}
    for h in horizons:
        # At reversed position for t, shift(1) selects the h observations t+1..t+h.
        sums = reverse_log.rolling(h, min_periods=h).sum().shift(1).to_numpy()[::-1]
        counts = reverse_valid.rolling(h, min_periods=h).sum().shift(1).to_numpy()[::-1]
        target = np.expm1(sums)
        target[(counts != h) | ~np.isfinite(target)] = np.nan
        out[f"future_return_{h}d"] = target
    return pd.DataFrame(out, index=returns.index)


def _features_one_asset(group: pd.DataFrame, price_col: str, return_col: str) -> pd.DataFrame:
    g = group.sort_values("date").copy().reset_index(drop=True)
    price = pd.to_numeric(g[price_col], errors="coerce")
    # Prefer recomputing returns from the selected price series to ensure consistent units.
    # First return is naturally missing. A non-positive price invalidates its return.
    computed = price.pct_change(fill_method=None)
    ret = computed.where((price > 0) & (price.shift(1) > 0)).replace([np.inf, -np.inf], np.nan)
    g["daily_return"] = ret
    g["return_lag_1d"] = ret.shift(1)
    g["return_lag_5d"] = ret.shift(5)
    g["return_lag_21d"] = ret.shift(21)
    for w in (5, 21, 63, 126, 252):
        g[f"trailing_return_{w}d"] = _rolling_return(ret, w)
    for w in ROLLING_WINDOWS:
        g[f"volatility_{w}d"] = ret.rolling(w, min_periods=w).std(ddof=1) * np.sqrt(252)
    g["drawdown_252d"] = price / price.rolling(252, min_periods=252).max() - 1.0
    targets = _forward_return_targets(ret)
    return pd.concat([g, targets], axis=1)


def _load_equities(input_dir: Path) -> pd.DataFrame:
    df = _read_csv(input_dir / EQUITY_FILE, ["date", "symbol", "adjusted_close"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["adjusted_close"] = pd.to_numeric(df["adjusted_close"], errors="coerce")
    df = df.dropna(subset=["date", "symbol", "adjusted_close"])
    if df.duplicated(["date", "symbol"]).any():
        raise ValueError("Equity data contains duplicate date/symbol rows; resolve before feature generation.")
    pieces = []
    for symbol, group in df.groupby("symbol", sort=False):
        part = _features_one_asset(group, "adjusted_close", "daily_return_adj_close" if "daily_return_adj_close" in group else "daily_return")
        part["asset_id"] = str(symbol)
        part["asset_type"] = "equity"
        part["asset_name"] = part["company_name"] if "company_name" in part else str(symbol)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def _load_mutual_funds(input_dir: Path) -> pd.DataFrame:
    df = _read_csv(input_dir / MF_FILE, ["date", "category", "scheme_code", "scheme_name", "nav"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "scheme_code", "nav"])
    if df.duplicated(["date", "scheme_code"]).any():
        raise ValueError("Mutual-fund data contains duplicate date/scheme_code rows; resolve before feature generation.")
    pieces = []
    for code, group in df.groupby("scheme_code", sort=False):
        part = _features_one_asset(group, "nav", "daily_nav_return" if "daily_nav_return" in group else "daily_return")
        part["asset_id"] = str(code)
        part["asset_type"] = "mutual_fund"
        part["asset_name"] = part["scheme_name"] if "scheme_name" in part else str(code)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def _yield_features(input_dir: Path) -> pd.DataFrame:
    df = _read_csv(input_dir / YIELD_FILE, ["date", "yield_pct", "yield_change_bps"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("yield_pct", "yield_change_bps"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df["date"].duplicated().any():
        raise ValueError("Government-security yield data contains duplicate dates.")
    df["gsec_yield_pct"] = df["yield_pct"]
    df["gsec_yield_change_bps"] = df["yield_change_bps"]
    df["gsec_yield_change_5d_bps"] = df["yield_change_bps"].rolling(5, min_periods=5).sum()
    df["gsec_yield_change_21d_bps"] = df["yield_change_bps"].rolling(21, min_periods=21).sum()
    return df[["date"] + MACRO_FEATURES].sort_values("date")


def _load_gold_isolated(input_dir: Path) -> pd.DataFrame:
    df = _read_csv(input_dir / GOLD_FILE, ["Date"])
    date_col = "Date"
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        raise ValueError("Gold file needs 'Adj Close' or 'Close'.")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["date", price_col]).sort_values("date")
    if df["date"].duplicated().any():
        raise ValueError("Gold data contains duplicate dates.")
    # Recompute return as a fraction. Do not trust the supplied Daily_Return column:
    # the audit found it is expressed inconsistently with recomputed returns.
    df["price_for_features"] = df[price_col]
    part = _features_one_asset(df, "price_for_features", "recomputed_return")
    part["asset_id"] = "GOLD_UNVALIDATED"
    part["asset_type"] = "gold_unvalidated"
    part["asset_name"] = "Unvalidated gold series"
    part["eligible_for_training"] = False
    return part


def _attach_macro(asset_df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    left = asset_df.sort_values("date").copy()
    # Only use the latest yield observation on or before each asset date.
    merged = pd.merge_asof(left, macro.sort_values("date"), on="date", direction="backward")
    return merged


def build_features(input_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading equity data...", flush=True)
    equities = _load_equities(input_dir)
    print("Loading mutual-fund data...", flush=True)
    funds = _load_mutual_funds(input_dir)
    investable = pd.concat([equities, funds], ignore_index=True, sort=False)
    investable["eligible_for_training"] = True
    print("Building yield features...", flush=True)
    macro = _yield_features(input_dir)
    investable = _attach_macro(investable, macro)
    # Stable schema; retain metadata and feature/target columns only, not raw OHLCV fields.
    keep = ["date", "asset_id", "asset_type", "asset_name", "category", "daily_return"] + BASE_FEATURES[1:] + MACRO_FEATURES + TARGET_COLUMNS + ["eligible_for_training"]
    for col in keep:
        if col not in investable.columns:
            investable[col] = np.nan
    investable = investable[keep].sort_values(["asset_type", "asset_id", "date"]).reset_index(drop=True)
    print("Writing investable features...", flush=True)
    investable_path = output_dir / "asset_features.csv.gz"
    investable.to_csv(investable_path, index=False, compression={"method": "gzip", "compresslevel": 9}, date_format="%Y-%m-%d")

    print("Building isolated gold features...", flush=True)
    gold = _load_gold_isolated(input_dir)
    gold_keep = ["date", "asset_id", "asset_type", "asset_name", "daily_return"] + BASE_FEATURES[1:] + TARGET_COLUMNS + ["eligible_for_training"]
    for col in gold_keep:
        if col not in gold.columns:
            gold[col] = np.nan
    gold = gold[gold_keep].sort_values("date").reset_index(drop=True)
    gold_path = output_dir / "gold_features_unvalidated.csv.gz"
    gold.to_csv(gold_path, index=False, compression={"method": "gzip", "compresslevel": 9}, date_format="%Y-%m-%d")

    macro_path = output_dir / "gsec_yield_features.csv"
    macro.to_csv(macro_path, index=False, date_format="%Y-%m-%d")

    # Per-horizon training sets: remove rows lacking the horizon target or complete features.
    feature_cols = BASE_FEATURES + MACRO_FEATURES
    training_counts = {}
    for h in HORIZONS:
        target = f"future_return_{h}d"
        needed = [c for c in feature_cols if c in investable.columns]
        ready = investable.dropna(subset=needed + [target]).copy()
        out_path = output_dir / f"training_dataset_{h}d.csv.gz"
        ready.to_csv(out_path, index=False, compression={"method": "gzip", "compresslevel": 9}, date_format="%Y-%m-%d")
        training_counts[str(h)] = {"rows": int(len(ready)), "assets": int(ready["asset_id"].nunique()), "path": out_path.name}

    # Feature dictionary and machine-readable run manifest.
    descriptions = {
        "daily_return": "Return from the previous available observation to this date; recomputed from adjusted close/NAV.",
        "return_lag_1d": "Daily return shifted by one observation; only prior observations.",
        "return_lag_5d": "Daily return shifted by five observations.",
        "return_lag_21d": "Daily return shifted by 21 observations.",
        "trailing_return_*": "Compounded return over the trailing window ending on feature date.",
        "volatility_*": "Rolling sample standard deviation of daily returns, annualized by sqrt(252).",
        "drawdown_252d": "Current price/NAV relative to the trailing 252-observation high, minus one.",
        "gsec_yield_pct": "10-year government-security yield level in percent; not a bond return.",
        "gsec_yield_change_*": "Government-security yield changes in basis points, not investment returns.",
        "future_return_*": "Compounded simple return from the next observation through the named horizon; supervised-learning target only, never a feature.",
    }
    manifest = {
        "phase": 2,
        "feature_date_rule": "Features at date t use price/return data through date t; forward targets begin at the next available observation t+1.",
        "horizons_trading_days": list(HORIZONS),
        "feature_columns": BASE_FEATURES + MACRO_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "training_counts_by_horizon": training_counts,
        "asset_counts": {
            "equities": int(equities["asset_id"].nunique()),
            "mutual_funds": int(funds["asset_id"].nunique()),
            "gold_rows_isolated": int(len(gold)),
        },
        "limitations": [
            "Gold source/instrument/market/currency are unverified; isolated and marked ineligible for training.",
            "Government-security yield changes are not bond investment returns.",
            "Current Nifty 50 constituents create survivorship bias.",
            "Historical features and targets are not a guarantee of future performance.",
            "Use chronological walk-forward validation; do not randomly split rows across time.",
            "Rows across assets on the same date are not independent samples.",
            "Mutual funds can overlap with Nifty 50 holdings; account for look-through exposure in portfolio optimization.",
        ],
        "feature_descriptions": descriptions,
        "outputs": [investable_path.name, gold_path.name, macro_path.name] + [v["path"] for v in training_counts.values()],
    }
    (output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(output_dir, manifest, investable, gold, macro)
    return manifest


def _write_report(output_dir: Path, manifest: dict, investable: pd.DataFrame, gold: pd.DataFrame, macro: pd.DataFrame) -> None:
    lines = [
        "# Phase 2 — Feature Engineering Report", "",
        "## What was generated", "",
        f"- Equity and mutual-fund feature rows: **{len(investable):,}**",
        f"- Equity symbols: **{manifest['asset_counts']['equities']}**",
        f"- Mutual-fund schemes: **{manifest['asset_counts']['mutual_funds']}**",
        f"- Isolated gold rows (not eligible for training): **{len(gold):,}**",
        f"- Government-security yield feature rows: **{len(macro):,}**", "",
        "## Features", "",
        "- Lagged daily returns: 1, 5 and 21 observations.",
        "- Compounded trailing returns: 5, 21, 63, 126 and 252 observations.",
        "- Annualized rolling volatility: 21, 63, 126 and 252 observations.",
        "- Drawdown from the trailing 252-observation high.",
        "- 10-year government-security yield level and 5/21-observation yield-change aggregates.", "",
        "## Targets", "",
        "Forward compounded returns for 21, 63, 126 and 252 trading-day horizons. Targets start at the next observation after the feature date; target columns must never be used as model inputs.", "",
        "## Training-ready row counts", "",
        "| Horizon | Rows | Assets | Output |", "|---:|---:|---:|---|",
    ]
    for h, info in manifest["training_counts_by_horizon"].items():
        lines.append(f"| {h} trading days | {info['rows']:,} | {info['assets']} | `{info['path']}` |")
    lines += ["", "## Data and modelling cautions", ""]
    lines += [f"- {item}" for item in manifest["limitations"]]
    lines += ["", "## Next phase", "", "Phase 3 should establish a baseline and backtesting protocol before selecting a predictive model. Use chronological walk-forward splits, fit preprocessing only on each training window, and compare any ML model against a historical-return baseline.", ""]
    (output_dir / "phase2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 2 investment feature tables.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing Phase 1 CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    args = parser.parse_args()
    manifest = build_features(args.input_dir, args.output_dir)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "training_counts_by_horizon": manifest["training_counts_by_horizon"]}, indent=2))


if __name__ == "__main__":
    main()
