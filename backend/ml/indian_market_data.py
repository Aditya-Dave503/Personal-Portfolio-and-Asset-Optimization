"""
Indian Multi-Asset Market Data Ingestion & Preprocessing Pipeline.

Loads, cleans, and synchronizes 10-year historical daily time series from:
1. Nifty 50 Equities (equity_daily_returns_nifty50.csv)
2. Indian Mutual Funds (mutual_fund_nav_returns.csv)
3. MCX Spot Gold (gold_data_unvalidated.csv)
4. Indian 10-Year Government Securities (india_10y_gsec_yield_changes.csv)

Converts 10Y G-Sec yield changes into real bond price returns using
Modified Duration (D_mod = 6.8y) and Convexity (C = 55.0).
Deliberately excludes Silver to align with project specifications.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import os
import numpy as np
import pandas as pd


# Default search locations for data directory to prevent FileNotFoundError
POSSIBLE_DATA_DIRS = [
    Path(__file__).resolve().parent.parent / "data" / "raw",
    Path(__file__).resolve().parent / "data" / "raw",
    Path.cwd() / "backend" / "data" / "raw",
    Path.cwd() / "data" / "raw",
    Path(__file__).resolve().parent.parent / "Personal-Portfolio-and-Asset-Optimization" / "backend" / "data" / "raw",
    Path("C:/Users/adity/.gemini/antigravity/scratch/Personal-Portfolio-and-Asset-Optimization/backend/data/raw"),
]


def resolve_data_directory(custom_path: Optional[str] = None) -> Path:
    """Find the valid raw data directory from candidate paths."""
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Custom data directory does not exist: {custom_path}")

    for candidate in POSSIBLE_DATA_DIRS:
        if candidate.exists() and (candidate / "equity_daily_returns_nifty50.csv").exists():
            return candidate

    raise FileNotFoundError(
        "Could not automatically locate the raw data folder containing equity_daily_returns_nifty50.csv. "
        "Please provide --data-dir explicitly."
    )


# Asset universe metadata (Silver is excluded)
ASSET_METADATA = {
    "NIFTY_EQUITIES": {
        "name": "Nifty 50 Large-Cap Equities",
        "category": "Equities",
        "risk_level": "High Risk / High Growth",
        "benchmark_weight": 0.40,
    },
    "MUTUAL_FUNDS": {
        "name": "Diversified Indian Mutual Funds",
        "category": "Active Equities / Hybrid",
        "risk_level": "Moderate-High Risk",
        "benchmark_weight": 0.30,
    },
    "MCX_GOLD": {
        "name": "Sovereign / MCX Gold",
        "category": "Precious Metals",
        "risk_level": "Moderate Risk / Inflation Hedge",
        "benchmark_weight": 0.15,
    },
    "GOVT_10Y_GSEC": {
        "name": "10-Year Indian Government Securities (G-Sec)",
        "category": "Sovereign Debt",
        "risk_level": "Low Risk / Capital Preservation",
        "benchmark_weight": 0.15,
    },
}


def load_nifty_equity_returns(data_dir: Path) -> pd.Series:
    """
    Load Nifty 50 stock data and compute equal-weighted index daily returns.
    Equal-weighting avoids mega-cap concentration bias (Reliance/HDFC dominating 25%+).
    """
    path = data_dir / "equity_daily_returns_nifty50.csv"
    if not path.exists():
        raise FileNotFoundError(f"Equities CSV not found at: {path}")

    df = pd.read_csv(path, usecols=["date", "adjusted_close", "symbol"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "adjusted_close", "symbol"])
    df = df[df["adjusted_close"] > 0]
    df = df.drop_duplicates(subset=["date", "symbol"], keep="last")

    # Compute equal-weighted average price across constituents, then daily return
    market_return = df.groupby("date")["adjusted_close"].mean().pct_change()
    market_return.name = "NIFTY_EQUITIES"
    return market_return.dropna()


def load_mutual_fund_returns(data_dir: Path) -> pd.Series:
    """Load diversified mutual fund schemes and compute aggregate daily return."""
    path = data_dir / "mutual_fund_nav_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Mutual Funds CSV not found at: {path}")

    df = pd.read_csv(path, usecols=["date", "nav", "scheme_code"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "nav", "scheme_code"])
    df = df[df["nav"] > 0]
    df = df.drop_duplicates(subset=["date", "scheme_code"], keep="last")

    mf_return = df.groupby("date")["nav"].mean().pct_change()
    mf_return.name = "MUTUAL_FUNDS"
    return mf_return.dropna()


def load_mcx_gold_returns(data_dir: Path) -> pd.Series:
    """Load MCX / spot gold daily prices and compute clean returns."""
    path = data_dir / "gold_data_unvalidated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Gold CSV not found at: {path}")

    df = pd.read_csv(path)
    date_col = "Date" if "Date" in df.columns else "date"
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    gold_series = df.set_index(date_col)[price_col].astype(float)
    gold_return = gold_series.pct_change()
    gold_return.name = "MCX_GOLD"
    return gold_return.dropna()


def load_gsec_bond_returns(
    data_dir: Path,
    modified_duration: float = 6.8,
    convexity: float = 55.0
) -> pd.Series:
    """
    Convert 10-Year Indian Government Security yield changes into daily bond price returns
    using professional 2nd-order Taylor approximation (Duration + Convexity):
    
    R_{bond, t} = (Yield_{t-1} / 252) - D_mod * Delta(Yield_t) + 0.5 * Convexity * (Delta(Yield_t))^2
    """
    path = data_dir / "india_10y_gsec_yield_changes.csv"
    if not path.exists():
        raise FileNotFoundError(f"G-Sec CSV not found at: {path}")

    df = pd.read_csv(path, usecols=["date", "yield_pct"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "yield_pct"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").set_index("date")

    y = df["yield_pct"].astype(float) / 100.0  # Decimal yield
    delta_y = y.diff()

    carry_return = y.shift(1) / 252.0
    duration_effect = -modified_duration * delta_y
    convexity_effect = 0.5 * convexity * (delta_y ** 2)

    bond_return = carry_return + duration_effect + convexity_effect
    bond_return.name = "GOVT_10Y_GSEC"
    return bond_return.dropna()


def load_macro_yield_features(data_dir: Path) -> pd.DataFrame:
    """Load Indian 10Y sovereign yield levels and yield changes as macro features."""
    path = data_dir / "india_10y_gsec_yield_changes.csv"
    df = pd.read_csv(path, usecols=["date", "yield_pct", "yield_change_bps"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    df = df.set_index("date")
    return df


def get_synchronized_market_data(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Align all 4 core Indian asset classes on common trading dates.
    Using inner-join prevents artificial zero-correlation distortion (Epps effect).
    """
    if data_dir is None:
        data_dir = resolve_data_directory()

    nifty = load_nifty_equity_returns(data_dir)
    mf = load_mutual_fund_returns(data_dir)
    gold = load_mcx_gold_returns(data_dir)
    gsec = load_gsec_bond_returns(data_dir)

    synchronized = pd.concat([nifty, mf, gold, gsec], axis=1, join="inner").dropna()
    return synchronized.sort_index()


if __name__ == "__main__":
    d = resolve_data_directory()
    df = get_synchronized_market_data(d)
    print("=" * 65)
    print("      INDIAN MULTI-ASSET MARKET DATA SYNCHRONIZATION")
    print("=" * 65)
    print(f"Data Directory:            {d}")
    print(f"Synchronized Observations: {len(df):,} trading days")
    print(f"Date Coverage:             {df.index.min().date()} to {df.index.max().date()}")
    print("\nAsset Summary Statistics (Annualized):")
    summary = pd.DataFrame({
        "Mean Return (%)": (df.mean() * 252 * 100).round(2),
        "Volatility (%)": (df.std() * np.sqrt(252) * 100).round(2),
        "Sharpe (Rf=6.5%)": ((df.mean() * 252 - 0.065) / (df.std() * np.sqrt(252))).round(2),
    })
    print(summary)
    print("=" * 65)
