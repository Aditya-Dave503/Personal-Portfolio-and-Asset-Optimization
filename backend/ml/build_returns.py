
from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def build_returns():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. NIFTY 50 CONSTITUENT RETURNS
    # --------------------------------------------------
    equity_path = RAW_DIR / "equity_daily_returns_nifty50.csv"
    equity = pd.read_csv(equity_path, parse_dates=["date"])

    equity["adjusted_close"] = pd.to_numeric(
        equity["adjusted_close"], errors="coerce"
    )

    equity = equity.dropna(
        subset=["date", "symbol", "adjusted_close"]
    )

    equity = equity.drop_duplicates(
        subset=["date", "symbol"],
        keep="last"
    )

    equity = equity.sort_values(["symbol", "date"])

    # Recompute returns from adjusted prices.
    equity["return"] = (
        equity.groupby("symbol")["adjusted_close"]
        .pct_change(fill_method=None)
    )

    equity_returns = equity.pivot(
        index="date",
        columns="symbol",
        values="return"
    )

    equity_returns.columns = [
        f"STOCK_{symbol}" for symbol in equity_returns.columns
    ]

    # --------------------------------------------------
    # 2. MUTUAL FUND RETURNS
    # --------------------------------------------------
    fund_path = RAW_DIR / "mutual_fund_nav_returns.csv"
    funds = pd.read_csv(fund_path, parse_dates=["date"])

    funds["nav"] = pd.to_numeric(
        funds["nav"], errors="coerce"
    )

    funds = funds.dropna(
        subset=["date", "scheme_code", "nav"]
    )

    funds = funds[funds["nav"] > 0]

    funds = funds.drop_duplicates(
        subset=["date", "scheme_code"],
        keep="last"
    )

    funds = funds.sort_values(["scheme_code", "date"])

    # Recompute returns from NAV values.
    funds["return"] = (
        funds.groupby("scheme_code")["nav"]
        .pct_change(fill_method=None)
    )

    fund_returns = funds.pivot(
        index="date",
        columns="scheme_code",
        values="return"
    )

    fund_returns.columns = [
        f"FUND_{scheme_code}"
        for scheme_code in fund_returns.columns
    ]

    # --------------------------------------------------
    # 3. ALIGN RETURNS BY DATE
    # --------------------------------------------------
    # Inner join uses only dates shared by all selected assets.
    # Do not forward-fill returns on missing trading dates.
    returns = equity_returns.join(
        fund_returns,
        how="inner"
    )

    returns = returns.sort_index()
    returns.index.name = "date"

    # Remove the first row if returns are unavailable.
    returns = returns.dropna(how="any")

    output_path = PROCESSED_DIR / "asset_daily_returns.csv"
    returns.to_csv(output_path)

    print("Return dataset created successfully.")
    print("Output:", output_path)
    print("Rows:", len(returns))
    print("Assets:", len(returns.columns))
    print("Date range:", returns.index.min(), "to", returns.index.max())
    print("\nFirst five rows:")
    print(returns.head().to_string())


if __name__ == "__main__":
    build_returns()
