
from pathlib import Path
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def build_returns():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Stock returns: retain each stock's available history.
    equity = pd.read_csv(
        RAW_DIR / "equity_daily_returns_nifty50.csv",
        parse_dates=["date"]
    )

    equity["adjusted_close"] = pd.to_numeric(
        equity["adjusted_close"], errors="coerce"
    )

    equity = equity.dropna(
        subset=["date", "symbol", "adjusted_close"]
    )
    equity = equity.drop_duplicates(
        ["date", "symbol"], keep="last"
    )
    equity = equity.sort_values(["symbol", "date"])

    equity["return"] = (
        equity.groupby("symbol")["adjusted_close"]
        .pct_change(fill_method=None)
    )

    stock_returns = equity.pivot(
        index="date", columns="symbol", values="return"
    )
    stock_returns.columns = [
        f"STOCK_{symbol}" for symbol in stock_returns.columns
    ]

    stock_path = PROCESSED_DIR / "stock_daily_returns.csv"
    stock_returns.to_csv(stock_path)

    # 2. Mutual fund returns: retain each scheme's history.
    funds = pd.read_csv(
        RAW_DIR / "mutual_fund_nav_returns.csv",
        parse_dates=["date"]
    )

    funds["nav"] = pd.to_numeric(funds["nav"], errors="coerce")
    funds = funds.dropna(subset=["date", "scheme_code", "nav"])
    funds = funds[funds["nav"] > 0]
    funds = funds.drop_duplicates(
        ["date", "scheme_code"], keep="last"
    )
    funds = funds.sort_values(["scheme_code", "date"])

    funds["return"] = (
        funds.groupby("scheme_code")["nav"]
        .pct_change(fill_method=None)
    )

    fund_returns = funds.pivot(
        index="date", columns="scheme_code", values="return"
    )
    fund_returns.columns = [
        f"FUND_{scheme_code}"
        for scheme_code in fund_returns.columns
    ]

    fund_path = PROCESSED_DIR / "fund_daily_returns.csv"
    fund_returns.to_csv(fund_path)

    # 3. Government yields: retain as a separate indicator.
    yields = pd.read_csv(
        RAW_DIR / "india_10y_gsec_yield_changes.csv",
        parse_dates=["date"]
    )
    yields = yields.drop_duplicates(["date"], keep="last")
    yields = yields.sort_values("date")

    yield_path = PROCESSED_DIR / "gsec_yields.csv"
    yields.to_csv(yield_path, index=False)

    print("Data processing completed.")
    print(
        f"Stock returns: {stock_returns.shape[0]} dates, "
        f"{stock_returns.shape[1]} stocks"
    )
    print(
        f"Mutual fund returns: {fund_returns.shape[0]} dates, "
        f"{fund_returns.shape[1]} schemes"
    )
    print("Saved:", stock_path)
    print("Saved:", fund_path)
    print("Saved:", yield_path)


if __name__ == "__main__":
    build_returns()
