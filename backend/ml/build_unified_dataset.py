
from pathlib import Path
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"


def monthly_returns_from_prices(df, date_col, asset_col, price_col, prefix):
    """Calculate monthly returns from each asset's price or NAV."""
    df = df[[date_col, asset_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    df = df.dropna(subset=[date_col, asset_col, price_col])
    df = df[df[price_col] > 0]
    df = df.drop_duplicates([date_col, asset_col], keep="last")
    df = df.sort_values([asset_col, date_col])

    # Use the last available price/NAV for each asset in each month.
    df["month"] = df[date_col].dt.to_period("M")
    month_end_prices = (
        df.groupby(["month", asset_col], observed=True)
        .tail(1)
        .pivot(index="month", columns=asset_col, values=price_col)
        .sort_index()
    )

    # Calculate returns separately for each asset.
    returns = month_end_prices.pct_change(fill_method=None)
    returns.columns = [
        f"{prefix}_{str(asset)}" for asset in returns.columns
    ]

    return returns


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Nifty 50 stocks
    equity_path = RAW_DIR / "equity_daily_returns_nifty50.csv"
    equity = pd.read_csv(equity_path, parse_dates=["date"])

    stock_returns = monthly_returns_from_prices(
        equity,
        date_col="date",
        asset_col="symbol",
        price_col="adjusted_close",
        prefix="STOCK"
    )

    # 2. Mutual funds
    funds_path = RAW_DIR / "mutual_fund_nav_returns.csv"
    funds = pd.read_csv(funds_path, parse_dates=["date"])

    # Use scheme_code as a stable asset identifier.
    funds["scheme_code"] = funds["scheme_code"].astype(str)

    fund_returns = monthly_returns_from_prices(
        funds,
        date_col="date",
        asset_col="scheme_code",
        price_col="nav",
        prefix="FUND"
    )

    # Preserve each asset's own history; do not drop rows
    # just because another asset has a missing value.
    asset_returns = pd.concat(
        [stock_returns, fund_returns],
        axis=1
    ).sort_index()

    asset_returns.index = asset_returns.index.astype(str)
    asset_returns.index.name = "month"
    asset_returns.to_csv(
        PROCESSED_DIR / "monthly_asset_returns.csv"
    )

    # 3. India 10-year G-Sec market-risk features
    gsec_path = RAW_DIR / "india_10y_gsec_yield_changes.csv"
    gsec = pd.read_csv(gsec_path, parse_dates=["date"])

    gsec["yield_pct"] = pd.to_numeric(
        gsec["yield_pct"], errors="coerce"
    )
    gsec = gsec.dropna(subset=["date", "yield_pct"])
    gsec = gsec.sort_values("date")
    gsec = gsec.drop_duplicates("date", keep="last")
    gsec["month"] = gsec["date"].dt.to_period("M")

    # Last observed yield in each month.
    monthly_yield = (
        gsec.groupby("month")["yield_pct"]
        .last()
        .sort_index()
        .to_frame()
    )

    # Change in yield in percentage points and basis points.
    monthly_yield["yield_change_pct_points"] = (
        monthly_yield["yield_pct"].diff()
    )
    monthly_yield["yield_change_bps"] = (
        monthly_yield["yield_change_pct_points"] * 100
    )

    monthly_yield.index = monthly_yield.index.astype(str)
    monthly_yield.index.name = "month"
    monthly_yield.to_csv(
        PROCESSED_DIR / "monthly_gsec_features.csv"
    )

    # 4. Gold: recalculate returns for inspection only.
    # Do NOT mark gold eligible until instrument/currency/source
    # are verified.
    gold_path = RAW_DIR / "gold_data_unvalidated.csv"
    gold = pd.read_csv(gold_path, parse_dates=["Date"])

    gold["Adj Close"] = pd.to_numeric(
        gold["Adj Close"], errors="coerce"
    )
    gold = gold.dropna(subset=["Date", "Adj Close"])
    gold = gold[gold["Adj Close"] > 0]
    gold = gold.drop_duplicates("Date", keep="last")
    gold = gold.sort_values("Date")
    gold["month"] = gold["Date"].dt.to_period("M")

    gold_monthly = (
        gold.groupby("month")["Adj Close"]
        .last()
        .sort_index()
        .pct_change(fill_method=None)
        .to_frame("GOLD_RECOMPUTED_RETURN")
    )

    gold_monthly.index = gold_monthly.index.astype(str)
    gold_monthly.index.name = "month"
    gold_monthly.to_csv(
        PROCESSED_DIR / "gold_monthly_returns_pending_verification.csv"
    )

    print("Unified dataset build completed.")
    print(
        "Monthly stock + fund returns:",
        asset_returns.shape[0], "months,",
        asset_returns.shape[1], "assets"
    )
    print(
        "Monthly G-Sec features:",
        monthly_yield.shape[0], "months"
    )
    print(
        "Gold returns recalculated and saved for inspection only."
    )
    print(
        "Gold remains excluded from eligible assets until "
        "its instrument, currency, and source are verified."
    )
    print("Saved files:")
    print(PROCESSED_DIR / "monthly_asset_returns.csv")
    print(PROCESSED_DIR / "monthly_gsec_features.csv")
    print(
        PROCESSED_DIR /
        "gold_monthly_returns_pending_verification.csv"
    )


if __name__ == "__main__":
    main()
