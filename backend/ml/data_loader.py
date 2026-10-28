
from pathlib import Path

import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data" / "raw"


def load_equity_data():
    path = DATA_DIR / "equity_daily_returns_nifty50.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    required = {
        "date",
        "symbol",
        "adjusted_close",
        "daily_return_adj_close",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Equity CSV missing columns: {missing}")

    df = df.drop_duplicates(subset=["date", "symbol"])
    df = df.sort_values(["symbol", "date"])

    df["adjusted_close"] = pd.to_numeric(
        df["adjusted_close"], errors="coerce"
    )
    df["daily_return_adj_close"] = pd.to_numeric(
        df["daily_return_adj_close"], errors="coerce"
    )

    return df


def load_mutual_fund_data():
    path = DATA_DIR / "mutual_fund_nav_returns.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    required = {
        "date",
        "scheme_code",
        "scheme_name",
        "nav",
        "daily_nav_return",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mutual fund CSV missing columns: {missing}")

    df = df.drop_duplicates(subset=["date", "scheme_code"])
    df = df.sort_values(["scheme_code", "date"])

    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["daily_nav_return"] = pd.to_numeric(
        df["daily_nav_return"], errors="coerce"
    )

    return df


def load_gold_data():
    path = DATA_DIR / "gold_data_unvalidated.csv"
    df = pd.read_csv(path, parse_dates=["Date"])

    # This dataset is not yet verified for market, instrument,
    # and currency. Load it for inspection, but do not yet
    # combine it with Indian asset returns.
    df = df.sort_values("Date")

    return df


def load_gsec_yields():
    path = DATA_DIR / "india_10y_gsec_yield_changes.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    required = {"date", "yield_pct", "yield_change_bps"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"G-Sec CSV missing columns: {missing}")

    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date")

    # Yield changes are macroeconomic features, NOT bond returns.
    return df


def print_data_summary():
    equity = load_equity_data()
    funds = load_mutual_fund_data()
    gold = load_gold_data()
    yields = load_gsec_yields()

    print("\nEQUITY")
    print("Rows:", len(equity))
    print("Unique symbols:", equity["symbol"].nunique())
    print("Date range:", equity["date"].min(), "to", equity["date"].max())

    print("\nMUTUAL FUNDS")
    print("Rows:", len(funds))
    print("Unique schemes:", funds["scheme_code"].nunique())
    print("Date range:", funds["date"].min(), "to", funds["date"].max())

    print("\nGOLD")
    print("Rows:", len(gold))
    print("Date range:", gold["Date"].min(), "to", gold["Date"].max())
    print("WARNING: market, instrument and currency need verification.")

    print("\nGOVERNMENT YIELDS")
    print("Rows:", len(yields))
    print("Date range:", yields["date"].min(), "to", yields["date"].max())


if __name__ == "__main__":
    print_data_summary()
