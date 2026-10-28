
from pathlib import Path
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"

INPUT_FILE = RAW_DIR / "equity_daily_returns_nifty50.csv"


def main():
    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    df["adjusted_close"] = pd.to_numeric(
        df["adjusted_close"], errors="coerce"
    )

    df = df.dropna(
        subset=["date", "symbol", "adjusted_close"]
    )
    df = df[df["adjusted_close"] > 0]
    df = df.drop_duplicates(
        ["date", "symbol"], keep="last"
    )
    df = df.sort_values(["symbol", "date"])

    # Select each stock's last adjusted close in each month.
    df["month"] = df["date"].dt.to_period("M")

    month_end = (
        df.groupby(["symbol", "month"], observed=True)
        .tail(1)
        [["symbol", "month", "date", "adjusted_close"]]
        .sort_values(["symbol", "month"])
        .copy()
    )

    month_end["previous_price"] = (
        month_end.groupby("symbol")["adjusted_close"].shift(1)
    )
    month_end["previous_month"] = (
        month_end.groupby("symbol")["month"].shift(1)
    )
    month_end["monthly_return"] = (
        month_end["adjusted_close"]
        / month_end["previous_price"] - 1
    )

    # Flag extreme movements for inspection, not automatic deletion.
    extreme = month_end[
        month_end["monthly_return"].abs() > 0.50
    ].copy()

    extreme = extreme.sort_values(
        "monthly_return", key=lambda s: s.abs(), ascending=False
    )

    print("=" * 75)
    print("EXTREME MONTHLY STOCK RETURNS")
    print("=" * 75)
    print("Number of flagged observations:", len(extreme))

    if extreme.empty:
        print("No monthly returns exceeded the 50% threshold.")
        return

    columns = [
        "symbol",
        "previous_month",
        "month",
        "previous_price",
        "adjusted_close",
        "monthly_return",
        "date",
    ]

    print(
        extreme[columns].to_string(
            index=False,
            formatters={
                "previous_price": lambda x: f"{x:.4f}",
                "adjusted_close": lambda x: f"{x:.4f}",
                "monthly_return": lambda x: f"{x:.2%}",
            },
        )
    )

    print(
        "\nThese observations are flagged for investigation, "
        "not automatically classified as errors."
    )


if __name__ == "__main__":
    main()
