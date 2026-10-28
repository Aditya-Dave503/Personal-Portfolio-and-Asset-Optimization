
from pathlib import Path
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    BACKEND_DIR
    / "data"
    / "raw"
    / "equity_daily_returns_nifty50.csv"
)

SUSPECTS = [
    ("ADANIENT", "2018-07"),
    ("ADANIENT", "2021-02"),
    ("ADANIENT", "2020-08"),
    ("BAJAJFINSV", "2020-11"),
    ("SHRIRAMFIN", "2020-11"),
    ("ADANIENT", "2023-02"),
    ("JSWSTEEL", "2021-04"),
    ("TMPV", "2019-10"),
    ("HINDALCO", "2021-02"),
    ("BAJFINANCE", "2020-03"),
]


def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])

    numeric_columns = [
        "adjusted_close",
        "close",
        "daily_return_adj_close",
        "daily_return_close",
        "adjustment_ratio",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    for symbol, month in SUSPECTS:
        period = pd.Period(month, freq="M")

        # Show the month containing the unusual return and
        # the previous month for comparison.
        start = (period - 1).start_time
        end = period.end_time

        subset = df[
            (df["symbol"] == symbol)
            & (df["date"] >= start)
            & (df["date"] <= end)
        ].sort_values("date")

        print("\n" + "=" * 90)
        print(f"{symbol} | flagged month: {month}")
        print("=" * 90)

        if subset.empty:
            print("No matching rows found.")
            continue

        columns = [
            "date",
            "close",
            "adjusted_close",
            "adjustment_ratio",
            "daily_return_close",
            "daily_return_adj_close",
        ]
        columns = [
            column for column in columns
            if column in subset.columns
        ]

        print(
            subset[columns].to_string(
                index=False,
                formatters={
                    "close": lambda x: f"{x:.4f}",
                    "adjusted_close": lambda x: f"{x:.4f}",
                    "adjustment_ratio": lambda x: f"{x:.6f}",
                    "daily_return_close": lambda x: f"{x:.4%}",
                    "daily_return_adj_close": lambda x: f"{x:.4%}",
                },
            )
        )

    print("\nInspection completed. No files were modified.")


if __name__ == "__main__":
    main()
