
from pathlib import Path
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BACKEND_DIR / "data" / "raw"

FILES = {
    "NIFTY 50 EQUITIES": (
        "equity_daily_returns_nifty50.csv",
        "date"
    ),
    "MUTUAL FUNDS": (
        "mutual_fund_nav_returns.csv",
        "date"
    ),
    "GOLD": (
        "gold_data_unvalidated.csv",
        "Date"
    ),
    "INDIA 10Y G-SEC YIELDS": (
        "india_10y_gsec_yield_changes.csv",
        "date"
    ),
}


def audit_file(name, filename, date_column):
    path = RAW_DIR / filename

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if not path.exists():
        print("ERROR: File not found:", path)
        return

    df = pd.read_csv(path)

    print("File:", filename)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("Duplicate rows:", df.duplicated().sum())

    print("\nMissing values by column:")
    print(df.isna().sum().to_string())

    if date_column in df.columns:
        dates = pd.to_datetime(
            df[date_column],
            errors="coerce"
        )

        print("\nFirst date:", dates.min())
        print("Last date:", dates.max())
        print("Invalid dates:", dates.isna().sum())

    if name == "NIFTY 50 EQUITIES":
        print("\nUnique stocks:", df["symbol"].nunique())

        print(
            "Example symbols:",
            df["symbol"].dropna().unique()[:10]
        )

        if {"date", "symbol"}.issubset(df.columns):
            duplicates = df.duplicated(
                ["date", "symbol"]
            ).sum()

            print("Duplicate date-symbol pairs:", duplicates)

    elif name == "MUTUAL FUNDS":
        print("\nUnique schemes:", df["scheme_code"].nunique())

        columns = [
            col for col in
            ["scheme_code", "scheme_name", "category"]
            if col in df.columns
        ]

        print(
            df[columns]
            .drop_duplicates()
            .to_string(index=False)
        )

        if {"date", "scheme_code"}.issubset(df.columns):
            duplicates = df.duplicated(
                ["date", "scheme_code"]
            ).sum()

            print("Duplicate date-scheme pairs:", duplicates)

    elif name == "GOLD":
        print("\nGold data sample:")
        print(df.head(3).to_string(index=False))

        if "Adj Close" in df.columns:
            prices = pd.to_numeric(
                df["Adj Close"],
                errors="coerce"
            )

            returns = prices.pct_change(fill_method=None)

            print(
                "\nRecomputed adjusted-price return range:",
                returns.min(),
                "to",
                returns.max()
            )

        if "Daily_Return" in df.columns:
            original = pd.to_numeric(
                df["Daily_Return"],
                errors="coerce"
            )

            print(
                "Original Daily_Return range:",
                original.min(),
                "to",
                original.max()
            )

        print(
            "\nIMPORTANT: Verify the gold instrument, currency, "
            "units and source before including it."
        )

    elif name == "INDIA 10Y G-SEC YIELDS":
        columns = [
            col for col in df.columns
            if "yield" in col.lower()
            or "change" in col.lower()
        ]

        print("\nYield/change columns:", columns)

        if "yield_pct" in df.columns:
            yields = pd.to_numeric(
                df["yield_pct"],
                errors="coerce"
            )

            print(
                "Yield range (%):",
                yields.min(),
                "to",
                yields.max()
            )

        if "yield_change_bps" in df.columns:
            changes = pd.to_numeric(
                df["yield_change_bps"],
                errors="coerce"
            )

            print(
                "Yield change range (bps):",
                changes.min(),
                "to",
                changes.max()
            )


if __name__ == "__main__":
    for name, (filename, date_column) in FILES.items():
        audit_file(name, filename, date_column)

    print("\nAudit completed. No files were modified.")
