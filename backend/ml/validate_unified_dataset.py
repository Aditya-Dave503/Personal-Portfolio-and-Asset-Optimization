
from pathlib import Path
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"

ASSET_FILE = PROCESSED_DIR / "monthly_asset_returns.csv"
GSEC_FILE = PROCESSED_DIR / "monthly_gsec_features.csv"
GOLD_FILE = PROCESSED_DIR / "gold_monthly_returns_pending_verification.csv"


def validate_returns():
    print("=" * 65)
    print("MONTHLY ASSET RETURNS")
    print("=" * 65)

    returns = pd.read_csv(ASSET_FILE, index_col="month")
    returns = returns.apply(pd.to_numeric, errors="coerce")

    print("Shape:", returns.shape)
    print("Date range:", returns.index.min(), "to", returns.index.max())
    print("Stock assets:", sum(c.startswith("STOCK_") for c in returns.columns))
    print("Fund assets:", sum(c.startswith("FUND_") for c in returns.columns))
    print("Total missing return values:", int(returns.isna().sum().sum()))

    print("\nValid monthly observations per asset:")
    print(returns.notna().sum().sort_values().to_string())

    print("\nAssets with fewer than 60 valid monthly returns:")
    print(returns.notna().sum()[returns.notna().sum() < 60].to_string())

    print("\nMonthly return range per asset:")
    print(returns.agg(["min", "max"]).T.sort_values("min").to_string())

    extreme = (returns.abs() > 0.50).sum()
    print("\nAssets with monthly returns above 50% in magnitude:")
    print(extreme[extreme > 0].to_string())

    print("\nDuplicate month labels:", returns.index.duplicated().sum())


def validate_gsec():
    print("\n" + "=" * 65)
    print("G-SEC FEATURES")
    print("=" * 65)

    gsec = pd.read_csv(GSEC_FILE, index_col="month")
    print("Shape:", gsec.shape)
    print(gsec.describe().to_string())
    print("\nMissing values:")
    print(gsec.isna().sum().to_string())


def validate_gold():
    print("\n" + "=" * 65)
    print("GOLD — PENDING VERIFICATION")
    print("=" * 65)

    gold = pd.read_csv(GOLD_FILE, index_col="month")
    print("Shape:", gold.shape)
    print(gold.describe().to_string())
    print("\nMissing values:")
    print(gold.isna().sum().to_string())


if __name__ == "__main__":
    validate_returns()
    validate_gsec()
    validate_gold()
    print("\nValidation report completed. No files were modified.")
