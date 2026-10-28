from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

files = [
    "equity_daily_returns_nifty50.csv",
    "mutual_fund_nav_returns.csv",
    "gold_data_unvalidated.csv",
    "india_10y_gsec_yield_changes.csv",
]

for filename in files:
    print(f"\n--- {filename} ---")
    df = pd.read_csv(DATA_DIR / filename)
    print("Columns:", df.columns.tolist())
    print(df.head(3).to_string(index=False))