"""
OptiWealth ML Model Training & Artifact Serialization Pipeline.

Retrains and serializes all 5 production models natively under the host
environment's installed scikit-learn version:
1. NIFTY_EQUITIES_model.joblib
2. MUTUAL_FUNDS_model.joblib
3. MCX_GOLD_model.joblib
4. GOVT_10Y_GSEC_model.joblib
5. risk_profiling_model.joblib
"""

import sys
from pathlib import Path

# Add current ml package directory to sys.path
_ML_DIR = str(Path(__file__).resolve().parent)
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from indian_market_data import resolve_data_directory
from ml_engine import train_and_forecast_all
from risk_profiler import train_risk_profiling_model


def retrain_all_production_models():
    print("=" * 65)
    print("OptiWealth: Training Production Indian Asset Models")
    print("=" * 65)

    # 1. Retrain Risk Profiler
    print("[1/2] Training Investor Risk Profiling Model (Section 1)...")
    train_risk_profiling_model(retrain=True)
    print("      Done. Saved risk_profiling_model.joblib")

    # 2. Retrain 4 Indian Asset Models
    print("[2/2] Training 4 Core Indian Asset Models (Section 2 & 4)...")
    data_dir = resolve_data_directory()
    predictions, market_df = train_and_forecast_all(data_dir=data_dir, retrain=True)
    for asset, v in predictions.items():
        print(f"      - {asset}: Ann. Return={v.get('annualized_expected_return', 0.12):.2%}, 21d Forecast={v.get('predicted_21d_return', 0.01):.2%}")

    print("=" * 65)
    print("All production model artifacts successfully serialized!")
    print("=" * 65)


if __name__ == "__main__":
    retrain_all_production_models()
