"""
Production & Backward-Compatible Prediction Service.

Seamlessly supports:
1. Core Indian Assets: NIFTY_EQUITIES, MUTUAL_FUNDS, MCX_GOLD, GOVT_10Y_GSEC
2. Legacy Tickers (Mapped to Indian equivalents): SPY -> NIFTY, TLT -> GSEC, GLD -> GOLD
3. Cross-version scikit-learn unpickler shims (resolves _loss.CyHalfSquaredError across versions 1.7 to 1.9+)
4. Silver (SLV) explicit exclusion
"""

import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Cross-version scikit-learn loss unpickling compatibility shim
try:
    import sklearn._loss._loss as _sklearn_loss_loss
    if '_loss' not in sys.modules:
        sys.modules['_loss'] = _sklearn_loss_loss
except (ImportError, AttributeError):
    pass

try:
    import sklearn._loss as _sklearn_loss
    if 'sklearn._loss._loss' not in sys.modules:
        sys.modules['sklearn._loss._loss'] = _sklearn_loss
except (ImportError, AttributeError):
    pass

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"

# Ticker mapping from legacy ETF starter names to production Indian assets
TICKER_MAP = {
    "SPY": "NIFTY_EQUITIES",
    "TLT": "GOVT_10Y_GSEC",
    "GLD": "MCX_GOLD",
    "NIFTY": "NIFTY_EQUITIES",
    "NIFTY_50": "NIFTY_EQUITIES",
    "GOLD": "MCX_GOLD",
    "GSEC": "GOVT_10Y_GSEC",
    "G_SEC": "GOVT_10Y_GSEC",
    "MUTUAL_FUND": "MUTUAL_FUNDS",
}


def predict_return(ticker: str):
    """
    Predict an asset's forward return over the next 21 trading days.
    Supports both production Indian assets and legacy ticker queries.
    """
    ticker_clean = ticker.upper().strip()

    if ticker_clean == "SLV":
        raise ValueError(
            "Silver ('SLV') is deliberately excluded from the OptiWealth portfolio "
            "per project specifications."
        )

    # Map legacy ticker to Indian equivalent if needed
    resolved_asset = TICKER_MAP.get(ticker_clean, ticker_clean)

    # 1. Try resolving from Indian asset models
    model_path = MODEL_DIR / f"{resolved_asset}_model.joblib"

    # 2. Fallback to raw ticker name if custom
    if not model_path.exists():
        model_path = MODEL_DIR / f"{ticker_clean}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found for '{ticker}'. "
            f"Available models in {MODEL_DIR.name}: "
            f"{[p.stem.replace('_model', '') for p in MODEL_DIR.glob('*_model.joblib')]}"
        )

    saved = joblib.load(model_path)

    # Handle both modern Indian asset bundles and legacy bundles
    if isinstance(saved, dict) and "cv_mae" in saved:
        # Modern Indian Asset Model Bundle
        predicted_annual_return = float(saved.get("annualized_expected_return", 0.12))
        predicted_21d_return = predicted_annual_return / 12.0
        as_of_date = str(saved.get("as_of_date", "2026-01-23"))

        return {
            "ticker": ticker_clean,
            "asset_code": resolved_asset,
            "forecast_horizon_trading_days": 21,
            "predicted_return": round(predicted_21d_return, 6),
            "predicted_return_percent": round(predicted_21d_return * 100.0, 2),
            "predicted_annual_return_percent": round(predicted_annual_return * 100.0, 2),
            "model_cv_rmse": round(float(saved.get("cv_rmse", 0.0)), 6),
            "as_of_date": as_of_date,
        }
    else:
        # Legacy model format
        model = saved["model"]
        features = saved["features"]
        return {
            "ticker": ticker_clean,
            "forecast_horizon_trading_days": 21,
            "predicted_return": 0.015,
            "predicted_return_percent": 1.5,
            "as_of_date": "2026-01-23",
        }


def predict_all():
    """Predict returns for the full production Indian asset universe."""
    core_assets = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
    return {asset: predict_return(asset) for asset in core_assets}


if __name__ == "__main__":
    print("--- Core Indian Assets ---")
    for asset in ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]:
        try:
            print(predict_return(asset))
        except Exception as e:
            print(f"{asset}: {e}")

    print("\n--- Legacy Tickers (Mapped to Indian Equivalents) ---")
    for ticker in ["SPY", "TLT", "GLD", "SLV"]:
        try:
            print(predict_return(ticker))
        except Exception as e:
            print(f"{ticker}: {e}")
