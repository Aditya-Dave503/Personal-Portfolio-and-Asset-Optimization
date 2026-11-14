"""Shared pytest fixtures for the OptiWealth ML pipeline test suite."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure `backend/` is on sys.path so `ml.*` relative imports resolve cleanly.
ML_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ML_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml import ml_engine, predict, risk_profiler  # noqa: E402


@pytest.fixture()
def isolated_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect model persistence directories to a temporary path."""
    model_dir = tmp_path / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(ml_engine, "MODEL_DIR", model_dir)
    monkeypatch.setattr(risk_profiler, "MODEL_DIR", model_dir)
    monkeypatch.setattr(risk_profiler, "MODEL_PATH", model_dir / "risk_profiling_model.joblib")
    monkeypatch.setattr(predict, "MODEL_DIR", model_dir)
    return model_dir


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    """
    Create a deterministic 320-trading-day synthetic Indian market dataset
    in a temporary directory, including realistic dirty rows (duplicates,
    non-positive prices, invalid dates) to exercise data-cleaning paths.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(12345)
    dates = pd.bdate_range("2024-01-01", periods=320)
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # 1. Nifty 50 Equities (2 constituents + dirty rows)
    nifty_rows = []
    p1, p2 = 1000.0, 2000.0
    for d in date_strs:
        p1 *= 1.0 + rng.normal(0.0005, 0.011)
        p2 *= 1.0 + rng.normal(0.0004, 0.012)
        nifty_rows.append({"date": d, "symbol": "RELIANCE.NS", "adjusted_close": round(p1, 4)})
        nifty_rows.append({"date": d, "symbol": "TCS.NS", "adjusted_close": round(p2, 4)})
    # Add duplicate, non-positive price, and invalid date
    nifty_rows.append({"date": date_strs[5], "symbol": "RELIANCE.NS", "adjusted_close": 999.0})
    nifty_rows.append({"date": date_strs[6], "symbol": "TCS.NS", "adjusted_close": -10.0})
    nifty_rows.append({"date": "not-a-date", "symbol": "TCS.NS", "adjusted_close": 100.0})
    pd.DataFrame(nifty_rows).to_csv(raw_dir / "equity_daily_returns_nifty50.csv", index=False)

    # 2. Mutual Funds (2 schemes + non-positive NAV)
    mf_rows = []
    n1, n2 = 50.0, 100.0
    for d in date_strs:
        n1 *= 1.0 + rng.normal(0.0004, 0.008)
        n2 *= 1.0 + rng.normal(0.00045, 0.009)
        mf_rows.append({"date": d, "scheme_code": "1001", "nav": round(n1, 4)})
        mf_rows.append({"date": d, "scheme_code": "1002", "nav": round(n2, 4)})
    mf_rows.append({"date": date_strs[10], "scheme_code": "1001", "nav": 0.0})
    pd.DataFrame(mf_rows).to_csv(raw_dir / "mutual_fund_nav_returns.csv", index=False)

    # 3. MCX Gold (using Date / Adj Close columns)
    gold_rows = []
    g = 60000.0
    for d in date_strs:
        g *= 1.0 + rng.normal(0.0003, 0.007)
        gold_rows.append({"Date": d, "Adj Close": round(g, 2)})
    pd.DataFrame(gold_rows).to_csv(raw_dir / "gold_data_unvalidated.csv", index=False)

    # 4. 10Y G-Sec yield changes
    gsec_rows = []
    y = 7.10
    for d in date_strs:
        dy_pct = rng.normal(0.0, 0.015)
        y = float(np.clip(y + dy_pct, 5.5, 8.5))
        gsec_rows.append({
            "date": d,
            "yield_pct": round(y, 4),
            "yield_change_bps": round(dy_pct * 100.0, 2),
        })
    pd.DataFrame(gsec_rows).to_csv(raw_dir / "india_10y_gsec_yield_changes.csv", index=False)

    return raw_dir


@pytest.fixture()
def sample_ml_predictions() -> dict:
    """Deterministic ML forecast dictionary for optimizer unit tests."""
    return {
        "NIFTY_EQUITIES": {
            "annualized_expected_return": 0.14,
            "pred_21d_pct": 1.17,
            "annualized_expected_return_pct": 14.0,
            "residual_variance_21d": 0.0015,
        },
        "MUTUAL_FUNDS": {
            "annualized_expected_return": 0.12,
            "pred_21d_pct": 1.00,
            "annualized_expected_return_pct": 12.0,
            "residual_variance_21d": 0.0012,
        },
        "MCX_GOLD": {
            "annualized_expected_return": 0.09,
            "pred_21d_pct": 0.75,
            "annualized_expected_return_pct": 9.0,
            "residual_variance_21d": 0.0010,
        },
        "GOVT_10Y_GSEC": {
            "annualized_expected_return": 0.072,
            "pred_21d_pct": 0.60,
            "annualized_expected_return_pct": 7.2,
            "residual_variance_21d": 0.0004,
        },
    }

