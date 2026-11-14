"""
Phase 3 & Phase 4 ML Models, Inference Service, and Risk Profiler Test Suite:
- Phase 3: Walk-Forward TimeSeriesSplit Cross-Validation, Residual Variance Calibration,
           Annualized Return Clipping [-25%, +35%], Zero-Staleness Inference,
           Legacy Ticker Mapping (SPY, TLT, GLD, nifty_50), and SLV Rejection.
- Phase 4: Investor Psychometric Risk Profiling Model, Synthetic Calibration Determinism,
           Lifecycle Monotonicity, Dynamic INR Loss Conversion, and Boundary Resilience.
"""

import sys
from pathlib import Path

# Bootstrap sys.path so imports work cleanly from repo root, backend/, or backend/tests/
_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _TESTS_DIR.parent
_REPO_ROOT = _BACKEND_DIR.parent
_ML_DIR = _BACKEND_DIR / "ml"

for _p in (_ML_DIR, _BACKEND_DIR, _REPO_ROOT, _TESTS_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import pytest

from ml.indian_market_data import (
    resolve_data_directory,
    get_synchronized_market_data,
    load_macro_yield_features,
)
from ml.ml_engine import (
    MODEL_DIR,
    train_asset_model,
    predict_asset_return,
    train_and_forecast_all,
)
from ml.predict import (
    TICKER_MAP,
    predict_return,
    predict_all,
)
from ml.risk_profiler import (
    generate_synthetic_calibration_profiles,
    train_risk_profiling_model,
    calculate_user_risk_profile,
    get_calculated_risk_score,
)


# ============================================================================
# SHARED FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def market_and_macro():
    data_dir = resolve_data_directory()
    market_df = get_synchronized_market_data(data_dir)
    macro_df = load_macro_yield_features(data_dir)
    return market_df, macro_df


# ============================================================================
# PHASE 3: ML MODEL TRAINING, WALK-FORWARD CV & INFERENCE TESTS
# ============================================================================

def test_phase3_walk_forward_cv_training_and_metrics(market_and_macro):
    """
    Verify train_asset_model executes chronological walk-forward TimeSeriesSplit CV
    and produces valid MAE, RMSE >= MAE, directional hit rate in [0, 1], and positive residual variance.
    """
    market_df, macro_df = market_and_macro
    bundle = train_asset_model(
        asset_name="NIFTY_EQUITIES",
        returns_series=market_df["NIFTY_EQUITIES"],
        macro_df=macro_df,
        horizon_days=21,
        n_splits=5,
    )

    required_keys = {
        "asset_name",
        "model",
        "feature_cols",
        "horizon_days",
        "cv_mae",
        "cv_rmse",
        "hit_rate",
        "residual_variance_21d",
        "observations",
        "as_of_date",
    }
    assert required_keys.issubset(bundle.keys())
    assert bundle["asset_name"] == "NIFTY_EQUITIES"
    assert bundle["horizon_days"] == 21
    assert len(bundle["feature_cols"]) == 14  # 14 technical features (macro features excluded per Issue #25)
    assert bundle["cv_mae"] > 0.0
    assert bundle["cv_rmse"] >= bundle["cv_mae"]
    assert 0.0 <= bundle["hit_rate"] <= 1.0
    assert bundle["residual_variance_21d"] > 0.0
    assert bundle["observations"] > 1500
    assert (MODEL_DIR / "NIFTY_EQUITIES_model.joblib").exists()


def test_phase3_predict_asset_return_clamping_and_freshness(market_and_macro):
    """
    Verify predict_asset_return uses the latest available market date (zero staleness)
    and clamps annualized expected returns to institutional bounds [-0.25, +0.35].
    """
    market_df, macro_df = market_and_macro
    expected_latest_date = str(market_df.index[-1].date())

    for asset in market_df.columns:
        pred = predict_asset_return(asset, market_df[asset], macro_df)
        assert pred["asset_name"] == asset
        assert pred["as_of_date"] == expected_latest_date
        assert -0.25 <= pred["annualized_expected_return"] <= 0.35
        assert pred["annualized_expected_return_pct"] == round(pred["annualized_expected_return"] * 100.0, 2)
        assert pred["residual_variance_21d"] > 0.0

    # Missing model raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="not found"):
        predict_asset_return("NON_EXISTENT_ASSET_XYZ", market_df["NIFTY_EQUITIES"], macro_df)


def test_phase3_train_and_forecast_all_pipeline(market_and_macro):
    """Verify train_and_forecast_all returns predictions for all 4 synchronized Indian assets."""
    predictions, df = train_and_forecast_all(retrain=False)
    expected_assets = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"]
    assert list(predictions.keys()) == expected_assets
    assert list(df.columns) == expected_assets
    for asset in expected_assets:
        assert -0.25 <= predictions[asset]["annualized_expected_return"] <= 0.35


def test_phase3_predict_service_legacy_mapping_and_slv_rejection():
    """
    Verify predict.py maps legacy ETFs (SPY, TLT, GLD, nifty_50) to Indian assets,
    rejects SLV with ValueError, and raises FileNotFoundError for unknown tickers.
    """
    # Legacy ticker mapping verification
    mapping_cases = {
        "SPY": "NIFTY_EQUITIES",
        "spy": "NIFTY_EQUITIES",
        "TLT": "GOVT_10Y_GSEC",
        "GLD": "MCX_GOLD",
        "nifty_50": "NIFTY_EQUITIES",
        "gold": "MCX_GOLD",
        "gsec": "GOVT_10Y_GSEC",
        "mutual_fund": "MUTUAL_FUNDS",
    }
    for raw_ticker, expected_code in mapping_cases.items():
        res = predict_return(raw_ticker)
        assert res["asset_code"] == expected_code
        assert res["forecast_horizon_trading_days"] == 21
        assert "predicted_return" in res
        assert "predicted_annual_return_percent" in res
        assert res["model_cv_rmse"] > 0.0

    # predict_all() returns all 4 Indian core assets
    all_preds = predict_all()
    assert set(all_preds.keys()) == {"NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC"}

    # Silver ('SLV') explicitly blocked
    for slv_variant in ("SLV", "slv", "  SLV  "):
        with pytest.raises(ValueError, match="Silver.*deliberately excluded"):
            predict_return(slv_variant)

    # Predict on legacy bundle format (SPY_model.joblib if present or synthetic legacy bundle)
    legacy_spy = predict_return("SPY")
    assert legacy_spy["asset_code"] == "NIFTY_EQUITIES"

    # Train risk profiler explicitly with retrain=True to verify serialization path
    retrained_model = train_risk_profiling_model(retrain=True)
    assert retrained_model is not None

    # Unknown ticker raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="No trained model found"):
        predict_return("UNKNOWN_CRYPTO_TICKER")


# ============================================================================
# PHASE 4: INVESTOR PSYCHOMETRIC RISK PROFILER TESTS
# ============================================================================

def test_phase4_synthetic_calibration_determinism_and_bounds():
    """Verify generate_synthetic_calibration_profiles is deterministic and bounded in [0.05, 0.95]."""
    X1, y1 = generate_synthetic_calibration_profiles(n_samples=1000, seed=42)
    X2, y2 = generate_synthetic_calibration_profiles(n_samples=1000, seed=42)

    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2)

    expected_cols = ["age", "investment_horizon_years", "savings_to_income_ratio", "max_loss_tolerance_pct"]
    assert list(X1.columns) == expected_cols
    assert len(X1) == 1000
    assert (y1 >= 0.05).all() and (y1 <= 0.95).all()

    model = train_risk_profiling_model(retrain=False)
    r2 = model.score(X1, y1)
    assert r2 > 0.85, f"Expected Risk Profiler R^2 > 0.85, got {r2:.4f}"


def test_phase4_risk_profiler_lifecycle_monotonicity():
    """
    Verify psychometric model respects behavioral finance monotonicity:
    - Young aggressive investor > Moderate mid-career investor > Older conservative retiree
    - Partial derivatives: younger > older, longer horizon > shorter horizon, higher loss tolerance > lower.
    """
    young_agg = calculate_user_risk_profile(
        age=23, investment_horizon_years=25, savings_to_income_ratio=1.2, max_loss_tolerance_pct=0.40
    )
    mid_mod = calculate_user_risk_profile(
        age=42, investment_horizon_years=12, savings_to_income_ratio=0.6, max_loss_tolerance_pct=0.20
    )
    older_cons = calculate_user_risk_profile(
        age=68, investment_horizon_years=3, savings_to_income_ratio=0.2, max_loss_tolerance_pct=0.06
    )

    for p in (young_agg, mid_mod, older_cons):
        assert 0.05 <= p["continuous_risk_score_Ru"] <= 0.95
        assert 1 <= p["discrete_risk_score"] <= 10

    assert young_agg["continuous_risk_score_Ru"] > mid_mod["continuous_risk_score_Ru"] > older_cons["continuous_risk_score_Ru"]
    assert young_agg["discrete_risk_score"] > mid_mod["discrete_risk_score"] > older_cons["discrete_risk_score"]
    assert "Aggressive" in young_agg["category"]
    assert "Conservative" in older_cons["category"]

    # Single-factor directional checks holding other inputs constant
    p_age_25 = calculate_user_risk_profile(age=25, investment_horizon_years=10, savings_to_income_ratio=0.5, max_loss_tolerance_pct=0.20)
    p_age_65 = calculate_user_risk_profile(age=65, investment_horizon_years=10, savings_to_income_ratio=0.5, max_loss_tolerance_pct=0.20)
    assert p_age_25["continuous_risk_score_Ru"] > p_age_65["continuous_risk_score_Ru"]

    p_hor_25 = calculate_user_risk_profile(age=35, investment_horizon_years=25, savings_to_income_ratio=0.5, max_loss_tolerance_pct=0.20)
    p_hor_2 = calculate_user_risk_profile(age=35, investment_horizon_years=2, savings_to_income_ratio=0.5, max_loss_tolerance_pct=0.20)
    assert p_hor_25["continuous_risk_score_Ru"] > p_hor_2["continuous_risk_score_Ru"]


def test_phase4_dynamic_inr_loss_amount_conversion():
    """
    Verify automatic conversion of INR max_loss_amount against portfolio_value
    and target_amount override.
    """
    # 50,000 INR loss on 500,000 INR portfolio -> 10% loss tolerance
    p1 = calculate_user_risk_profile(max_loss_amount=50000.0, portfolio_value=500000.0)
    assert np.isclose(p1["max_loss_tolerance_pct"], 0.10, atol=1e-6)

    # target_amount (1,000,000 INR) overrides portfolio_value (500,000 INR) -> 5% loss tolerance
    p2 = calculate_user_risk_profile(
        max_loss_amount=50000.0, portfolio_value=500000.0, target_amount=1000000.0
    )
    assert p2["portfolio_value"] == 1000000.0
    assert np.isclose(p2["max_loss_tolerance_pct"], 0.05, atol=1e-6)

    # Fractional max_loss_amount <= 1.0 treated directly as percentage
    p3 = calculate_user_risk_profile(max_loss_amount=0.18)
    assert np.isclose(p3["max_loss_tolerance_pct"], 0.18, atol=1e-6)

    # Extreme INR loss clipped to [0.02, 0.80]
    p_high = calculate_user_risk_profile(max_loss_amount=950000.0, portfolio_value=1000000.0)
    assert np.isclose(p_high["max_loss_tolerance_pct"], 0.80, atol=1e-6)

    p_low = calculate_user_risk_profile(max_loss_amount=500.0, portfolio_value=1000000.0)
    assert np.isclose(p_low["max_loss_tolerance_pct"], 0.02, atol=1e-6)


def test_phase4_out_of_bounds_demographic_normalization_and_entrypoint():
    """
    Verify out-of-bounds demographics normalize gracefully without raising exceptions,
    and get_calculated_risk_score returns an integer in [1, 10].
    """
    res_low = calculate_user_risk_profile(
        age=10,
        investment_horizon_years=-5,
        savings_to_income_ratio=-1.5,
        unexpected_extra_param="ignored_safely",
    )
    assert res_low["age"] == 30
    assert res_low["investment_horizon_years"] == 1
    assert 0.05 <= res_low["continuous_risk_score_Ru"] <= 0.95

    res_high_age = calculate_user_risk_profile(age=150, investment_horizon_years=5)
    assert res_high_age["age"] == 30

    score = get_calculated_risk_score(
        investment_horizon_years=12,
        max_loss_amount=100000.0,
        target_amount=500000.0,
    )
    assert isinstance(score, int)
    assert 1 <= score <= 10
