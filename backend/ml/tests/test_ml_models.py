"""
Suite 3: ML Model Training, Evaluation Metrics & Prediction Sanity Tests
"""
import pytest
import numpy as np
from ml.ml_engine import train_asset_model, predict_asset_return
from ml.indian_market_data import resolve_data_directory, get_synchronized_market_data, load_macro_yield_features
from ml.risk_profiler import calculate_user_risk_profile

@pytest.fixture(scope="module")
def data_setup():
    data_dir = resolve_data_directory()
    market_df = get_synchronized_market_data(data_dir)
    macro_df = load_macro_yield_features(data_dir)
    return market_df, macro_df

def test_walk_forward_cv_metrics(data_setup):
    """Verify walk-forward cross validation computes MAE, RMSE, hit rate, and positive residual variance."""
    market_df, macro_df = data_setup
    bundle = train_asset_model(
        asset_name="NIFTY_EQUITIES",
        returns_series=market_df["NIFTY_EQUITIES"],
        macro_df=macro_df,
        n_splits=3
    )

    assert bundle["cv_mae"] > 0
    assert bundle["cv_rmse"] > 0
    assert 0.0 <= bundle["hit_rate"] <= 1.0
    assert bundle["residual_variance_21d"] > 0, "Residual variance must be strictly positive!"

def test_prediction_bounded_output(data_setup):
    """Verify that predictions are clamped to realistic institutional bounds [-25%, +35%]."""
    market_df, macro_df = data_setup
    pred = predict_asset_return("NIFTY_EQUITIES", market_df["NIFTY_EQUITIES"], macro_df)

    assert -0.25 <= pred["annualized_expected_return"] <= 0.35
    assert pred["annualized_expected_return_pct"] == round(pred["annualized_expected_return"] * 100.0, 2)
    assert pred["residual_variance_21d"] > 0

def test_risk_profiling_classification():
    """Verify psychometric risk profiler outputs proper continuous and discrete ranges."""
    profile_young = calculate_user_risk_profile(age=22, investment_horizon_years=25, savings_to_income_ratio=1.0, max_loss_tolerance_pct=0.35)
    profile_retiree = calculate_user_risk_profile(age=70, investment_horizon_years=2, savings_to_income_ratio=0.2, max_loss_tolerance_pct=0.05)

    assert 0.05 <= profile_young["continuous_risk_score_Ru"] <= 0.95
    assert 1 <= profile_young["discrete_risk_score"] <= 10
    assert profile_young["discrete_risk_score"] > profile_retiree["discrete_risk_score"], "Young aggressive profile must have higher risk score than retiree!"

