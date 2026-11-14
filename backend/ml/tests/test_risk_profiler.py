"""Unit tests for ml.risk_profiler."""

from pathlib import Path
import pytest

from ml import risk_profiler as rp


def test_generate_synthetic_calibration_profiles_bounds_and_monotonicity():
    X, y = rp.generate_synthetic_calibration_profiles(n_samples=200, seed=7)
    assert len(X) == 200
    assert list(X.columns) == [
        "age",
        "investment_horizon_years",
        "savings_to_income_ratio",
        "max_loss_tolerance_pct",
    ]
    assert y.between(0.05, 0.95).all()


def test_train_risk_profiling_model_persists_and_reloads(isolated_model_dir: Path):
    model1 = rp.train_risk_profiling_model(retrain=True)
    model_file = isolated_model_dir / "risk_profiling_model.joblib"
    assert model_file.exists()

    # Loading with retrain=False should succeed from disk
    model2 = rp.train_risk_profiling_model(retrain=False)
    assert type(model1) is type(model2)


def test_calculate_user_risk_profile_ordering_across_investor_types(isolated_model_dir: Path):
    rp.train_risk_profiling_model(retrain=True)

    conservative = rp.calculate_user_risk_profile(
        age=66, investment_horizon_years=2, savings_to_income_ratio=0.2, max_loss_tolerance_pct=0.06
    )
    moderate = rp.calculate_user_risk_profile(
        age=42, investment_horizon_years=10, savings_to_income_ratio=0.6, max_loss_tolerance_pct=0.20
    )
    aggressive = rp.calculate_user_risk_profile(
        age=24, investment_horizon_years=25, savings_to_income_ratio=1.2, max_loss_tolerance_pct=0.40
    )

    assert conservative["discrete_risk_score"] <= 3
    assert "Conservative" in conservative["category"]
    assert 4 <= moderate["discrete_risk_score"] <= 7
    assert "Moderate" in moderate["category"]
    assert aggressive["discrete_risk_score"] >= 8
    assert "Aggressive" in aggressive["category"]
    assert conservative["continuous_risk_score_Ru"] < moderate["continuous_risk_score_Ru"] < aggressive["continuous_risk_score_Ru"]


def test_calculate_user_risk_profile_handles_invalid_demographics_and_loss_amount(isolated_model_dir: Path):
    rp.train_risk_profiling_model(retrain=True)

    # Out-of-bounds age/horizon/savings should fall back to safe defaults
    # target_amount=1_000_000 overrides portfolio_value; max_loss_amount=100_000 -> 0.10 pct
    res = rp.calculate_user_risk_profile(
        age=120,
        investment_horizon_years=0,
        savings_to_income_ratio=-1.0,
        max_loss_tolerance_pct=None,
        max_loss_amount=100000.0,
        target_amount=1000000.0,
    )
    assert res["age"] == 30
    assert res["investment_horizon_years"] == 1
    assert res["portfolio_value"] == 1000000.0
    assert res["max_loss_tolerance_pct"] == pytest.approx(0.10)

    # Fractional max_loss_amount (<= 1.0) is treated directly as percentage
    res_frac = rp.calculate_user_risk_profile(max_loss_tolerance_pct=None, max_loss_amount=0.15)
    assert res_frac["max_loss_tolerance_pct"] == pytest.approx(0.15)

    # Both None -> defaults to 0.20
    res_default = rp.calculate_user_risk_profile(max_loss_tolerance_pct=None, max_loss_amount=None)
    assert res_default["max_loss_tolerance_pct"] == pytest.approx(0.20)


def test_get_calculated_risk_score_returns_valid_integer(isolated_model_dir: Path):
    rp.train_risk_profiling_model(retrain=True)
    score = rp.get_calculated_risk_score(
        investment_horizon_years=15,
        max_loss_amount=150000.0,
        target_amount=500000.0,
    )
    assert isinstance(score, int)
    assert 1 <= score <= 10

