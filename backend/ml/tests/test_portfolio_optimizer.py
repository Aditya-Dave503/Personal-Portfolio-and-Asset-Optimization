"""Unit tests for ml.portfolio_optimizer."""

from pathlib import Path
import numpy as np
import pytest

from ml import indian_market_data as imd
from ml import portfolio_optimizer as po


def test_compute_shrinkage_covariance_is_symmetric_and_psd(synthetic_data_dir: Path):
    market_df = imd.get_synchronized_market_data(synthetic_data_dir)
    cov = po.compute_shrinkage_covariance(market_df)
    assert cov.shape == (4, 4)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-10)


@pytest.mark.parametrize("bad_score", [0, 11, -3])
def test_get_risk_profile_parameters_invalid_score_raises(bad_score: int):
    with pytest.raises(ValueError, match="between 1 and 10"):
        po.get_risk_profile_parameters(bad_score, ["NIFTY_EQUITIES"])


def test_get_risk_profile_parameters_bounds_and_aversion():
    assets = ["NIFTY_EQUITIES", "MUTUAL_FUNDS", "MCX_GOLD", "GOVT_10Y_GSEC", "OTHER_ASSET"]
    lam_cons, b_cons = po.get_risk_profile_parameters(2, assets)
    lam_mod, b_mod = po.get_risk_profile_parameters(5, assets)
    lam_agg, b_agg = po.get_risk_profile_parameters(9, assets)

    assert lam_cons > lam_mod > lam_agg
    # Conservative should require at least 40% G-Sec; aggressive caps G-Sec at 25%
    assert b_cons[3] == (0.40, 0.75)
    assert b_mod[3] == (0.15, 0.50)
    assert b_agg[3] == (0.05, 0.25)
    # Fallback bound for unknown asset
    assert b_mod[4] == (0.05, 0.50)


def test_compute_black_litterman_blends_prior_and_views(synthetic_data_dir: Path, sample_ml_predictions: dict):
    market_df = imd.get_synchronized_market_data(synthetic_data_dir)
    assets = list(market_df.columns)
    cov = po.compute_shrinkage_covariance(market_df)

    mu_bl, cov_bl = po.compute_black_litterman(
        asset_names=assets,
        cov_matrix=cov,
        ml_predictions=sample_ml_predictions,
        risk_aversion=8.4,
    )
    assert mu_bl.shape == (4,)
    assert cov_bl.shape == (4, 4)
    assert np.all(np.isfinite(mu_bl))


def test_optimize_indian_portfolio_invalid_amount_raises(synthetic_data_dir: Path):
    with pytest.raises(ValueError, match="Investment amount must be positive"):
        po.optimize_indian_portfolio(investment_amount=0.0, data_dir=synthetic_data_dir)


@pytest.mark.parametrize("mode", ["black_litterman", "max_sharpe", "min_variance"])
def test_optimize_indian_portfolio_all_modes_satisfy_constraints(
    synthetic_data_dir: Path, sample_ml_predictions: dict, mode: str
):
    result = po.optimize_indian_portfolio(
        investment_amount=500000.0,
        risk_score=5,
        optimization_mode=mode,
        max_loss_tolerance_pct=0.20,
        data_dir=synthetic_data_dir,
        custom_predictions=sample_ml_predictions,
    )

    weights = [a["weight_percent"] for a in result["allocations"]]
    amounts = [a["allocated_amount"] for a in result["allocations"]]

    assert sum(weights) == pytest.approx(100.0, abs=0.1)
    assert sum(amounts) == pytest.approx(500000.0, abs=100.0)
    assert all(w >= 0.0 for w in weights)
    assert result["var_99_constraint_status"] == "SATISFIED"
    assert result["risk_profile_category"] == "Moderate / Balanced"


def test_optimize_indian_portfolio_strict_var_triggers_fallback_or_violation(
    synthetic_data_dir: Path, sample_ml_predictions: dict
):
    # Extremely tight 0.1% VaR limit is impossible given minimum equity bounds
    result = po.optimize_indian_portfolio(
        investment_amount=500000.0,
        risk_score=9,
        optimization_mode="black_litterman",
        max_loss_tolerance_pct=0.001,
        data_dir=synthetic_data_dir,
        custom_predictions=sample_ml_predictions,
    )
    assert result["risk_profile_category"] == "Aggressive Growth"
    assert "VIOLATED" in result["var_99_constraint_status"]
    assert sum(a["weight_percent"] for a in result["allocations"]) == pytest.approx(100.0, abs=0.1)


def test_optimize_indian_portfolio_without_custom_predictions_runs_ml(
    synthetic_data_dir: Path, isolated_model_dir: Path
):
    result = po.optimize_indian_portfolio(
        investment_amount=250000.0,
        risk_score=2,
        max_loss_tolerance_pct=None,
        data_dir=synthetic_data_dir,
        retrain_ml=True,
        custom_predictions=None,
    )
    assert result["risk_profile_category"] == "Conservative"
    assert result["var_99_constraint_status"] == "SATISFIED"


def test_optimize_indian_portfolio_auto_dir_and_solver_failure(
    synthetic_data_dir: Path, sample_ml_predictions: dict, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(po, "resolve_data_directory", lambda: synthetic_data_dir)
    res = po.optimize_indian_portfolio(
        investment_amount=100000.0,
        risk_score=5,
        data_dir=None,
        custom_predictions=sample_ml_predictions,
    )
    assert res["investment_amount"] == 100000.0

    class _FailedOpt:
        success = False
        message = "Simulated solver failure"

    monkeypatch.setattr(po, "minimize", lambda *args, **kwargs: _FailedOpt())
    with pytest.raises(RuntimeError, match="Optimization failed: Simulated solver failure"):
        po.optimize_indian_portfolio(
            investment_amount=100000.0,
            risk_score=5,
            data_dir=synthetic_data_dir,
            custom_predictions=sample_ml_predictions,
        )

