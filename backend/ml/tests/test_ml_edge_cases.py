"""
Suite 5: Edge Cases, Boundary Values & Failure Mode Tests
"""
import pytest
from ml.portfolio_optimizer import optimize_indian_portfolio
from ml.risk_profiler import calculate_user_risk_profile
from ml.predict import predict_return

def test_negative_investment_amount_rejected():
    """Investing negative or zero capital must raise ValueError."""
    with pytest.raises(ValueError, match="Investment amount must be positive"):
        optimize_indian_portfolio(investment_amount=-50000)

def test_invalid_risk_score_rejected():
    """Risk scores outside [1, 10] must raise ValueError."""
    with pytest.raises(ValueError, match="between 1 and 10"):
        optimize_indian_portfolio(investment_amount=100000, risk_score=15)

def test_silver_ticker_query_raises_exception():
    """Querying Silver 'SLV' via prediction endpoint must explicitly fail as designed."""
    with pytest.raises(ValueError, match="Silver.*deliberately excluded"):
        predict_return("SLV")

def test_risk_profiler_graceful_bounds():
    """Unrealistic user ages or horizons must gracefully normalize without crash."""
    res_underage = calculate_user_risk_profile(age=12, investment_horizon_years=-5, savings_to_income_ratio=-2.0)
    assert res_underage["age"] == 30  # Fallback to default
    assert res_underage["investment_horizon_years"] == 1  # Clipped to minimum 1
    assert res_underage["continuous_risk_score_Ru"] >= 0.05

