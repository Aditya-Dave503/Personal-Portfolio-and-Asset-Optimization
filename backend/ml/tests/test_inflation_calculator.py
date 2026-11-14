"""
Inflation-Adjusted Purchasing Power & Real Return Tests
Verifies Fisher equation calculations and real wealth discounting.
"""
import pytest
from ml.inflation_calculator import (
    calculate_real_return,
    compute_inflation_adjusted_wealth,
)


def test_fisher_equation_nominal_equals_inflation():
    """If nominal return equals inflation rate, real return must be exactly 0.0%."""
    r_real = calculate_real_return(nominal_return=0.06, inflation_rate=0.06)
    assert abs(r_real) < 1e-6


def test_zero_inflation_real_equals_nominal():
    """When inflation is 0%, real return equals nominal return."""
    r_real = calculate_real_return(nominal_return=0.12, inflation_rate=0.0)
    assert r_real == pytest.approx(0.12)


def test_negative_real_return_purchasing_power_erosion():
    """When inflation exceeds return (e.g. savings account 3% vs inflation 6%), real return is negative."""
    r_real = calculate_real_return(nominal_return=0.03, inflation_rate=0.06)
    assert r_real < 0.0


def test_inflation_adjusted_wealth_projection():
    """Verify purchasing power discounting on a 10-year investment."""
    res = compute_inflation_adjusted_wealth(
        starting_capital=100000.0,
        annual_nominal_return=0.10,
        horizon_years=10,
        annual_inflation_rate=0.05,
    )
    # 100k at 10% for 10y = 100k * (1.10^10) = 259,374.25
    assert res.nominal_future_wealth > res.starting_capital
    # Real wealth in today's rupees = Nominal / (1.05^10) = 159,255.70
    assert res.real_purchasing_power_today < res.nominal_future_wealth
    assert res.real_purchasing_power_today > res.starting_capital
    assert res.inflation_drag_rupees > 0
    assert res.purchasing_power_loss_pct > 0
