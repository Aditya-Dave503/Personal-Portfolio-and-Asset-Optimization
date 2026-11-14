"""
Inflation-Adjusted Purchasing Power & Real Return Calculator

Applies the Fisher Equation to eliminate 'money illusion', discounting nominal portfolio
projections and milestone financial goals by the Reserve Bank of India CPI inflation rate.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class RealWealthProjection:
    investment_horizon_years: int
    annual_nominal_return_pct: float
    annual_inflation_rate_pct: float
    annual_real_return_pct: float
    starting_capital: float
    nominal_future_wealth: float
    real_purchasing_power_today: float
    inflation_drag_rupees: float
    purchasing_power_loss_pct: float


def calculate_real_return(
    nominal_return: float,
    inflation_rate: float = 0.055,
) -> float:
    """
    Exact Fisher Equation: r_real = (1 + r_nominal) / (1 + i) - 1.
    Default inflation rate is 5.5% (RBI Indian CPI long-term target band).
    """
    if inflation_rate <= -1.0:
        raise ValueError("Inflation rate cannot be <= -100%.")
    return float((1.0 + float(nominal_return)) / (1.0 + float(inflation_rate)) - 1.0)


# Backwards compatibility alias
compute_fisher_real_return = calculate_real_return


def compute_inflation_adjusted_wealth(
    starting_capital: float,
    annual_nominal_return: float,
    horizon_years: int,
    annual_inflation_rate: float = 0.055,
    monthly_sip: float = 0.0,
) -> RealWealthProjection:
    """
    Projects multi-year nominal wealth and discounts it back to real purchasing power.
    """
    if starting_capital < 0:
        raise ValueError("Starting capital must be non-negative.")
    if horizon_years <= 0:
        raise ValueError("Horizon years must be strictly positive.")

    r_real = calculate_real_return(annual_nominal_return, annual_inflation_rate)

    # 1. Nominal Future Wealth
    nominal_lump = starting_capital * ((1.0 + annual_nominal_return) ** horizon_years)
    if monthly_sip > 0:
        m_rate = annual_nominal_return / 12.0
        n_months = horizon_years * 12
        if m_rate > 0:
            nominal_sip = monthly_sip * (((1.0 + m_rate) ** n_months - 1.0) / m_rate) * (1.0 + m_rate)
        else:
            nominal_sip = monthly_sip * n_months
    else:
        nominal_sip = 0.0

    nominal_total = nominal_lump + nominal_sip

    # 2. Discount back to today's purchasing power using CPI compounding
    discount_factor = (1.0 + annual_inflation_rate) ** horizon_years
    real_power = nominal_total / discount_factor

    inflation_drag = nominal_total - real_power
    loss_pct = (inflation_drag / nominal_total) * 100.0 if nominal_total > 0 else 0.0

    return RealWealthProjection(
        investment_horizon_years=horizon_years,
        annual_nominal_return_pct=round(annual_nominal_return * 100.0, 2),
        annual_inflation_rate_pct=round(annual_inflation_rate * 100.0, 2),
        annual_real_return_pct=round(r_real * 100.0, 2),
        starting_capital=round(starting_capital, 2),
        nominal_future_wealth=round(nominal_total, 2),
        real_purchasing_power_today=round(real_power, 2),
        inflation_drag_rupees=round(inflation_drag, 2),
        purchasing_power_loss_pct=round(loss_pct, 2),
    )


def adjust_portfolio_for_inflation(
    nominal_expected_return: float,
    current_wealth: float,
    horizon_years: int,
    cpi_inflation_rate: float = 0.055
) -> Dict[str, Any]:
    """Decompose forward nominal portfolio accumulation into real purchasing power adjustments."""
    proj = compute_inflation_adjusted_wealth(
        starting_capital=current_wealth,
        annual_nominal_return=nominal_expected_return,
        horizon_years=horizon_years,
        annual_inflation_rate=cpi_inflation_rate,
    )
    return {
        "nominal_annual_return_pct": proj.annual_nominal_return_pct,
        "cpi_inflation_rate_pct": proj.annual_inflation_rate_pct,
        "exact_real_annual_return_pct": proj.annual_real_return_pct,
        "horizon_years": proj.investment_horizon_years,
        "initial_capital_inr": proj.starting_capital,
        "nominal_terminal_wealth_inr": proj.nominal_future_wealth,
        "real_terminal_wealth_inr": proj.real_purchasing_power_today,
        "inflation_drag_loss_inr": proj.inflation_drag_rupees,
        "purchasing_power_retention_pct": round(100.0 - proj.purchasing_power_loss_pct, 2),
        "wealth_multiplier_nominal": round(proj.nominal_future_wealth / max(current_wealth, 1.0), 2),
        "wealth_multiplier_real": round(proj.real_purchasing_power_today / max(current_wealth, 1.0), 2),
    }


def convert_asset_returns_to_real(
    asset_predictions: Dict[str, Dict[str, Any]],
    cpi_inflation_rate: float = 0.055
) -> Dict[str, Dict[str, Any]]:
    """Apply Fisher purchasing power adjustment across all multi-asset forecast projections."""
    real_predictions = {}
    for asset, pred in asset_predictions.items():
        nom_ret = pred.get("annualized_expected_return", 0.10)
        real_ret = calculate_real_return(nom_ret, cpi_inflation_rate)
        real_predictions[asset] = {
            "nominal_annual_return_pct": round(nom_ret * 100.0, 2),
            "real_annual_return_pct": round(real_ret * 100.0, 2),
            "cpi_inflation_pct": round(cpi_inflation_rate * 100.0, 2),
            "is_wealth_generating": bool(real_ret > 0.0),
        }
    return real_predictions
