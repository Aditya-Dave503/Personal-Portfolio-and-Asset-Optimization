"""
Portfolio Drift Detection & Smart Rebalancing Tests
Verifies drift metric, self-financing invariant, and buy/sell trade generation.
"""
import pytest
from ml.rebalancer import calculate_portfolio_drift


def test_rebalancing_self_financing_invariant():
    """Fundamental Invariant: Sum of buy orders must equal sum of sell orders (net cash impact == 0)."""
    holdings = {
        "NIFTY_EQUITIES": 300000.0,  # 60%
        "MUTUAL_FUNDS": 100000.0,    # 20%
        "MCX_GOLD": 50000.0,         # 10%
        "GOVT_10Y_GSEC": 50000.0,    # 10%
    }
    # Target: 40% Equities, 30% MF, 15% Gold, 15% G-Sec
    targets = {
        "NIFTY_EQUITIES": 0.40,
        "MUTUAL_FUNDS": 0.30,
        "MCX_GOLD": 0.15,
        "GOVT_10Y_GSEC": 0.15,
    }

    plan = calculate_portfolio_drift(holdings, targets, threshold_pct=5.0)

    assert plan.total_portfolio_value == 500000.0
    assert plan.rebalance_recommended is True
    # Self-financing invariant: buys equal sells within rounding
    assert abs(plan.net_cash_impact) <= 1.0


def test_drift_below_threshold_does_not_trigger():
    """When portfolio drift is within acceptable tolerance (<5%), no rebalance is triggered."""
    holdings = {
        "NIFTY_EQUITIES": 41000.0,  # 41% (drift +1%)
        "MUTUAL_FUNDS": 29000.0,    # 29% (drift -1%)
        "MCX_GOLD": 15000.0,        # 15% (drift 0%)
        "GOVT_10Y_GSEC": 15000.0,   # 15% (drift 0%)
    }
    targets = {
        "NIFTY_EQUITIES": 0.40,
        "MUTUAL_FUNDS": 0.30,
        "MCX_GOLD": 0.15,
        "GOVT_10Y_GSEC": 0.15,
    }

    plan = calculate_portfolio_drift(holdings, targets, threshold_pct=5.0)
    assert plan.max_drift_pct == 1.0
    assert plan.rebalance_recommended is False


def test_action_types_and_trade_directions():
    """Verify overweight assets trigger SELL and underweight assets trigger BUY."""
    holdings = {
        "NIFTY_EQUITIES": 70000.0,  # 70% (Target 50% -> Overweight)
        "GOVT_10Y_GSEC": 30000.0,   # 30% (Target 50% -> Underweight)
    }
    targets = {
        "NIFTY_EQUITIES": 0.50,
        "GOVT_10Y_GSEC": 0.50,
    }

    plan = calculate_portfolio_drift(holdings, targets, threshold_pct=5.0)
    inst_map = {inst.asset_code: inst for inst in plan.instructions}

    assert inst_map["NIFTY_EQUITIES"].action == "SELL"
    assert inst_map["NIFTY_EQUITIES"].trade_amount == 20000.0

    assert inst_map["GOVT_10Y_GSEC"].action == "BUY"
    assert inst_map["GOVT_10Y_GSEC"].trade_amount == 20000.0

