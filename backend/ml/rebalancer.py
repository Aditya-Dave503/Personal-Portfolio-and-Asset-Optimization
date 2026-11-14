"""
Portfolio Drift Detection & Smart Rebalancing Engine

Monitors asset allocation drift against target policy boundaries.
Generates self-financing rebalancing trade orders with exact rupee amounts.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np


@dataclass
class TradeInstruction:
    asset_code: str
    action: str  # 'BUY' | 'SELL' | 'HOLD'
    trade_amount: float
    current_weight_pct: float
    target_weight_pct: float
    drift_pct: float

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


@dataclass
class RebalancePlan:
    total_portfolio_value: float
    rebalance_recommended: bool
    rebalance_threshold_pct: float
    max_drift_pct: float
    total_absolute_drift_pct: float
    instructions: List[TradeInstruction]
    net_cash_impact: float

    def __getitem__(self, item: str) -> Any:
        if item == "total_portfolio_value_inr":
            return self.total_portfolio_value
        if item == "rebalance_triggered":
            return self.rebalance_recommended
        if item == "total_drift_pct":
            return self.total_absolute_drift_pct
        return getattr(self, item)


def calculate_portfolio_drift(
    current_holdings: Dict[str, float],
    target_weights: Dict[str, float],
    threshold_pct: float = 5.0,
) -> RebalancePlan:
    """
    Computes allocation drift and generates self-financing rebalancing trades.

    Parameters:
        current_holdings: Map of {asset_code: current_rupee_value}
        target_weights: Map of {asset_code: target_weight_fraction (0.0 to 1.0) or percentage (0 to 100)}
        threshold_pct: Maximum allowable percentage point drift before triggering rebalance (default 5.0%)
    """
    raw_weights = {}
    sum_w = sum(target_weights.values())
    if sum_w <= 0:
        raise ValueError("Sum of target weights must be strictly positive.")

    scale = 1.0 / sum_w if sum_w > 1.5 else 1.0
    for k, v in target_weights.items():
        raw_weights[k] = (v * scale) if sum_w > 1.5 else (v / sum_w)

    total_value = float(sum(current_holdings.values()))
    if total_value <= 0:
        raise ValueError("Total portfolio value must be strictly positive.")

    all_assets = sorted(set(list(current_holdings.keys()) + list(raw_weights.keys())))

    instructions = []
    max_drift = 0.0
    total_drift = 0.0

    for asset in all_assets:
        val = float(current_holdings.get(asset, 0.0))
        cur_w = val / total_value
        tgt_w = float(raw_weights.get(asset, 0.0))

        # Drift in percentage points
        drift_pp = (cur_w - tgt_w) * 100.0
        abs_drift = abs(drift_pp)
        max_drift = max(max_drift, abs_drift)
        total_drift += abs_drift

        # Required trade in rupees: Target - Current
        trade_amount = round((tgt_w - cur_w) * total_value, 2)

        if trade_amount > 1.0:
            action = "BUY"
        elif trade_amount < -1.0:
            action = "SELL"
        else:
            action = "HOLD"

        instructions.append(TradeInstruction(
            asset_code=asset,
            action=action,
            trade_amount=abs(trade_amount),
            current_weight_pct=round(cur_w * 100.0, 2),
            target_weight_pct=round(tgt_w * 100.0, 2),
            drift_pct=round(drift_pp, 2),
        ))

    rebalance_needed = bool(max_drift >= threshold_pct)

    # Invariant: Net cash impact must sum to 0.0 within numerical rounding
    net_impact = round(sum(
        inst.trade_amount if inst.action == "BUY" else (-inst.trade_amount if inst.action == "SELL" else 0.0)
        for inst in instructions
    ), 2)

    return RebalancePlan(
        total_portfolio_value=round(total_value, 2),
        rebalance_recommended=rebalance_needed,
        rebalance_threshold_pct=threshold_pct,
        max_drift_pct=round(max_drift, 2),
        total_absolute_drift_pct=round(total_drift, 2),
        instructions=instructions,
        net_cash_impact=net_impact,
    )


def generate_rebalancing_orders(
    current_holdings: Dict[str, float],
    target_weights: Dict[str, float],
    drift_threshold_pct: float = 5.0,
    min_trade_amount_inr: float = 100.0
) -> Dict[str, Any]:
    """Generate executable trade instructions if cumulative drift exceeds threshold."""
    plan = calculate_portfolio_drift(current_holdings, target_weights, threshold_pct=drift_threshold_pct)
    orders = []
    instructions = []
    for inst in plan.instructions:
        if inst.trade_amount >= min_trade_amount_inr and inst.action in ("BUY", "SELL"):
            orders.append({
                "asset_code": inst.asset_code,
                "action": inst.action,
                "amount_inr": inst.trade_amount,
                "drift_pct": inst.drift_pct,
            })
            instructions.append(f"{inst.action} INR {inst.trade_amount:,.2f} of {inst.asset_code}")

    return {
        "rebalance_triggered": plan.rebalance_recommended,
        "threshold_pct": drift_threshold_pct,
        "observed_drift_pct": plan.total_absolute_drift_pct,
        "total_portfolio_value_inr": plan.total_portfolio_value,
        "orders_count": len(orders),
        "orders": orders,
        "trade_instructions": instructions,
        "status": "REBALANCE_REQUIRED" if plan.rebalance_recommended else "PORTFOLIO_WITHIN_TOLERANCE",
    }
