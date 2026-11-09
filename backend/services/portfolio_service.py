
import math
from decimal import Decimal, InvalidOperation

from sqlalchemy.orm import Session

from models.risk_profile import RiskProfile
from models.portfolio import Portfolio
from models.portfolio_allocation import PortfolioAllocation

from ml.pipeline import run_pipeline


class PortfolioValidationError(Exception):
    """Raised when portfolio inputs or ML output are invalid."""


class RiskProfileNotFoundError(Exception):
    """Raised when the user has no saved risk profile."""


class PortfolioOptimizationError(Exception):
    """Raised when portfolio optimization fails."""


SUPPORTED_MODES = {
    "black_litterman",
    "max_sharpe",
    "min_variance",
}

SUPPORTED_ASSETS = {
    "NIFTY_EQUITIES",
    "MUTUAL_FUNDS",
    "MCX_GOLD",
    "GOVT_10Y_GSEC",
}


def _to_decimal(value, field_name):
    if isinstance(value, bool):
        raise PortfolioValidationError(f"Invalid {field_name}")

    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise PortfolioValidationError(f"Invalid {field_name}")

    if not number.is_finite():
        raise PortfolioValidationError(f"Invalid {field_name}")

    return number


def _validate_ml_result(result, investment_amount, risk_score):
    if not isinstance(result, dict):
        raise PortfolioOptimizationError("ML returned an invalid result")

    required_fields = {
        "investment_amount",
        "risk_score",
        "method",
        "expected_annual_return_pct",
        "expected_annual_volatility_pct",
        "sharpe_ratio",
        "value_at_risk_1m_95pct",
        "value_at_risk_1m_99pct",
        "conditional_var_1m_95pct",
        "var_99_constraint_status",
        "allocations",
    }

    if not required_fields.issubset(result):
        raise PortfolioOptimizationError("ML result is missing required fields")

    if _to_decimal(result["investment_amount"], "ML investment amount") != investment_amount:
        raise PortfolioOptimizationError("ML returned a different investment amount")

    if result["risk_score"] != risk_score:
        raise PortfolioOptimizationError("ML returned a different risk score")

    if not isinstance(result["method"], str) or not result["method"].strip():
        raise PortfolioOptimizationError("ML returned an invalid optimization method")

    numeric_fields = [
        "expected_annual_return_pct",
        "expected_annual_volatility_pct",
        "sharpe_ratio",
        "value_at_risk_1m_95pct",
        "value_at_risk_1m_99pct",
        "conditional_var_1m_95pct",
    ]

    for field in numeric_fields:
        value = _to_decimal(result[field], field)
        if field != "sharpe_ratio" and value < 0 and field != "expected_annual_return_pct":
            raise PortfolioOptimizationError(f"ML returned an invalid {field}")

    status = result["var_99_constraint_status"]
    if status not in {"SATISFIED", "VIOLATED (Needs More G-Sec)"}:
        raise PortfolioOptimizationError("ML returned an invalid VaR status")

    if status != "SATISFIED":
        raise PortfolioOptimizationError("ML could not satisfy the VaR constraint")

    allocations = result["allocations"]

    if not isinstance(allocations, list) or not allocations:
        raise PortfolioOptimizationError("ML returned no allocations")

    required_allocation_fields = {
        "asset_code",
        "asset_name",
        "category",
        "risk_level",
        "weight_percent",
        "allocated_amount",
        "ml_expected_annual_return_pct",
    }

    total_weight = Decimal("0")
    total_amount = Decimal("0")
    seen_assets = set()

    for allocation in allocations:
        if not isinstance(allocation, dict):
            raise PortfolioOptimizationError("Invalid allocation entry")

        if not required_allocation_fields.issubset(allocation):
            raise PortfolioOptimizationError("Allocation fields are missing")

        asset_code = allocation["asset_code"]

        if asset_code not in SUPPORTED_ASSETS or asset_code in seen_assets:
            raise PortfolioOptimizationError("Unsupported or duplicate asset")

        seen_assets.add(asset_code)

        for field in ("asset_name", "category", "risk_level"):
            if not isinstance(allocation[field], str) or not allocation[field].strip():
                raise PortfolioOptimizationError(f"Invalid allocation {field}")

        weight = _to_decimal(allocation["weight_percent"], "allocation weight")
        amount = _to_decimal(allocation["allocated_amount"], "allocation amount")
        expected_return = _to_decimal(
            allocation["ml_expected_annual_return_pct"],
            "allocation expected return",
        )

        if not Decimal("0") <= weight <= Decimal("100"):
            raise PortfolioOptimizationError("Allocation weight is outside 0–100")

        if amount < 0:
            raise PortfolioOptimizationError("Allocation amount cannot be negative")

        if amount != (investment_amount * weight / Decimal("100")).quantize(
            Decimal("0.01")
        ):
            raise PortfolioOptimizationError("Allocation amount does not match its weight")

        if not expected_return.is_finite():
            raise PortfolioOptimizationError("Invalid allocation expected return")

        total_weight += weight
        total_amount += amount

    if abs(total_weight - Decimal("100")) > Decimal("0.05"):
        raise PortfolioOptimizationError("Allocation weights do not total 100%")

    if abs(total_amount - investment_amount) > Decimal("0.05"):
        raise PortfolioOptimizationError("Allocation amounts do not match investment")

    return allocations


def create_optimized_portfolio(
    db: Session,
    user_id: int,
    investment_amount,
    optimization_mode: str = "black_litterman",
) -> Portfolio:
    """Generate and persist an optimized portfolio for a user."""

    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise PortfolioValidationError("Invalid user ID")

    amount = _to_decimal(investment_amount, "investment amount")

    if amount <= 0:
        raise PortfolioValidationError("Investment amount must be positive")

    if amount > Decimal("999999999999.99"):
        raise PortfolioValidationError("Investment amount exceeds supported limit")

    if not isinstance(optimization_mode, str):
        raise PortfolioValidationError("Invalid optimization mode")

    optimization_mode = optimization_mode.strip().lower()

    if optimization_mode not in SUPPORTED_MODES:
        raise PortfolioValidationError("Unsupported optimization mode")

    risk_profile = (
        db.query(RiskProfile)
        .filter(RiskProfile.user_id == user_id)
        .first()
    )

    if risk_profile is None:
        raise RiskProfileNotFoundError("Risk profile not found")

    risk_score = risk_profile.calculated_risk_score

    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, (int, float))
        or not math.isfinite(risk_score)
        or not float(risk_score).is_integer()
        or not 1 <= risk_score <= 10
    ):
        raise PortfolioValidationError("Saved risk score is invalid")

    risk_score = int(risk_score)

    max_loss = _to_decimal(
        risk_profile.max_loss_amount,
        "maximum loss amount",
    )

    if max_loss <= 0 or max_loss > amount:
        raise PortfolioValidationError(
            "Maximum loss must be positive and cannot exceed investment amount"
        )

    # The ML optimizer expects a fraction, not an INR amount.
    max_loss_tolerance_pct = float(max_loss / amount)

    try:
        result = run_pipeline(
            investment_amount=float(amount),
            explicit_risk_score=risk_score,
            optimization_mode=optimization_mode,
            investment_horizon_years=risk_profile.investment_horizon_years,
            max_loss_tolerance_pct=max_loss_tolerance_pct,
        )
    except Exception as exc:
        raise PortfolioOptimizationError(
            "Portfolio optimization failed"
        ) from exc

    allocations = _validate_ml_result(result, amount, risk_score)

    portfolio = Portfolio(
        user_id=user_id,
        investment_amount=amount,
        risk_score=risk_score,
        optimization_mode=optimization_mode,
        ml_status="success",
        method=result["method"],
        expected_annual_return_pct=_to_decimal(
            result["expected_annual_return_pct"], "expected return"
        ),
        expected_annual_volatility_pct=_to_decimal(
            result["expected_annual_volatility_pct"], "volatility"
        ),
        sharpe_ratio=_to_decimal(result["sharpe_ratio"], "Sharpe ratio"),
        value_at_risk_1m_95pct=_to_decimal(
            result["value_at_risk_1m_95pct"], "VaR 95"
        ),
        value_at_risk_1m_99pct=_to_decimal(
            result["value_at_risk_1m_99pct"], "VaR 99"
        ),
        conditional_var_1m_95pct=_to_decimal(
            result["conditional_var_1m_95pct"], "CVaR 95"
        ),
        var_99_constraint_status=result["var_99_constraint_status"],
    )

    try:
        db.add(portfolio)
        db.flush()

        for allocation in allocations:
            db.add(
                PortfolioAllocation(
                    portfolio_id=portfolio.id,
                    asset_code=allocation["asset_code"],
                    asset_name=allocation["asset_name"],
                    category=allocation["category"],
                    risk_level=allocation["risk_level"],
                    weight_percent=_to_decimal(
                        allocation["weight_percent"], "allocation weight"
                    ),
                    allocated_amount=_to_decimal(
                        allocation["allocated_amount"], "allocation amount"
                    ),
                    ml_expected_annual_return_pct=_to_decimal(
                        allocation["ml_expected_annual_return_pct"],
                        "allocation expected return",
                    ),
                )
            )

        db.commit()
        db.refresh(portfolio)

    except Exception:
        db.rollback()
        raise

    return portfolio
