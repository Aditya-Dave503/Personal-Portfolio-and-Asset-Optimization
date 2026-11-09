
from sqlalchemy.orm import Session, selectinload

from models.portfolio import Portfolio


class PortfolioQueryValidationError(Exception):
    """Raised when portfolio query inputs are invalid."""


class PortfolioNotFoundError(Exception):
    """Raised when a portfolio is not found for the given user."""


def _validate_ids(user_id, portfolio_id=None):
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise PortfolioQueryValidationError("Invalid user ID")

    if portfolio_id is not None:
        if (
            isinstance(portfolio_id, bool)
            or not isinstance(portfolio_id, int)
            or portfolio_id <= 0
        ):
            raise PortfolioQueryValidationError("Invalid portfolio ID")


def get_portfolio_history(db: Session, user_id: int):
    """Return all portfolios belonging to a user, newest first."""

    _validate_ids(user_id)

    return (
        db.query(Portfolio)
        .filter(Portfolio.user_id == user_id)
        .order_by(Portfolio.created_at.desc(), Portfolio.id.desc())
        .all()
    )


def get_portfolio_detail(db: Session, user_id: int, portfolio_id: int):
    """Return a user's portfolio with its allocations."""

    _validate_ids(user_id, portfolio_id)

    portfolio = (
        db.query(Portfolio)
        .options(selectinload(Portfolio.allocations))
        .filter(
            Portfolio.id == portfolio_id,
            Portfolio.user_id == user_id,
        )
        .first()
    )

    if portfolio is None:
        raise PortfolioNotFoundError("Portfolio not found")

    return portfolio


from decimal import Decimal

from models.holding import Holding


def compare_portfolio_with_holdings(
    db: Session,
    user_id: int,
    portfolio_id: int,
):
    """
    Compare existing holdings with recommended portfolio allocations.

    Comparison is based on asset-wise monetary values.
    """

    portfolio = get_portfolio_detail(db, user_id, portfolio_id)

    holdings = (
        db.query(Holding)
        .filter(Holding.user_id == user_id)
        .all()
    )

    current_values = {}

    for holding in holdings:
        current_values[holding.asset_code] = (
            current_values.get(holding.asset_code, Decimal("0"))
            + holding.current_value
        )

    recommended_values = {
        allocation.asset_code: allocation.allocated_amount
        for allocation in portfolio.allocations
    }

    asset_codes = sorted(set(current_values) | set(recommended_values))

    comparison = []

    for asset_code in asset_codes:
        current_value = current_values.get(asset_code, Decimal("0"))
        recommended_value = recommended_values.get(
            asset_code, Decimal("0")
        )

        comparison.append({
            "asset_code": asset_code,
            "current_value": current_value,
            "recommended_value": recommended_value,
            "difference": recommended_value - current_value,
        })

    return comparison


def calculate_net_worth(db: Session, user_id: int):
    """
    Calculate net worth using the current valuation of recorded holdings.

    This is a holdings-based asset total, not a complete financial
    net-worth calculation including liabilities.
    """

    _validate_ids(user_id)

    holdings = (
        db.query(Holding)
        .filter(Holding.user_id == user_id)
        .all()
    )

    return sum(
        (holding.current_value for holding in holdings),
        Decimal("0"),
    )
