
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.database import Base
from models import User
from models.portfolio import Portfolio
from models.portfolio_allocation import PortfolioAllocation
from models.holding import Holding

from services.portfolio_comparison_service import (
    get_portfolio_history,
    get_portfolio_detail,
    compare_portfolio_with_holdings,
    calculate_net_worth,
    PortfolioQueryValidationError,
    PortfolioNotFoundError,
)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)

    TestingSession = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    session = TestingSession()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def test_users(db_session):
    user1 = User(
        email="user1@example.com",
        password_hash="test_hash",
    )
    user2 = User(
        email="user2@example.com",
        password_hash="test_hash",
    )

    db_session.add_all([user1, user2])
    db_session.commit()

    return user1, user2


def create_portfolio(db_session, user_id, investment_amount=100000):
    portfolio = Portfolio(
        user_id=user_id,
        investment_amount=Decimal(str(investment_amount)),
        risk_score=7,
        optimization_mode="black_litterman",
        ml_status="success",
        method="Test Optimization",
    )

    db_session.add(portfolio)
    db_session.commit()
    db_session.refresh(portfolio)

    return portfolio


def create_allocation(db_session, portfolio_id, asset_code, amount):
    allocation = PortfolioAllocation(
        portfolio_id=portfolio_id,
        asset_code=asset_code,
        asset_name=asset_code,
        category="Test Category",
        risk_level="Moderate",
        weight_percent=Decimal("100"),
        allocated_amount=Decimal(str(amount)),
    )

    db_session.add(allocation)
    db_session.commit()

    return allocation


def create_holding(db_session, user_id, asset_code, current_value):
    holding = Holding(
        user_id=user_id,
        asset_code=asset_code,
        asset_class="Test Class",
        instrument_name=asset_code,
        quantity=Decimal("1"),
        purchase_value=Decimal("1000"),
        current_value=Decimal(str(current_value)),
    )

    db_session.add(holding)
    db_session.commit()

    return holding


def test_portfolio_history_is_owner_scoped(db_session, test_users):
    user1, user2 = test_users

    p1 = create_portfolio(db_session, user1.id)
    create_portfolio(db_session, user2.id)

    history = get_portfolio_history(db_session, user1.id)

    assert len(history) == 1
    assert history[0].id == p1.id
    assert all(p.user_id == user1.id for p in history)


def test_portfolio_history_is_newest_first(db_session, test_users):
    user1, _ = test_users

    first = create_portfolio(db_session, user1.id)
    second = create_portfolio(db_session, user1.id)

    history = get_portfolio_history(db_session, user1.id)

    assert history[0].id == second.id
    assert history[1].id == first.id


def test_portfolio_detail_includes_allocations(db_session, test_users):
    user1, _ = test_users

    portfolio = create_portfolio(db_session, user1.id)
    create_allocation(
        db_session, portfolio.id, "NIFTY_EQUITIES", 100000
    )

    result = get_portfolio_detail(
        db_session, user1.id, portfolio.id
    )

    assert result.id == portfolio.id
    assert len(result.allocations) == 1


def test_user_cannot_access_another_users_portfolio(
    db_session, test_users
):
    user1, user2 = test_users

    portfolio = create_portfolio(db_session, user1.id)

    with pytest.raises(PortfolioNotFoundError):
        get_portfolio_detail(
            db_session, user2.id, portfolio.id
        )


def test_missing_portfolio_raises_error(db_session, test_users):
    user1, _ = test_users

    with pytest.raises(PortfolioNotFoundError):
        get_portfolio_detail(db_session, user1.id, 99999)


def test_holdings_comparison(db_session, test_users):
    user1, user2 = test_users

    portfolio = create_portfolio(db_session, user1.id)

    create_allocation(
        db_session, portfolio.id, "NIFTY_EQUITIES", 60000
    )
    create_allocation(
        db_session, portfolio.id, "MCX_GOLD", 40000
    )

    create_holding(
        db_session, user1.id, "NIFTY_EQUITIES", 50000
    )
    create_holding(
        db_session, user1.id, "OTHER_ASSET", 10000
    )

    # Another user's holding must not be included.
    create_holding(
        db_session, user2.id, "MCX_GOLD", 90000
    )

    result = compare_portfolio_with_holdings(
        db_session, user1.id, portfolio.id
    )

    comparison = {item["asset_code"]: item for item in result}

    assert comparison["NIFTY_EQUITIES"]["difference"] == Decimal("10000")
    assert comparison["MCX_GOLD"]["difference"] == Decimal("40000")
    assert comparison["OTHER_ASSET"]["difference"] == Decimal("-10000")


def test_net_worth_uses_only_users_holdings(db_session, test_users):
    user1, user2 = test_users

    create_holding(db_session, user1.id, "NIFTY_EQUITIES", 50000)
    create_holding(db_session, user1.id, "MCX_GOLD", 25000)
    create_holding(db_session, user2.id, "OTHER_ASSET", 90000)

    result = calculate_net_worth(db_session, user1.id)

    assert result == Decimal("75000")


@pytest.mark.parametrize(
    "user_id",
    [0, -1, None, True, "1"],
)
def test_invalid_user_id(db_session, user_id):
    with pytest.raises(PortfolioQueryValidationError):
        get_portfolio_history(db_session, user_id)
