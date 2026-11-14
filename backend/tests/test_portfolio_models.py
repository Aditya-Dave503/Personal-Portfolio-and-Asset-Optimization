from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.database import Base
from models import User, Portfolio, PortfolioAllocation, Holding


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


def create_user(db_session):
    user = User(
        email="test@example.com",
        password_hash="test_hash",
    )
    db_session.add(user)
    db_session.commit()
    return user


def test_create_portfolio_and_allocation(db_session):
    user = create_user(db_session)

    portfolio = Portfolio(
        user_id=user.id,
        investment_amount=Decimal("100000.00"),
        risk_score=7,
        optimization_mode="black_litterman",
        ml_status="succeeded",
        method="Black-Litterman",
    )

    allocation = PortfolioAllocation(
        asset_code="NIFTY_EQUITIES",
        asset_name="Nifty 50 Large-Cap Equities",
        category="Equities",
        risk_level="High Risk",
        weight_percent=Decimal("60.0000"),
        allocated_amount=Decimal("60000.00"),
    )

    portfolio.allocations.append(allocation)
    db_session.add(portfolio)
    db_session.commit()

    assert portfolio.user.email == "test@example.com"
    assert len(portfolio.allocations) == 1
    assert portfolio.allocations[0].asset_code == "NIFTY_EQUITIES"


def test_create_holding(db_session):
    user = create_user(db_session)

    holding = Holding(
        user_id=user.id,
        asset_code="MCX_GOLD",
        asset_class="Precious Metals",
        instrument_name="Gold",
        quantity=Decimal("10.000000"),
        purchase_value=Decimal("50000.00"),
        current_value=Decimal("55000.00"),
    )

    db_session.add(holding)
    db_session.commit()

    assert holding.user.email == "test@example.com"
    assert holding.current_value == Decimal("55000.00")


def test_portfolio_allocation_cascade_delete(db_session):
    user = create_user(db_session)

    portfolio = Portfolio(
        user_id=user.id,
        investment_amount=Decimal("100000.00"),
        risk_score=7,
        optimization_mode="max_sharpe",
        ml_status="succeeded",
        method="Maximum Sharpe",
    )

    portfolio.allocations.append(
        PortfolioAllocation(
            asset_code="MCX_GOLD",
            asset_name="Gold",
            category="Precious Metals",
            risk_level="Moderate Risk",
            weight_percent=Decimal("100.0000"),
            allocated_amount=Decimal("100000.00"),
        )
    )

    db_session.add(portfolio)
    db_session.commit()

    portfolio_id = portfolio.id

    db_session.delete(portfolio)
    db_session.commit()

    assert db_session.query(PortfolioAllocation).filter_by(
        portfolio_id=portfolio_id
    ).count() == 0


def test_duplicate_asset_in_same_portfolio_rejected(db_session):
    user = create_user(db_session)

    portfolio = Portfolio(
        user_id=user.id,
        investment_amount=Decimal("100000.00"),
        risk_score=7,
        optimization_mode="min_variance",
        ml_status="succeeded",
        method="Minimum Variance",
    )

    db_session.add(portfolio)
    db_session.commit()

    allocation1 = PortfolioAllocation(
        portfolio_id=portfolio.id,
        asset_code="MCX_GOLD",
        asset_name="Gold",
        category="Precious Metals",
        risk_level="Moderate Risk",
        weight_percent=Decimal("50.0000"),
        allocated_amount=Decimal("50000.00"),
    )

    allocation2 = PortfolioAllocation(
        portfolio_id=portfolio.id,
        asset_code="MCX_GOLD",
        asset_name="Gold",
        category="Precious Metals",
        risk_level="Moderate Risk",
        weight_percent=Decimal("50.0000"),
        allocated_amount=Decimal("50000.00"),
    )

    db_session.add_all([allocation1, allocation2])

    with pytest.raises(IntegrityError):
        db_session.commit()

    db_session.rollback()
