
from decimal import Decimal

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.database import Base
from models import User
from models.risk_profile import RiskProfile
from models.portfolio import Portfolio
from models.portfolio_allocation import PortfolioAllocation

import services.portfolio_service as portfolio_service


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
def test_user(db_session):
    user = User(
        email="portfolio@example.com",
        password_hash="test_hash",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    profile = RiskProfile(
        user_id=user.id,
        calculated_risk_score=7,
        max_loss_amount=10000,
        investment_horizon_years=10,
    )
    db_session.add(profile)
    db_session.commit()

    return user


def mock_ml_result(investment_amount=100000, risk_score=7):
    return {
        "investment_amount": investment_amount,
        "risk_score": risk_score,
        "risk_profile_category": "Moderate / Balanced",
        "method": "Test Optimization",
        "expected_annual_return_pct": 12.5,
        "expected_annual_volatility_pct": 8.5,
        "sharpe_ratio": 1.2,
        "value_at_risk_1m_95pct": 3.5,
        "value_at_risk_1m_99pct": 5.0,
        "conditional_var_1m_95pct": 4.0,
        "var_99_constraint_status": "SATISFIED",
        "allocations": [
            {
                "asset_code": "NIFTY_EQUITIES",
                "asset_name": "Nifty 50",
                "category": "Equities",
                "risk_level": "High",
                "weight_percent": 60.0,
                "allocated_amount": 60000.0,
                "ml_expected_annual_return_pct": 14.0,
            },
            {
                "asset_code": "MCX_GOLD",
                "asset_name": "MCX Gold",
                "category": "Precious Metals",
                "risk_level": "Moderate",
                "weight_percent": 40.0,
                "allocated_amount": 40000.0,
                "ml_expected_annual_return_pct": 8.0,
            },
        ],
    }


def test_create_optimized_portfolio(db_session, test_user, monkeypatch):
    monkeypatch.setattr(
        portfolio_service,
        "run_pipeline",
        lambda **kwargs: mock_ml_result(),
    )

    portfolio = portfolio_service.create_optimized_portfolio(
        db_session,
        test_user.id,
        100000,
        "black_litterman",
    )

    assert portfolio.user_id == test_user.id
    assert portfolio.investment_amount == Decimal("100000")
    assert portfolio.risk_score == 7
    assert portfolio.ml_status == "success"


def test_portfolio_allocations_are_saved(db_session, test_user, monkeypatch):
    monkeypatch.setattr(
        portfolio_service,
        "run_pipeline",
        lambda **kwargs: mock_ml_result(),
    )

    portfolio = portfolio_service.create_optimized_portfolio(
        db_session,
        test_user.id,
        100000,
    )

    allocations = (
        db_session.query(PortfolioAllocation)
        .filter(PortfolioAllocation.portfolio_id == portfolio.id)
        .all()
    )

    assert len(allocations) == 2
    assert sum(a.weight_percent for a in allocations) == Decimal("100")


def test_missing_risk_profile(db_session):
    user = User(
        email="no-profile@example.com",
        password_hash="test_hash",
    )
    db_session.add(user)
    db_session.commit()

    with pytest.raises(portfolio_service.RiskProfileNotFoundError):
        portfolio_service.create_optimized_portfolio(
            db_session,
            user.id,
            100000,
        )


@pytest.mark.parametrize("amount", [0, -100, "invalid", None])
def test_invalid_investment_amount(db_session, test_user, amount):
    with pytest.raises(portfolio_service.PortfolioValidationError):
        portfolio_service.create_optimized_portfolio(
            db_session,
            test_user.id,
            amount,
        )


def test_invalid_optimization_mode(db_session, test_user):
    with pytest.raises(portfolio_service.PortfolioValidationError):
        portfolio_service.create_optimized_portfolio(
            db_session,
            test_user.id,
            100000,
            "invalid_mode",
        )


def test_maximum_loss_cannot_exceed_investment(db_session, test_user):
    with pytest.raises(portfolio_service.PortfolioValidationError):
        portfolio_service.create_optimized_portfolio(
            db_session,
            test_user.id,
            5000,
        )


def test_ml_pipeline_failure(db_session, test_user, monkeypatch):
    def fail_pipeline(**kwargs):
        raise RuntimeError("ML failure")

    monkeypatch.setattr(
        portfolio_service,
        "run_pipeline",
        fail_pipeline,
    )

    with pytest.raises(portfolio_service.PortfolioOptimizationError):
        portfolio_service.create_optimized_portfolio(
            db_session,
            test_user.id,
            100000,
        )


def test_invalid_ml_output(db_session, test_user, monkeypatch):
    invalid_result = mock_ml_result()
    invalid_result["var_99_constraint_status"] = "VIOLATED (Needs More G-Sec)"

    monkeypatch.setattr(
        portfolio_service,
        "run_pipeline",
        lambda **kwargs: invalid_result,
    )

    with pytest.raises(portfolio_service.PortfolioOptimizationError):
        portfolio_service.create_optimized_portfolio(
            db_session,
            test_user.id,
            100000,
        )


def test_database_failure_rolls_back(db_session, test_user, monkeypatch):
    monkeypatch.setattr(
        portfolio_service,
        "run_pipeline",
        lambda **kwargs: mock_ml_result(),
    )

    def fail_commit():
        raise RuntimeError("Database failure")

    monkeypatch.setattr(db_session, "commit", fail_commit)

    with pytest.raises(RuntimeError):
        portfolio_service.create_optimized_portfolio(
            db_session,
            test_user.id,
            100000,
        )

    assert db_session.query(Portfolio).count() == 0
    assert db_session.query(PortfolioAllocation).count() == 0
