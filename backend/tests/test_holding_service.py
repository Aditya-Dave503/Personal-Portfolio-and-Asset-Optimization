
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.database import Base
from models import User, Holding

from services.holding_service import (
    create_holding,
    get_holdings,
    get_holding,
    update_holding,
    delete_holding,
    is_optimizer_supported_asset,
    HoldingValidationError,
    HoldingNotFoundError,
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


def create_test_user(db_session, email):
    user = User(
        email=email,
        password_hash="test_hash",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def create_test_holding(db_session, user_id):
    return create_holding(
        db_session,
        user_id,
        asset_code="MCX_GOLD",
        asset_class="Precious Metals",
        instrument_name="Gold",
        quantity=10,
        purchase_value=50000,
        current_value=55000,
    )


def test_create_holding(db_session):
    user = create_test_user(db_session, "user1@example.com")

    holding = create_test_holding(db_session, user.id)

    assert holding.user_id == user.id
    assert holding.asset_code == "MCX_GOLD"
    assert holding.quantity == Decimal("10")


def test_get_all_user_holdings(db_session):
    user1 = create_test_user(db_session, "user1@example.com")
    user2 = create_test_user(db_session, "user2@example.com")

    create_test_holding(db_session, user1.id)
    create_test_holding(db_session, user2.id)

    holdings = get_holdings(db_session, user1.id)

    assert len(holdings) == 1
    assert holdings[0].user_id == user1.id


def test_get_specific_holding(db_session):
    user = create_test_user(db_session, "user1@example.com")
    holding = create_test_holding(db_session, user.id)

    result = get_holding(db_session, user.id, holding.id)

    assert result is not None
    assert result.id == holding.id


def test_user_cannot_access_another_users_holding(db_session):
    user1 = create_test_user(db_session, "user1@example.com")
    user2 = create_test_user(db_session, "user2@example.com")

    holding = create_test_holding(db_session, user1.id)

    result = get_holding(db_session, user2.id, holding.id)

    assert result is None


def test_update_holding(db_session):
    user = create_test_user(db_session, "user1@example.com")
    holding = create_test_holding(db_session, user.id)

    updated = update_holding(
        db_session,
        user.id,
        holding.id,
        {"current_value": 60000},
    )

    assert updated.current_value == Decimal("60000")


def test_delete_holding(db_session):
    user = create_test_user(db_session, "user1@example.com")
    holding = create_test_holding(db_session, user.id)

    delete_holding(db_session, user.id, holding.id)

    assert get_holding(db_session, user.id, holding.id) is None


def test_user_cannot_update_another_users_holding(db_session):
    user1 = create_test_user(db_session, "user1@example.com")
    user2 = create_test_user(db_session, "user2@example.com")

    holding = create_test_holding(db_session, user1.id)

    with pytest.raises(HoldingNotFoundError):
        update_holding(
            db_session,
            user2.id,
            holding.id,
            {"current_value": 60000},
        )


def test_user_cannot_delete_another_users_holding(db_session):
    user1 = create_test_user(db_session, "user1@example.com")
    user2 = create_test_user(db_session, "user2@example.com")

    holding = create_test_holding(db_session, user1.id)

    with pytest.raises(HoldingNotFoundError):
        delete_holding(db_session, user2.id, holding.id)


def test_invalid_quantity_rejected(db_session):
    user = create_test_user(db_session, "user1@example.com")

    with pytest.raises(HoldingValidationError):
        create_holding(
            db_session,
            user.id,
            asset_code="MCX_GOLD",
            asset_class="Precious Metals",
            instrument_name="Gold",
            quantity=-10,
            purchase_value=50000,
            current_value=55000,
        )


def test_optimizer_asset_classification():
    assert is_optimizer_supported_asset("MCX_GOLD")
    assert is_optimizer_supported_asset("NIFTY_EQUITIES")
    assert not is_optimizer_supported_asset("SLV")
