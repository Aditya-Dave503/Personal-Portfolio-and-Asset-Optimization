
from unittest.mock import MagicMock

import pytest

import services.risk_profile_service as service
from schemas.risk_profile import RiskProfileCreate


def get_profile_data():
    return RiskProfileCreate(
        max_loss_amount=50000,
        investment_horizon_years=10
    )


def test_create_risk_profile_success(monkeypatch):
    db = MagicMock()

    monkeypatch.setattr(
        service,
        "get_calculated_risk_score",
        lambda investment_horizon_years, max_loss_amount: 7
    )

    profile = service.create_risk_profile(db, 1, get_profile_data())

    assert profile.user_id == 1
    assert profile.max_loss_amount == 50000
    assert profile.investment_horizon_years == 10
    assert profile.calculated_risk_score == 7

    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.refresh.assert_called_once_with(profile)


def test_invalid_user_id():
    db = MagicMock()

    with pytest.raises(service.RiskProfileValidationError):
        service.create_risk_profile(db, 0, get_profile_data())


def test_ml_scoring_failure(monkeypatch):
    db = MagicMock()

    def failing_ml(**kwargs):
        raise RuntimeError("ML model failed")

    monkeypatch.setattr(
        service,
        "get_calculated_risk_score",
        failing_ml
    )

    with pytest.raises(service.RiskScoringError):
        service.create_risk_profile(db, 1, get_profile_data())

    db.add.assert_not_called()


def test_invalid_ml_score(monkeypatch):
    db = MagicMock()

    monkeypatch.setattr(
        service,
        "get_calculated_risk_score",
        lambda **kwargs: 15
    )

    with pytest.raises(service.RiskScoringError):
        service.create_risk_profile(db, 1, get_profile_data())

    db.add.assert_not_called()


def test_database_failure_rolls_back(monkeypatch):
    db = MagicMock()

    monkeypatch.setattr(
        service,
        "get_calculated_risk_score",
        lambda **kwargs: 7
    )

    db.commit.side_effect = RuntimeError("Database failure")

    with pytest.raises(RuntimeError):
        service.create_risk_profile(db, 1, get_profile_data())

    db.rollback.assert_called_once()
