
import math

from sqlalchemy.orm import Session

from models.risk_profile import RiskProfile
from schemas.risk_profile import RiskProfileCreate
from ml.risk_profiler import get_calculated_risk_score


class RiskProfileValidationError(Exception):
    """Raised when risk profile inputs are invalid."""


class RiskScoringError(Exception):
    """Raised when ML risk scoring fails or returns an invalid score."""


def create_risk_profile(
    db: Session,
    user_id: int,
    profile_data: RiskProfileCreate
):
    # Validate user ID
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise RiskProfileValidationError("Invalid user ID")

    # Validate investment horizon
    horizon = profile_data.investment_horizon_years

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise RiskProfileValidationError(
            "Investment horizon must be a positive integer"
        )

    # Validate maximum loss amount
    max_loss = profile_data.max_loss_amount

    if (
        isinstance(max_loss, bool)
        or not isinstance(max_loss, (int, float))
        or not math.isfinite(max_loss)
        or max_loss <= 0
    ):
        raise RiskProfileValidationError(
            "Maximum loss amount must be a positive finite number"
        )

    # Calculate risk score using the ML module
    try:
        risk_score = get_calculated_risk_score(
            investment_horizon_years=horizon,
            max_loss_amount=max_loss
        )
    except Exception as exc:
        raise RiskScoringError("Unable to calculate risk score") from exc

    # Validate ML output
    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, int)
        or not 1 <= risk_score <= 10
    ):
        raise RiskScoringError(
            "ML model returned an invalid risk score"
        )

    # Save profile and score
    profile = RiskProfile(
        user_id=user_id,
        max_loss_amount=max_loss,
        investment_horizon_years=horizon,
        calculated_risk_score=risk_score
    )

    try:
        db.add(profile)
        db.commit()
        db.refresh(profile)
    except Exception:
        db.rollback()
        raise

    return profile


def get_risk_profile(db: Session, user_id: int):
    return (
        db.query(RiskProfile)
        .filter(RiskProfile.user_id == user_id)
        .first()
    )
