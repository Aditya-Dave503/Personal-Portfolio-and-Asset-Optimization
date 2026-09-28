from sqlalchemy.orm import Session

from models.risk_profile import RiskProfile
from schemas.risk_profile import RiskProfileCreate


def create_risk_profile(
    db: Session,
    user_id: int,
    profile_data: RiskProfileCreate
):
    profile = RiskProfile(
        user_id=user_id,
        max_loss_amount=profile_data.max_loss_amount,
        investment_horizon_years=profile_data.investment_horizon_years
    )

    db.add(profile)
    db.commit()
    db.refresh(profile)

    return profile


def get_risk_profile(db: Session, user_id: int):
    return (
        db.query(RiskProfile)
        .filter(RiskProfile.user_id == user_id)
        .first()
    )