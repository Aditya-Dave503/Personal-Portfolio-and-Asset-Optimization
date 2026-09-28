from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user_id
from schemas.risk_profile import (
    RiskProfileCreate,
    RiskProfileResponse
)
from services.risk_profile_service import (
    create_risk_profile,
    get_risk_profile
)


router = APIRouter(
    prefix="/risk-profiles",
    tags=["Risk Profiles"]
)


@router.post(
    "/",
    response_model=RiskProfileResponse,
    status_code=201
)
def create_profile(
    profile_data: RiskProfileCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    existing_profile = get_risk_profile(db, user_id)

    if existing_profile:
        raise HTTPException(
            status_code=409,
            detail="Risk profile already exists"
        )

    return create_risk_profile(
        db,
        user_id,
        profile_data
    )


@router.get(
    "/me",
    response_model=RiskProfileResponse
)
def get_my_profile(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    profile = get_risk_profile(db, user_id)

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Risk profile not found"
        )

    return profile