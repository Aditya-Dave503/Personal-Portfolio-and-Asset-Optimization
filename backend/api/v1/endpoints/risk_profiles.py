
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
    get_risk_profile,
    get_risk_profile_history,
    RiskProfileValidationError,
    RiskScoringError,
)

from questionnaire.risk_questionnaire import get_risk_questions


router = APIRouter(
    prefix="/risk-profiles",
    tags=["Risk Profiles"]
)


# --------------------------------------------------
# CREATE NEW RISK ASSESSMENT
# --------------------------------------------------

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
    try:
        return create_risk_profile(
            db,
            user_id,
            profile_data
        )

    except RiskProfileValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        )

    except RiskScoringError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# --------------------------------------------------
# GET LATEST RISK ASSESSMENT
# --------------------------------------------------

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


# --------------------------------------------------
# GET RISK ASSESSMENT HISTORY
# --------------------------------------------------

@router.get(
    "/history",
    response_model=list[RiskProfileResponse]
)
def get_my_profile_history(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    profiles = get_risk_profile_history(db, user_id)

    return profiles


# --------------------------------------------------
# GET BEHAVIOURAL QUESTIONNAIRE
# --------------------------------------------------

@router.get(
    "/questions"
)
def get_questionnaire():
    return get_risk_questions()
