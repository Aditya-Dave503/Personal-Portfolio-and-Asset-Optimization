
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user_id

from schemas.risk_profile import (
    RiskProfileCreate,
    RiskProfileResponse,
)

from services.risk_profile_service import (
    create_risk_profile,
    get_risk_profile,
    RiskProfileValidationError,
    RiskScoringError,
)

from questionnaire.risk_questionnaire import get_risk_questions


router = APIRouter(
    prefix="/risk-profiles",
    tags=["Risk Profiles"],
)


# --------------------------------------------------
# CREATE OR UPDATE RISK PROFILE
# --------------------------------------------------

@router.post(
    "/",
    response_model=RiskProfileResponse,
)
def submit_risk_assessment(
    profile_data: RiskProfileCreate,
    response: Response,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    try:
        profile, created = create_risk_profile(
            db,
            user_id,
            profile_data,
        )

        response.status_code = 201 if created else 200

        return profile

    except RiskProfileValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RiskScoringError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# --------------------------------------------------
# GET CURRENT RISK PROFILE
# --------------------------------------------------

@router.get(
    "/me",
    response_model=RiskProfileResponse,
)
def get_my_profile(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    profile = get_risk_profile(db, user_id)

    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="Risk profile not found",
        )

    return profile


# --------------------------------------------------
# GET BEHAVIOURAL QUESTIONNAIRE
# --------------------------------------------------

@router.get("/questions")
def get_questionnaire():
    return get_risk_questions()
