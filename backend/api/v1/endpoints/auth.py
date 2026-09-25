from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import (
    create_access_token,
    get_current_user_id,
    verify_password
)
from schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from schemas.user import UserResponse
from services.user_service import (
    create_user,
    get_user_by_email,
    get_user_by_id
)


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=201
)
def register(
    user_data: RegisterRequest,
    db: Session = Depends(get_db)
):
    existing_user = get_user_by_email(db, user_data.email)

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Email already registered"
        )

    return create_user(
        db,
        user_data
    )


@router.post(
    "/login",
    response_model=TokenResponse
)
def login(
    user_data: LoginRequest,
    db: Session = Depends(get_db)
):
    user = get_user_by_email(db, user_data.email)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        user_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    access_token = create_access_token(user.id)

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get(
    "/me",
    response_model=UserResponse
)
def get_me(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    user = get_user_by_id(db, user_id)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    return user
