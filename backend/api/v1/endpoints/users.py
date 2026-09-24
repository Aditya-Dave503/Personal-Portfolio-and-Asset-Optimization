from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from schemas.user import UserCreate, UserResponse
from services.user_service import (
    create_user,
    get_user_by_email,
    get_user_by_id
)

router = APIRouter(
    prefix="/users",
    tags=["Users"]
)


@router.post("/", response_model=UserResponse, status_code=201)
def create_new_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    existing_user = get_user_by_email(db, user_data.email)

    if existing_user:
        raise HTTPException(
            status_code=409,
            detail="Email already registered"
        )

    return create_user(db, user_data)


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    user = get_user_by_id(db, user_id)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return user