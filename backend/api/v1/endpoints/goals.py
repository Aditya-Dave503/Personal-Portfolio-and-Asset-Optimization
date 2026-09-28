from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user_id
from schemas.goal import GoalCreate, GoalResponse, GoalUpdate
from services.goal_service import (
    create_goal,
    delete_goal,
    get_goal,
    get_goals,
    update_goal
)


router = APIRouter(
    prefix="/goals",
    tags=["Goals"]
)


@router.post(
    "/",
    response_model=GoalResponse,
    status_code=status.HTTP_201_CREATED
)
def create_user_goal(
    goal_data: GoalCreate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    return create_goal(db, user_id, goal_data)


@router.get(
    "/",
    response_model=list[GoalResponse]
)
def list_user_goals(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    return get_goals(db, user_id)


@router.get(
    "/{goal_id}",
    response_model=GoalResponse
)
def get_user_goal(
    goal_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    goal = get_goal(db, user_id, goal_id)

    if goal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )

    return goal


@router.patch(
    "/{goal_id}",
    response_model=GoalResponse
)
def update_user_goal(
    goal_id: int,
    goal_data: GoalUpdate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    goal = get_goal(db, user_id, goal_id)

    if goal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )

    return update_goal(db, goal, goal_data)


@router.delete(
    "/{goal_id}",
    status_code=status.HTTP_204_NO_CONTENT
)
def delete_user_goal(
    goal_id: int,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    goal = get_goal(db, user_id, goal_id)

    if goal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )

    delete_goal(db, goal)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
