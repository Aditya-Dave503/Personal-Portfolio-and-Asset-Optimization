from sqlalchemy.orm import Session

from models.goal import Goal
from schemas.goal import GoalCreate, GoalUpdate


def create_goal(db: Session, user_id: int, goal_data: GoalCreate) -> Goal:
    goal = Goal(
        user_id=user_id,
        name=goal_data.name,
        category=goal_data.category,
        target_amount=goal_data.target_amount,
        target_date=goal_data.target_date
    )

    db.add(goal)
    db.commit()
    db.refresh(goal)

    return goal


def get_goals(db: Session, user_id: int) -> list[Goal]:
    return (
        db.query(Goal)
        .filter(Goal.user_id == user_id)
        .order_by(Goal.target_date, Goal.id)
        .all()
    )


def get_goal(db: Session, user_id: int, goal_id: int) -> Goal | None:
    return (
        db.query(Goal)
        .filter(Goal.user_id == user_id, Goal.id == goal_id)
        .first()
    )


def update_goal(db: Session, goal: Goal, goal_data: GoalUpdate) -> Goal:
    for field, value in goal_data.model_dump(exclude_unset=True).items():
        setattr(goal, field, value)

    db.commit()
    db.refresh(goal)

    return goal


def delete_goal(db: Session, goal: Goal) -> None:
    db.delete(goal)
    db.commit()
