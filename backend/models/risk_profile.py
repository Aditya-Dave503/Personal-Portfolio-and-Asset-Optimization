
from datetime import datetime

from sqlalchemy import (
    Float,
    Integer,
    Numeric,
    ForeignKey,
    String,
    DateTime,
    UniqueConstraint,
    func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class RiskProfile(Base):
    __tablename__ = "risk_profiles"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True
    )

    # User relationship
    # Removed unique=True to support assessment history
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # Financial inputs
    age: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    annual_income: Mapped[float | None] = mapped_column(
        Numeric(14, 2),
        nullable=True
    )

    annual_savings: Mapped[float | None] = mapped_column(
        Numeric(14, 2),
        nullable=True
    )

    portfolio_value: Mapped[float | None] = mapped_column(
        Numeric(14, 2),
        nullable=True
    )

    max_loss_amount: Mapped[float | None] = mapped_column(
        Numeric(14, 2),
        nullable=True
    )

    investment_horizon_years: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # Derived financial features
    savings_to_income_ratio: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    max_loss_tolerance_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Calculated scores
    objective_risk_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    behavioral_risk_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    combined_risk_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Preserve the existing optimizer-compatible score
    calculated_risk_score: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    risk_category: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True
    )

    # Version tracking
    model_version: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="1.0"
    )

    questionnaire_version: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="1.0"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # One assessment has multiple answers
    answers: Mapped[list["RiskProfileAnswer"]] = relationship(
        back_populates="risk_profile",
        cascade="all, delete-orphan"
    )


class RiskProfileAnswer(Base):
    __tablename__ = "risk_profile_answers"

    __table_args__ = (
        UniqueConstraint(
            "risk_profile_id",
            "question_id",
            name="uq_risk_profile_question"
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True
    )

    risk_profile_id: Mapped[int] = mapped_column(
        ForeignKey("risk_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    question_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False
    )

    selected_option: Mapped[str] = mapped_column(
        String(1),
        nullable=False
    )

    behavioral_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    # Relationship back to assessment
    risk_profile: Mapped["RiskProfile"] = relationship(
        back_populates="answers"
    )
