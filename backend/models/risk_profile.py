from sqlalchemy import Float, Integer, Numeric, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class RiskProfile(Base):
    __tablename__ = "risk_profiles"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        unique=True,
        nullable=False,
        index=True
    )

    calculated_risk_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    max_loss_amount: Mapped[float | None] = mapped_column(
        Numeric(12, 2),
        nullable=True
    )

    investment_horizon_years: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )