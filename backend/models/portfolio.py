
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column,relationship

from core.database import Base
class Portfolio(Base):
    __tablename__ = "portfolios"

    __table_args__ = (
        CheckConstraint(
            "investment_amount > 0",
            name="ck_portfolios_investment_amount_positive",
        ),
        CheckConstraint(
            "risk_score >= 1 AND risk_score <= 10",
            name="ck_portfolios_risk_score_range",
        ),
        Index("ix_portfolios_user_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    investment_amount: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), nullable=False
    )

    risk_score: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    optimization_mode: Mapped[str] = mapped_column(
        String(32), nullable=False
    )

    ml_status: Mapped[str] = mapped_column(
    String(32), nullable=False, default="pending"
)

    method: Mapped[str] = mapped_column(
        String(150), nullable=False
    )

    expected_annual_return_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    expected_annual_volatility_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    sharpe_ratio: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    value_at_risk_1m_95pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    value_at_risk_1m_99pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    conditional_var_1m_95pct: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4), nullable=True
    )

    var_99_constraint_status: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    user: Mapped["User"] = relationship(
        back_populates="portfolios"
    )

    allocations: Mapped[list["PortfolioAllocation"]] = relationship(
        back_populates="portfolio",
        cascade="all, delete-orphan",
    )
