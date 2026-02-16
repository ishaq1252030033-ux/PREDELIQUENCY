"""SQLAlchemy models and database session management for the engine."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Generator, List, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
    Session,
)


# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pre_delinquency.db")


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""


def _create_engine(url: str = DATABASE_URL):
    connect_args = {}
    if url.startswith("sqlite"):
        # Needed for SQLite with FastAPI in multi-threaded environment
        connect_args = {"check_same_thread": False}
    return create_engine(url, echo=False, future=True, connect_args=connect_args)


engine = _create_engine(DATABASE_URL)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------


class Customer(Base):
    """Customer master data."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    credit_score: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    income_bracket: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    transactions: Mapped[List["Transaction"]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    risk_assessments: Mapped[List["RiskAssessment"]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    interventions: Mapped[List["Intervention"]] = relationship(
        back_populates="customer",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Transaction(Base):
    """Individual financial transaction."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    transaction_id: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )

    # Use customer_id as an FK to the customers table
    customer_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    merchant: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    customer: Mapped["Customer"] = relationship(back_populates="transactions")

    __table_args__ = (
        Index("ix_transactions_customer_date", "customer_id", "date"),
    )


class RiskAssessment(Base):
    """Model prediction and risk assessment for a customer at a point in time."""

    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    customer_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    risk_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    prediction_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    top_risk_factors: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, doc="JSON list of top risk factors for this prediction."
    )
    model_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    customer: Mapped["Customer"] = relationship(back_populates="risk_assessments")

    __table_args__ = (
        Index(
            "ix_risk_assessments_customer_pred_date",
            "customer_id",
            "prediction_date",
        ),
    )


class Intervention(Base):
    """Triggered intervention for a customer."""

    __tablename__ = "interventions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)

    customer_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("customers.customer_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    intervention_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    channel: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    message: Mapped[str] = mapped_column(String(1024), nullable=False)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        index=True,
        doc="e.g., pending, sent, failed, acknowledged",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    customer: Mapped["Customer"] = relationship(back_populates="interventions")

    __table_args__ = (
        Index(
            "ix_interventions_customer_created",
            "customer_id",
            "created_at",
        ),
    )


# ---------------------------------------------------------------------------
# Utility: create tables
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create all tables based on the current models."""
    Base.metadata.create_all(bind=engine)


