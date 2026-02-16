"""Pydantic schema models for the Pre-Delinquency Intervention Engine."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field, constr, conint, confloat


class TransactionType(str, Enum):
    """Allowed transaction directions."""

    CREDIT = "credit"
    DEBIT = "debit"


class RiskLevel(str, Enum):
    """Risk level buckets for customers."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Channel(str, Enum):
    """Outbound communication channels."""

    SMS = "sms"
    EMAIL = "email"
    APP = "app"


class Customer(BaseModel):
    """Customer profile used by the risk engine."""

    customer_id: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Unique identifier for the customer (e.g., 'C000123').",
        examples=["C000123"],
    )
    name: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Full customer name.",
        examples=["Rahul Sharma"],
    )
    age: conint(ge=18, le=90) = Field(
        ...,
        description="Customer age in years (18-90).",
        examples=[35],
    )
    credit_score: conint(ge=300, le=900) = Field(
        ...,
        description="Credit score as reported by bureau (300-900).",
        examples=[720],
    )
    account_balance: confloat(ge=0) = Field(
        ...,
        description="Current savings or primary account balance in INR.",
        examples=[45000.75],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C000123",
                "name": "Rahul Sharma",
                "age": 32,
                "credit_score": 715,
                "account_balance": 58250.0,
            }
        }
    }


class Transaction(BaseModel):
    """Single banking transaction associated with a customer."""

    transaction_id: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Unique identifier for the transaction (e.g., UUID).",
        examples=["tx_01HXYZ1234ABC"],
    )
    customer_id: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Customer identifier this transaction belongs to.",
        examples=["C000123"],
    )
    date: datetime = Field(
        ...,
        description="Timestamp when the transaction was posted.",
        examples=["2024-06-15T10:30:00Z"],
    )
    amount: confloat(gt=0) = Field(
        ...,
        description="Absolute transaction amount in INR.",
        examples=[2500.0],
    )
    transaction_type: TransactionType = Field(
        ...,
        description="Direction of transaction: credit or debit.",
        examples=["debit"],
    )
    category: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Category label (e.g., 'salary', 'utility_electricity', 'upi_lending_app').",
        examples=["utility_electricity"],
    )
    merchant: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Merchant or counterparty name, if available.",
        examples=["BESCOM"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "tx_01HXYZ1234ABC",
                "customer_id": "C000123",
                "date": "2024-06-15T10:30:00Z",
                "amount": 2499.0,
                "transaction_type": "debit",
                "category": "discretionary_restaurant",
                "merchant": "Swiggy",
            }
        }
    }


class RiskScore(BaseModel):
    """Risk score output for a customer."""

    customer_id: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Customer identifier for whom the risk is computed.",
        examples=["C000123"],
    )
    risk_score: confloat(ge=0, le=100) = Field(
        ...,
        description="Model-predicted risk score on a 0-100 scale.",
        examples=[78.5],
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Discrete risk bucket derived from risk_score.",
        examples=["high"],
    )
    prediction_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the prediction was generated.",
        examples=["2024-06-30T12:00:00Z"],
    )
    top_risk_factors: List[str] = Field(
        default_factory=list,
        description="Human-readable list of key risk drivers for explainability.",
        examples=[["salary delays", "savings depletion >40%", "high lending app usage"]],
    )
    recommended_action: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Recommended next-best action for this customer.",
        examples=["Proactive outreach with restructuring offer via app notification."],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C000123",
                "risk_score": 82.4,
                "risk_level": "critical",
                "prediction_date": "2024-06-30T12:00:00Z",
                "top_risk_factors": [
                    "Salary credited 8 days late in last 2 months",
                    "Savings balance dropped by 55% over 90 days",
                    "Spike in UPI lending app transactions",
                ],
                "recommended_action": (
                    "Trigger high-priority call center outreach and offer short-term "
                    "payment deferral with structured repayment plan."
                ),
            }
        }
    }


class InterventionRequest(BaseModel):
    """Request to trigger a customer-facing intervention."""

    customer_id: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Customer identifier to intervene on.",
        examples=["C000123"],
    )
    intervention_type: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Type of intervention (e.g., 'reminder', 'restructuring_offer', 'collection_call').",
        examples=["restructuring_offer"],
    )
    channel: Channel = Field(
        ...,
        description="Preferred communication channel.",
        examples=["app"],
    )
    message: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Localized, customer-ready message body.",
        examples=[
            (
                "Hi Rahul, we noticed changes in your repayment behavior. "
                "Tap here to review flexible repayment options and avoid late fees."
            )
        ],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id": "C000123",
                "intervention_type": "reminder",
                "channel": "sms",
                "message": "Your EMI is due in 5 days. Pay now to avoid late fees.",
            }
        }
    }

