"""API routes for the Pre-Delinquency Intervention Engine.

All endpoints live under ``/api/v1`` and are mounted via the
:pydata:`router` APIRouter exported from this module.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field, field_validator

from backend.app.models.database import SessionLocal, Transaction as TransactionORM
from backend.app.models.schemas import (
    Channel,
    InterventionRequest,
    RiskLevel,
    RiskScore,
    Transaction,
    TransactionType,
)
from backend.app.services.message_generator import (
    ABVariant,
    Language,
    MessageContext,
    MessageGenerator,
    Scenario,
)
from backend.app.services.risk_predictor import RiskPredictor
from backend.app.utils.logger import get_logger, get_predictions_logger
from backend.app.utils.metrics import MetricsStore
from ml.explainability import ExplainabilityService

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MAX_BATCH_SIZE: int = 500
"""Maximum customer IDs allowed in a single batch-predict request."""

_CUSTOMER_ID_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
"""Regex enforcing safe customer-ID format."""

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

logger = get_logger(__name__)
predictions_log = get_predictions_logger()


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Track active WebSocket connections and broadcast messages to all.

    Attributes:
        active: List of currently connected WebSocket clients.
    """

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket client.

        Args:
            websocket: The incoming WebSocket connection.
        """
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a disconnected WebSocket client.

        Args:
            websocket: The WebSocket that has disconnected.
        """
        if websocket in self.active:
            self.active.remove(websocket)

    async def broadcast_json(self, message: dict[str, Any]) -> None:
        """Broadcast a JSON payload to every active WebSocket.

        Broken connections are silently removed.

        Args:
            message: Serializable dict to send.
        """
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)


ws_manager = ConnectionManager()

router = APIRouter(prefix="/api/v1", tags=["risk"])

_explain_service: Optional[ExplainabilityService] = None


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------


def get_explain_service(predictor: RiskPredictor) -> ExplainabilityService:
    """Lazily create and return the singleton :class:`ExplainabilityService`.

    Args:
        predictor: The active ``RiskPredictor`` instance.

    Returns:
        The shared ``ExplainabilityService``.
    """
    global _explain_service  # noqa: PLW0603
    if _explain_service is None:
        _explain_service = ExplainabilityService(predictor)
    return _explain_service


def _validate_customer_id(customer_id: str) -> str:
    """FastAPI dependency that rejects unsafe customer-ID formats.

    Args:
        customer_id: Raw path parameter value.

    Returns:
        The validated customer ID.

    Raises:
        HTTPException: 422 if the format is invalid.
    """
    if not _CUSTOMER_ID_RE.match(customer_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid customer_id format: {customer_id!r}",
        )
    return customer_id


@lru_cache(maxsize=1)
def get_risk_predictor() -> RiskPredictor:
    """Return a singleton :class:`RiskPredictor`.

    Returns:
        The cached ``RiskPredictor`` instance.

    Raises:
        HTTPException: 500 if the predictor fails to initialise.
    """
    try:
        return RiskPredictor()
    except Exception as exc:
        logger.exception("Failed to initialise RiskPredictor: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk prediction service is unavailable.",
        ) from exc


@lru_cache(maxsize=1)
def get_transactions_df() -> pd.DataFrame:
    """Load the synthetic transactions CSV used by demo endpoints.

    In production this would be replaced by a database or data-lake query.

    Returns:
        A :class:`~pandas.DataFrame` of transactions (may be empty).
    """
    path = _PROJECT_ROOT / "ml" / "data" / "transactions.csv"
    if not path.exists():
        logger.warning("transactions.csv not found at %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["transaction_date"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Request body for single-customer risk prediction."""

    customer_id: str = Field(
        ...,
        description="Unique customer identifier to score.",
        examples=["C000123"],
        min_length=1,
        max_length=64,
    )

    @field_validator("customer_id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        """Ensure the customer ID matches the safe-character regex."""
        if not _CUSTOMER_ID_RE.match(v):
            raise ValueError(
                "customer_id must be 1-64 alphanumeric/dash/underscore characters"
            )
        return v


class BatchPredictRequest(BaseModel):
    """Request body for batch prediction (up to 500 items)."""

    customer_ids: List[str] = Field(
        ...,
        description="List of customer identifiers to score.",
        examples=[["C000001", "C000002", "C000003"]],
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )


class GenerateMessageRequest(BaseModel):
    """Request body for ``/generate-message`` and ``/generate-message/preview``."""

    customer_id: str = Field(..., description="Customer to generate message for.")
    customer_name: str = Field(..., description="Customer display name.")
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: str = Field(..., description="low | medium | high | critical")
    top_risk_factors: List[str] = Field(default_factory=list)
    channel: str = Field(default="app", description="sms | email | app")
    language: str = Field(default="en", description="en | hi")
    salary_delay_days: Optional[int] = None
    savings_drop_pct: Optional[float] = None
    lending_app_count: Optional[int] = None
    lending_app_amount: Optional[float] = None
    failed_payment_count: Optional[int] = None
    upcoming_emi_amount: Optional[float] = None
    upcoming_emi_date: Optional[str] = None
    variant: Optional[str] = Field(
        default=None, description="Force A or B variant for A/B testing."
    )
    scenario: Optional[str] = Field(
        default=None,
        description=(
            "Force scenario: salary_delay | savings_depletion | "
            "lending_app_spike | payment_failure | general_risk"
        ),
    )


class StreamingTransaction(BaseModel):
    """Incoming transaction event for the streaming simulator."""

    customer_id: str = Field(
        ..., min_length=1, max_length=64, description="Customer ID"
    )
    date: datetime
    amount: float = Field(..., gt=0)
    transaction_type: TransactionType
    category: str = Field(..., min_length=1)
    merchant: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Endpoints — Prediction
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=RiskScore,
    status_code=status.HTTP_200_OK,
    summary="Predict risk for a single customer",
    response_description="Risk score and explanation for the requested customer.",
)
def predict_risk(
    payload: PredictRequest = Body(...),
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> RiskScore:
    """Compute the pre-delinquency risk score for **one** customer.

    Args:
        payload: Validated request containing the ``customer_id``.
        predictor: Injected risk-prediction service.

    Returns:
        A :class:`RiskScore` with score, level, factors and action.

    Raises:
        HTTPException: 404 if the customer has no feature data;
            500 on unexpected failures.
    """
    try:
        t0 = time.perf_counter()
        result = predictor.predict_risk(payload.customer_id)
        duration = time.perf_counter() - t0
        MetricsStore.record_prediction(duration, customer_id=payload.customer_id)
        predictions_log.info(
            "predict customer_id=%s score=%.2f level=%s (%.3fs)",
            payload.customer_id,
            result.risk_score,
            result.risk_level.value,
            duration,
        )
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /predict: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to compute risk score.",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=List[RiskScore],
    status_code=status.HTTP_200_OK,
    summary="Batch risk prediction",
    response_description="Risk scores for the requested customers.",
)
def predict_risk_batch(
    payload: BatchPredictRequest = Body(...),
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> List[RiskScore]:
    """Compute risk scores for **multiple** customers in one call.

    Args:
        payload: Validated request containing up to 500 ``customer_ids``.
        predictor: Injected risk-prediction service.

    Returns:
        List of :class:`RiskScore` objects.

    Raises:
        HTTPException: 404 if none of the customers have feature data;
            500 on unexpected failures.
    """
    try:
        t0 = time.perf_counter()
        scores = predictor.batch_predict(payload.customer_ids)
        duration = time.perf_counter() - t0
        MetricsStore.record_prediction(duration)
        predictions_log.info("predict_batch count=%d (%.3fs)", len(scores), duration)
        if not scores:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No features found for any of the provided customer_ids.",
            )
        return scores
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /predict/batch: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to compute batch risk scores.",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints — Customer data
# ---------------------------------------------------------------------------


@router.get(
    "/customers/{customer_id}/transactions",
    response_model=List[Transaction],
    status_code=status.HTTP_200_OK,
    summary="Get recent transactions for a customer",
    response_description="List of recent transactions for the customer.",
)
def get_customer_transactions(
    customer_id: str = Depends(_validate_customer_id),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Maximum number of most-recent transactions to return.",
    ),
    tx_df: pd.DataFrame = Depends(get_transactions_df),
) -> List[Transaction]:
    """Return the most recent transactions for a customer.

    The current implementation reads from the synthetic
    ``ml/data/transactions.csv`` file.

    Args:
        customer_id: Validated customer identifier.
        limit: Maximum rows to return (1-500).
        tx_df: Injected transactions DataFrame.

    Returns:
        Sorted list of :class:`Transaction` objects.

    Raises:
        HTTPException: 503 if transaction data is unavailable;
            404 if no rows match the customer.
    """
    if tx_df.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction data is not available.",
        )

    subset = (
        tx_df[tx_df["customer_id"] == customer_id]
        .sort_values("transaction_date", ascending=False)
        .head(limit)
    )

    if subset.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transactions found for customer_id={customer_id}",
        )

    results: List[Transaction] = []
    for _, row in subset.iterrows():
        results.append(
            Transaction(
                transaction_id=(
                    f"{row.get('customer_id')}-"
                    f"{row.get('transaction_date').isoformat()}-"
                    f"{row.get('category')}"
                ),
                customer_id=row["customer_id"],
                date=row["transaction_date"],
                amount=float(row["amount_inr"]),
                transaction_type=row["type"],
                category=row["category"],
                merchant=row.get("description") or row.get("category"),
            )
        )
    return results


@router.get(
    "/high-risk-customers",
    response_model=List[RiskScore],
    status_code=status.HTTP_200_OK,
    summary="List customers above a risk threshold",
    response_description="Risk scores for customers above the given risk threshold.",
)
def get_high_risk_customers(
    threshold: float = Query(
        80.0,
        ge=0.0,
        le=100.0,
        description="Minimum risk score (0-100) to include a customer.",
    ),
    limit: int = Query(
        100,
        ge=1,
        le=100_000,
        description="Maximum number of customers to return.",
    ),
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> List[RiskScore]:
    """Return customers whose predicted risk exceeds *threshold*.

    Args:
        threshold: Minimum inclusive risk score.
        limit: Cap on returned results.
        predictor: Injected risk-prediction service.

    Returns:
        Descending-sorted list of :class:`RiskScore` objects.
    """
    customer_ids: list[str] = predictor.features_df["customer_id"].tolist()
    scores = predictor.batch_predict(customer_ids)

    filtered = sorted(
        (s for s in scores if s.risk_score >= threshold),
        key=lambda s: s.risk_score,
        reverse=True,
    )
    result = filtered[:limit]
    MetricsStore.record_high_risk_count(len(result))
    return result


# ---------------------------------------------------------------------------
# Endpoints — Intervention
# ---------------------------------------------------------------------------


@router.post(
    "/intervention/trigger",
    status_code=status.HTTP_200_OK,
    summary="Trigger an intervention workflow",
    response_description="Status of the triggered intervention.",
)
def trigger_intervention(
    request: InterventionRequest = Body(...),
) -> dict[str, str]:
    """Trigger an intervention for the specified customer.

    In a production system this would enqueue a job or call downstream
    services (SMS gateway, email service, call-centre CRM, etc.).

    Args:
        request: Validated intervention parameters.

    Returns:
        Dict with ``status`` and human-readable ``message``.
    """
    logger.info(
        "Intervention requested: customer_id=%s type=%s channel=%s",
        request.customer_id,
        request.intervention_type,
        request.channel,
    )
    return {
        "status": "success",
        "message": (
            f"Intervention '{request.intervention_type}' triggered via "
            f"{request.channel} for customer {request.customer_id}."
        ),
    }


# ---------------------------------------------------------------------------
# Endpoints — Health & metrics
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="API and model health check",
    response_description="Health status including whether the model is loaded.",
)
def api_health(
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> dict[str, Any]:
    """Health-check that also reports model-load status.

    Args:
        predictor: Injected risk-prediction service.

    Returns:
        Dict with ``status`` and ``model_loaded`` fields.
    """
    return {"status": "healthy", "model_loaded": predictor is not None}


@router.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Observability metrics",
    response_description="Prediction counts, response times, error rate, high-risk history.",
)
def get_metrics() -> dict[str, Any]:
    """Return the current metrics snapshot.

    Returns:
        Dict with prediction counts, response times, errors and high-risk counts.
    """
    return MetricsStore.get_snapshot()


# ---------------------------------------------------------------------------
# Endpoints — Explainability
# ---------------------------------------------------------------------------


@router.get(
    "/explain/{customer_id}",
    status_code=status.HTTP_200_OK,
    summary="Explain prediction for a customer",
    response_description="SHAP values and plots for the customer's latest features.",
)
def explain_customer(
    customer_id: str = Depends(_validate_customer_id),
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> dict[str, Any]:
    """Return SHAP explanation (values + base-64 plots) for *customer_id*.

    Args:
        customer_id: Validated customer identifier.
        predictor: Injected risk-prediction service.

    Returns:
        Dict with ``feature_contributions``, ``force_plot_base64``,
        and ``waterfall_plot_base64`` keys.

    Raises:
        HTTPException: 404 if the customer is unknown; 500 on errors.
    """
    service = get_explain_service(predictor)
    try:
        return service.explain_customer(customer_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /explain/%s: %s", customer_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to generate explanation.",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints — Message generation
# ---------------------------------------------------------------------------

_msg_generator = MessageGenerator()


def _build_message_context(req: GenerateMessageRequest) -> MessageContext:
    """Convert an incoming request into the internal :class:`MessageContext`.

    Args:
        req: The validated generate-message request.

    Returns:
        A fully populated ``MessageContext``.
    """
    return MessageContext(
        customer_id=req.customer_id,
        customer_name=req.customer_name,
        risk_score=req.risk_score,
        risk_level=RiskLevel(req.risk_level),
        top_risk_factors=req.top_risk_factors,
        channel=Channel(req.channel),
        language=Language(req.language),
        salary_delay_days=req.salary_delay_days,
        savings_drop_pct=req.savings_drop_pct,
        lending_app_count=req.lending_app_count,
        lending_app_amount=req.lending_app_amount,
        failed_payment_count=req.failed_payment_count,
        upcoming_emi_amount=req.upcoming_emi_amount,
        upcoming_emi_date=req.upcoming_emi_date,
    )


@router.post(
    "/generate-message",
    status_code=status.HTTP_200_OK,
    summary="Generate personalized intervention message",
    response_description="Rendered message with scenario, variant, and language metadata.",
)
def generate_message(req: GenerateMessageRequest) -> dict[str, Any]:
    """Generate a personalised intervention message for a customer.

    Args:
        req: Validated message-generation parameters.

    Returns:
        Dict with ``customer_id``, ``scenario``, ``channel``, ``language``,
        ``variant``, ``subject``, ``body``, and ``generated_at``.
    """
    ctx = _build_message_context(req)
    variant_override = ABVariant(req.variant) if req.variant in ("A", "B") else None
    scenario_override = Scenario(req.scenario) if req.scenario else None

    msg = _msg_generator.generate(
        ctx,
        variant_override=variant_override,
        scenario_override=scenario_override,
    )
    return {
        "customer_id": msg.customer_id,
        "scenario": msg.scenario.value,
        "channel": msg.channel.value,
        "language": msg.language.value,
        "variant": msg.variant.value,
        "subject": msg.subject,
        "body": msg.body,
        "generated_at": msg.generated_at,
    }


@router.post(
    "/generate-message/preview",
    status_code=status.HTTP_200_OK,
    summary="Preview both A/B message variants",
    response_description="Both A and B variants for comparison.",
)
def preview_message_variants(req: GenerateMessageRequest) -> dict[str, Any]:
    """Return both A and B message variants for side-by-side comparison.

    Args:
        req: Validated message-generation parameters.

    Returns:
        Dict keyed by ``"A"`` and ``"B"`` with scenario, variant,
        subject, and body for each.
    """
    ctx = _build_message_context(req)
    scenario_override = Scenario(req.scenario) if req.scenario else None
    variants = _msg_generator.generate_all_variants(
        ctx, scenario_override=scenario_override
    )
    return {
        k: {
            "scenario": v.scenario.value,
            "variant": v.variant.value,
            "subject": v.subject,
            "body": v.body,
        }
        for k, v in variants.items()
    }


# ---------------------------------------------------------------------------
# Endpoints — Streaming & WebSocket
# ---------------------------------------------------------------------------


@router.post(
    "/stream/transaction",
    status_code=status.HTTP_200_OK,
    summary="Ingest a single transaction event and recalculate risk",
    response_description="Echoed transaction and latest risk score for the customer.",
)
async def ingest_streaming_transaction(
    tx: StreamingTransaction,
    predictor: RiskPredictor = Depends(get_risk_predictor),
) -> dict[str, Any]:
    """Ingest a streaming transaction, persist it, recalculate risk, and broadcast.

    Args:
        tx: The incoming transaction payload.
        predictor: Injected risk-prediction service.

    Returns:
        Dict with ``transaction`` echo and updated ``risk`` score.
    """
    session = SessionLocal()
    try:
        tx_id = f"tx_{uuid.uuid4().hex[:16]}"
        db_tx = TransactionORM(
            transaction_id=tx_id,
            customer_id=tx.customer_id,
            date=tx.date,
            amount=tx.amount,
            transaction_type=tx.transaction_type.value,
            category=tx.category,
            merchant=tx.merchant,
        )
        session.add(db_tx)
        session.commit()
    except Exception as db_exc:
        session.rollback()
        logger.exception("Failed to persist streaming transaction: %s", db_exc)
    finally:
        session.close()

    t0 = time.perf_counter()
    risk = predictor.predict_risk(tx.customer_id)
    duration = time.perf_counter() - t0
    MetricsStore.record_prediction(duration, customer_id=tx.customer_id)
    predictions_log.info(
        "stream_predict customer_id=%s score=%.2f level=%s (%.3fs)",
        risk.customer_id,
        risk.risk_score,
        risk.risk_level.value,
        duration,
    )

    event: dict[str, Any] = {
        "type": "prediction_update",
        "customer_id": risk.customer_id,
        "risk": risk.model_dump(),
    }
    await ws_manager.broadcast_json(event)

    return {"transaction": tx.model_dump(), "risk": risk.model_dump()}


@router.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time prediction / stream updates.

    The server pushes events; client messages are ignored.

    Args:
        websocket: The incoming WebSocket connection.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        ws_manager.disconnect(websocket)
