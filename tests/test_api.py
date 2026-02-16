"""API integration tests for FastAPI endpoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.database import (
    Customer,
    Intervention,
    RiskAssessment,
    Transaction,
    SessionLocal,
    init_db,
)
from backend.app.models.schemas import RiskScore


client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures: database setup / teardown
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function", autouse=True)
def db_setup_teardown():
    """
    Ensure database tables exist and are clean for each API test.

    The current API implementation mostly reads from CSV/ML artifacts,
    but we keep the database ready for future extensions and to satisfy
    the requirement for setup/teardown.
    """
    init_db()
    db = SessionLocal()
    try:
        # Truncate tables for a clean slate
        db.query(Intervention).delete()
        db.query(RiskAssessment).delete()
        db.query(Transaction).delete()
        db.query(Customer).delete()
        db.commit()
        yield
    finally:
        db.close()


@pytest.fixture(scope="session")
def sample_customer_id() -> str:
    """
    Return a valid customer_id from the engineered features, if available.
    """
    features_path = Path("ml/data/processed/features.csv")
    if not features_path.exists():
        pytest.skip("features.csv not found; run feature engineering pipeline first.")
    df = pd.read_csv(features_path)
    if "customer_id" not in df.columns or df.empty:
        pytest.skip("features.csv has no customer_id column or is empty.")
    return str(df["customer_id"].iloc[0])


# ---------------------------------------------------------------------------
# Tests for /api/v1/predict
# ---------------------------------------------------------------------------


def test_predict_valid_customer(sample_customer_id: str):
    """Valid customer_id should return 200 and a valid RiskScore payload."""
    payload = {"customer_id": sample_customer_id}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Validate against RiskScore schema
    score = RiskScore(**data)
    assert score.customer_id == sample_customer_id
    assert 0.0 <= score.risk_score <= 100.0
    assert score.risk_level in {"low", "medium", "high", "critical"}


def test_predict_invalid_customer():
    """Invalid customer_id should return 404."""
    payload = {"customer_id": "NON_EXISTENT_CUSTOMER"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests for /api/v1/predict/batch
# ---------------------------------------------------------------------------


def test_predict_batch(sample_customer_id: str):
    """Batch prediction should return a list of RiskScore-like objects."""
    payload = {"customer_ids": [sample_customer_id, sample_customer_id]}
    resp = client.post("/api/v1/predict/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 2

    scores: List[RiskScore] = [RiskScore(**item) for item in data]
    for s in scores:
        assert s.customer_id == sample_customer_id
        assert 0.0 <= s.risk_score <= 100.0


# ---------------------------------------------------------------------------
# Tests for /api/v1/high-risk-customers
# ---------------------------------------------------------------------------


def test_high_risk_customers_threshold_and_sorting():
    """Returned customers should all be above threshold and sorted by risk_score desc."""
    threshold = 80.0
    resp = client.get("/api/v1/high-risk-customers", params={"threshold": threshold, "limit": 50})
    assert resp.status_code == 200
    data = resp.json()

    if not data:
        pytest.skip("No high-risk customers above threshold; adjust test threshold or generate data.")

    scores = [RiskScore(**item) for item in data]

    # All scores should be >= threshold
    assert all(s.risk_score >= threshold for s in scores)

    # Sorted descending by risk_score
    scores_values = [s.risk_score for s in scores]
    assert scores_values == sorted(scores_values, reverse=True)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


def test_root_health_endpoint():
    """Root /health endpoint should respond with healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert "version" in data


def test_api_health_endpoint():
    """/api/v1/health should reflect API and model health."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    # model_loaded is a bool flag
    assert isinstance(data.get("model_loaded"), bool)

