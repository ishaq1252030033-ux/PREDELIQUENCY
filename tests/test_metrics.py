"""Tests for the in-memory metrics store."""

from __future__ import annotations

import pytest

from backend.app.utils.metrics import MetricsStore


@pytest.fixture(autouse=True)
def _reset_metrics() -> None:
    """Reset metrics before each test."""
    MetricsStore.reset()


def test_initial_snapshot_is_zeroed() -> None:
    snap = MetricsStore.get_snapshot()
    assert snap["total_predictions"] == 0
    assert snap["total_requests"] == 0
    assert snap["total_errors"] == 0
    assert snap["error_rate"] == 0.0


def test_record_prediction() -> None:
    MetricsStore.record_prediction(0.5)
    MetricsStore.record_prediction(1.5)
    snap = MetricsStore.get_snapshot()
    assert snap["total_predictions"] == 2
    assert snap["prediction_count"] == 2
    assert snap["average_prediction_time_seconds"] == pytest.approx(1.0, abs=0.01)


def test_record_api_request_success() -> None:
    MetricsStore.record_api_request("GET", "/health", 10.0, 200)
    snap = MetricsStore.get_snapshot()
    assert snap["total_requests"] == 1
    assert snap["total_errors"] == 0
    assert snap["average_response_time_ms"] == pytest.approx(10.0, abs=0.01)


def test_record_api_request_error() -> None:
    MetricsStore.record_api_request("POST", "/predict", 50.0, 500)
    snap = MetricsStore.get_snapshot()
    assert snap["total_errors"] == 1
    assert snap["error_rate"] == pytest.approx(1.0, abs=0.01)


def test_error_rate_mixed() -> None:
    for _ in range(8):
        MetricsStore.record_api_request("GET", "/health", 5.0, 200)
    for _ in range(2):
        MetricsStore.record_api_request("POST", "/predict", 5.0, 422)
    snap = MetricsStore.get_snapshot()
    assert snap["total_requests"] == 10
    assert snap["error_rate"] == pytest.approx(0.2, abs=0.01)


def test_high_risk_history() -> None:
    MetricsStore.record_high_risk_count(5)
    MetricsStore.record_high_risk_count(10)
    snap = MetricsStore.get_snapshot()
    history = snap["high_risk_history"]
    assert len(history) == 2
    assert history[0]["count"] == 5
    assert history[1]["count"] == 10


def test_reset() -> None:
    MetricsStore.record_prediction(1.0)
    MetricsStore.record_api_request("GET", "/", 1.0, 200)
    MetricsStore.record_high_risk_count(3)
    MetricsStore.reset()
    snap = MetricsStore.get_snapshot()
    assert snap["total_predictions"] == 0
    assert snap["total_requests"] == 0
    assert snap["high_risk_history"] == []
