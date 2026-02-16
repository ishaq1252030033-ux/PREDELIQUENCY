"""In-memory metrics for predictions, API response times, errors, and high-risk counts."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

# Cap for rolling windows to avoid unbounded growth
MAX_RESPONSE_TIMES = 10_000
MAX_PREDICTION_TIMES = 10_000
MAX_HIGH_RISK_HISTORY = 1_000


@dataclass
class HighRiskSnapshot:
    """Timestamp and count of high-risk customers at a point in time."""

    timestamp: float
    count: int


class MetricsStore:
    """Thread-safe store for API and prediction metrics."""

    _lock = threading.Lock()
    _total_predictions = 0
    _prediction_times_sum = 0.0
    _prediction_times_count = 0
    _prediction_times_recent: Deque[float] = deque(maxlen=MAX_PREDICTION_TIMES)
    _request_count = 0
    _error_count = 0
    _response_times: Deque[Tuple[str, str, float]] = deque(maxlen=MAX_RESPONSE_TIMES)  # (method, path, ms)
    _high_risk_history: List[HighRiskSnapshot] = []

    @classmethod
    def record_prediction(cls, duration_seconds: float, customer_id: Optional[str] = None) -> None:
        with cls._lock:
            cls._total_predictions += 1
            cls._prediction_times_sum += duration_seconds
            cls._prediction_times_count += 1
            cls._prediction_times_recent.append(duration_seconds)

    @classmethod
    def record_api_request(cls, method: str, path: str, duration_ms: float, status_code: int) -> None:
        with cls._lock:
            cls._request_count += 1
            if status_code >= 400:
                cls._error_count += 1
            cls._response_times.append((method, path, duration_ms))

    @classmethod
    def record_high_risk_count(cls, count: int) -> None:
        with cls._lock:
            cls._high_risk_history.append(HighRiskSnapshot(timestamp=time.time(), count=count))
            if len(cls._high_risk_history) > MAX_HIGH_RISK_HISTORY:
                cls._high_risk_history.pop(0)

    @classmethod
    def get_snapshot(cls) -> dict:
        with cls._lock:
            avg_prediction_time = (
                cls._prediction_times_sum / cls._prediction_times_count
                if cls._prediction_times_count else 0.0
            )
            error_rate = (
                cls._error_count / cls._request_count if cls._request_count else 0.0
            )
            # Average response time (last N)
            times = [t[2] for t in cls._response_times]
            avg_response_time_ms = sum(times) / len(times) if times else 0.0
            return {
                "total_predictions": cls._total_predictions,
                "average_prediction_time_seconds": round(avg_prediction_time, 4),
                "prediction_count": cls._prediction_times_count,
                "total_requests": cls._request_count,
                "total_errors": cls._error_count,
                "error_rate": round(error_rate, 4),
                "average_response_time_ms": round(avg_response_time_ms, 2),
                "response_times_sample_size": len(times),
                "high_risk_history": [
                    {"timestamp": s.timestamp, "count": s.count}
                    for s in cls._high_risk_history[-100:]
                ],
            }

    @classmethod
    def reset(cls) -> None:
        """Reset all metrics (e.g. for tests)."""
        with cls._lock:
            cls._total_predictions = 0
            cls._prediction_times_sum = 0.0
            cls._prediction_times_count = 0
            cls._prediction_times_recent.clear()
            cls._request_count = 0
            cls._error_count = 0
            cls._response_times.clear()
            cls._high_risk_history.clear()
