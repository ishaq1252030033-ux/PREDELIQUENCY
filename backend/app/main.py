"""FastAPI application entry point for the Pre-Delinquency Intervention Engine.

Run in development with either::

    # From project root
    uvicorn backend.app.main:app --reload

    # From backend/ directory
    uvicorn app.main:app --reload
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``backend.*`` and ``ml.*``
# imports resolve regardless of the working directory uvicorn starts from.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = str(_Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes import router as api_router
from backend.app.services.risk_predictor import RiskPredictor
from backend.app.utils.logger import get_api_requests_logger, get_logger, setup_logging
from backend.app.utils.metrics import MetricsStore
from backend.app.utils.validate_config import validate_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_TITLE: str = "Pre-Delinquency Intervention Engine API"
API_VERSION: str = "1.0.0"
API_DESCRIPTION: str = "Early detection and intervention for at-risk customers"

# ---------------------------------------------------------------------------
# Logging bootstrap (runs at import time so every module has a logger)
# ---------------------------------------------------------------------------

setup_logging()
logger = get_logger("backend.app.main")
api_requests_log = get_api_requests_logger()

# ---------------------------------------------------------------------------
# CORS — configurable via CORS_ORIGINS env var (comma-separated)
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:8501,http://localhost:3000"
    ).split(",")
    if o.strip()
]

# ---------------------------------------------------------------------------
# Rate limiter (in-memory, per-IP sliding window)
# ---------------------------------------------------------------------------

_RATE_LIMIT_MAX: int = int(os.environ.get("RATE_LIMIT_MAX", "120"))
_RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
_rate_store: dict[str, list[float]] = {}


def _check_rate_limit(client_ip: str) -> bool:
    """Check whether *client_ip* is within the sliding-window rate limit.

    Args:
        client_ip: The IP address of the incoming request.

    Returns:
        ``True`` if the request is allowed, ``False`` if rate-limited.
    """
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW
    hits = [t for t in _rate_store.get(client_ip, []) if t > window_start]
    hits.append(now)
    _rate_store[client_ip] = hits
    return len(hits) <= _RATE_LIMIT_MAX


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: validate config and warm-load the ML model."""
    validate_config()

    model_path = Path(_PROJECT_ROOT) / "ml" / "models" / "risk_model.pkl"
    if not model_path.exists():
        logger.error("ML model not found at %s", model_path)
    else:
        logger.info("ML model found at %s", model_path)

    try:
        predictor = RiskPredictor(model_path=model_path)
        logger.info(
            "RiskPredictor ready — %d customers, %d features",
            len(predictor.features_df),
            len(predictor.feature_cols),
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("RiskPredictor init failed: %s", exc)

    yield  # application runs here


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: logging, rate-limiting, response timing
# ---------------------------------------------------------------------------


@app.middleware("http")
async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log every request, enforce rate limits, and track response-time metrics.

    Args:
        request: The incoming HTTP request.
        call_next: ASGI call chain.

    Returns:
        The HTTP response with an ``X-Process-Time-ms`` header appended.
    """
    client_ip: str = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip):
        MetricsStore.record_api_request(request.method, request.url.path, 0.0, 429)
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception during request processing")
        elapsed = (time.perf_counter() - start) * 1000
        MetricsStore.record_api_request(request.method, request.url.path, elapsed, 500)
        raise

    elapsed = (time.perf_counter() - start) * 1000
    MetricsStore.record_api_request(
        request.method, request.url.path, elapsed, response.status_code
    )
    api_requests_log.info(
        "%s %s — %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(api_router)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return structured JSON for HTTP exceptions.

    Args:
        request: The incoming request that caused the error.
        exc: The raised ``HTTPException``.

    Returns:
        A ``JSONResponse`` with the appropriate status code and detail.
    """
    logger.warning(
        "HTTPException %s on %s %s: %s",
        exc.status_code,
        request.method,
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return structured JSON for Pydantic / request-validation errors.

    Args:
        request: The incoming request.
        exc: The validation error containing field-level details.

    Returns:
        A 422 ``JSONResponse`` listing all validation issues.
    """
    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all that prevents stack-trace leakage to the client.

    Args:
        request: The incoming request.
        exc: The unhandled exception.

    Returns:
        A generic 500 ``JSONResponse``.
    """
    logger.exception(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


@app.get("/health", tags=["internal"])
async def health_check() -> dict[str, str]:
    """Return basic health status for load-balancer probes.

    Returns:
        JSON with ``status`` and ``version`` fields.
    """
    return {"status": "healthy", "version": API_VERSION}


@app.get("/", tags=["internal"])
async def root() -> dict[str, str]:
    """Return the application name.

    Returns:
        JSON with a ``message`` field.
    """
    return {"message": API_TITLE}
