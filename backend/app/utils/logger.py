"""Structured logging with rotation and separate files for API, predictions, and errors."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

# Default log directory: project root / logs (backend/app/utils -> 3 levels up = root)
_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_LOG_DIR = _ROOT / "logs"

# Rotation: 10MB per file, keep 5 backups
MAX_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


class StructuredFormatter(logging.Formatter):
    """Format: timestamp | level | module | message | user_context (if present)."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record with structured pipe-delimited fields."""
        base = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if getattr(record, "user_context", None) not in (None, ""):
            base += " | user_context=%(user_context)s"
        # Ensure asctime is set
        if not hasattr(record, "asctime") or not record.asctime:
            record.asctime = self.formatTime(record, self.datefmt)
        msg = record.getMessage()
        name = record.name
        levelname = record.levelname
        user_ctx = getattr(record, "user_context", None) or ""
        if user_ctx:
            return f"{record.asctime} | {levelname} | {name} | {msg} | user_context={user_ctx}"
        return f"{record.asctime} | {levelname} | {name} | {msg}"


def _make_rotating_handler(
    filepath: Path,
    level: int = logging.DEBUG,
    formatter: Optional[logging.Formatter] = None,
) -> RotatingFileHandler:
    """Create a rotating file handler with 10 MB max size and 5 backups."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(filepath),
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    if formatter is None:
        formatter = StructuredFormatter(
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    handler.setFormatter(formatter)
    return handler


class ErrorLevelFilter(logging.Filter):
    """Only allow ERROR and CRITICAL."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.DEBUG,
    console: bool = True,
) -> None:
    """
    Configure application-wide structured logging.

    - log_dir: directory for log files (default: project_root/logs)
    - level: minimum level for root logger
    - console: whether to attach a StreamHandler to root
    """
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir = Path(log_dir)
    fmt = StructuredFormatter(datefmt="%Y-%m-%d %H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers to avoid duplicate logs
    for h in root.handlers[:]:
        root.removeHandler(h)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # errors.log: only ERROR and above (from any logger that propagates to root)
    errors_handler = _make_rotating_handler(log_dir / "errors.log", level=logging.ERROR, formatter=fmt)
    errors_handler.addFilter(ErrorLevelFilter())
    root.addHandler(errors_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger with structured format; propagates to root (console + errors.log)."""
    return logging.getLogger(name)


def get_api_requests_logger() -> logging.Logger:
    """Logger that writes only to api_requests.log (no propagation)."""
    log_dir = Path(os.environ.get("LOG_DIR", str(_DEFAULT_LOG_DIR)))
    logger = logging.getLogger("backend.api_requests")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _make_rotating_handler(log_dir / "api_requests.log")
    logger.addHandler(handler)
    return logger


def get_predictions_logger() -> logging.Logger:
    """Logger that writes only to predictions.log (no propagation)."""
    log_dir = Path(os.environ.get("LOG_DIR", str(_DEFAULT_LOG_DIR)))
    logger = logging.getLogger("backend.predictions")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handler = _make_rotating_handler(log_dir / "predictions.log")
    logger.addHandler(handler)
    return logger


def log_with_context(logger: logging.Logger, level: int, msg: str, *args: Any, user_context: Optional[str] = None, **kwargs: Any) -> None:
    """Log message with optional user_context in extra."""
    extra = kwargs.pop("extra", None) or {}
    if user_context is not None:
        extra["user_context"] = str(user_context)
    if extra:
        kwargs["extra"] = extra
    logger.log(level, msg, *args, **kwargs)
