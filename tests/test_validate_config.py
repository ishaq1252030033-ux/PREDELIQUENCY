"""Tests for configuration validation."""

from __future__ import annotations

import pytest

from backend.app.utils.validate_config import validate_config
from backend.config import Environment, Settings


def test_dev_config_valid(tmp_path) -> None:
    """Development config with defaults should pass validation."""
    model_file = tmp_path / "model.pkl"
    model_file.touch()
    settings = Settings(
        environment=Environment.DEVELOPMENT,
        debug=True,
        model_path=model_file,
    )
    validate_config(settings)  # should not raise


def test_production_debug_true_fails(tmp_path) -> None:
    """Production + debug=True must fail."""
    model_file = tmp_path / "model.pkl"
    model_file.touch()
    settings = Settings(
        environment=Environment.PRODUCTION,
        debug=True,
        api_key="secret",
        database_url="postgresql://user:pass@host/db",
        model_path=model_file,
    )
    with pytest.raises(RuntimeError, match="DEBUG must be False"):
        validate_config(settings)


def test_production_missing_api_key_fails(tmp_path) -> None:
    """Production without API_KEY must fail."""
    model_file = tmp_path / "model.pkl"
    model_file.touch()
    settings = Settings(
        environment=Environment.PRODUCTION,
        debug=False,
        api_key=None,
        database_url="postgresql://user:pass@host/db",
        model_path=model_file,
    )
    with pytest.raises(RuntimeError, match="API_KEY is required"):
        validate_config(settings)


def test_production_sqlite_fails(tmp_path) -> None:
    """Production with SQLite must fail."""
    model_file = tmp_path / "model.pkl"
    model_file.touch()
    settings = Settings(
        environment=Environment.PRODUCTION,
        debug=False,
        api_key="secret",
        database_url="sqlite:///./test.db",
        model_path=model_file,
    )
    with pytest.raises(RuntimeError, match="must not use local SQLite"):
        validate_config(settings)


def test_missing_model_path_fails() -> None:
    """Non-existent model path must fail."""
    settings = Settings(
        environment=Environment.DEVELOPMENT,
        debug=False,
        model_path="/nonexistent/path/model.pkl",
    )
    with pytest.raises(RuntimeError, match="MODEL_PATH does not exist"):
        validate_config(settings)


def test_staging_requires_api_key(tmp_path) -> None:
    """Staging without API_KEY must fail."""
    model_file = tmp_path / "model.pkl"
    model_file.touch()
    settings = Settings(
        environment=Environment.STAGING,
        debug=False,
        api_key=None,
        database_url="postgresql://user:pass@host/db",
        model_path=model_file,
    )
    with pytest.raises(RuntimeError, match="API_KEY is required"):
        validate_config(settings)
