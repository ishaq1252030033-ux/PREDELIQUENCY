"""Application configuration loaded from environment variables (.env)."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings with environment-aware configuration and feature flags."""

    # ------------------------------------------------------------------ #
    # Core
    # ------------------------------------------------------------------ #
    app_name: str = Field(
        default="Pre-Delinquency Intervention Engine",
        description="Human-readable application name.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (never enable in production).",
    )
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment: development, staging, production.",
    )

    # ------------------------------------------------------------------ #
    # API / security
    # ------------------------------------------------------------------ #
    api_prefix: str = Field(
        default="/api/v1",
        description="URL prefix for API routes.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Shared API key for securing endpoints (required in staging/production).",
    )

    # ------------------------------------------------------------------ #
    # Database / secrets
    # ------------------------------------------------------------------ #
    database_url: str = Field(
        default="sqlite:///./pre_delinquency.db",
        description="SQLAlchemy database URL. Use managed DB in staging/production.",
    )

    # ------------------------------------------------------------------ #
    # Feature flags
    # ------------------------------------------------------------------ #
    enable_interventions: bool = Field(
        default=False,
        description="Whether intervention workflows are enabled.",
    )
    auto_trigger_threshold: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Risk score (0-100) above which interventions may auto-trigger.",
    )
    batch_size: int = Field(
        default=100,
        gt=0,
        description="Default batch size for background jobs and batch predictions.",
    )
    cache_ttl: int = Field(
        default=3600,
        gt=0,
        description="Prediction cache TTL in seconds.",
    )

    # ------------------------------------------------------------------ #
    # Model configuration
    # ------------------------------------------------------------------ #
    model_path: Path = Field(
        default=Path("ml/models/risk_model.pkl"),
        description="Path to primary ML model file.",
    )
    model_version: Optional[str] = Field(
        default=None,
        description="Deployed model version identifier (e.g., git SHA, semantic version).",
    )
    retrain_schedule: Optional[str] = Field(
        default=None,
        description="Retraining schedule (e.g., cron expression or 'weekly', 'monthly').",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    """Return application settings instance (loaded from environment/.env)."""
    return Settings()

