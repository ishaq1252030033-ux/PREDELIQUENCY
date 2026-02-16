"""Configuration validation utilities.

Loaded on startup to ensure required settings are present and sane.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from backend.config import Environment, Settings, get_settings
from backend.app.utils.logger import get_logger


logger = get_logger(__name__)


def validate_config(settings: Settings | None = None) -> None:
    """Validate critical configuration and raise RuntimeError on failures."""
    if settings is None:
        settings = get_settings()

    errors: List[str] = []

    env = settings.environment

    # Environment/DEBUG consistency
    if env is Environment.PRODUCTION and settings.debug:
        errors.append("DEBUG must be False in production.")

    # Secrets required in non-development environments
    if env in (Environment.STAGING, Environment.PRODUCTION):
        if not settings.api_key:
            errors.append("API_KEY is required in staging/production (env var API_KEY).")
        if settings.database_url.startswith("sqlite:///"):
            errors.append(
                "DATABASE_URL must not use local SQLite in staging/production. "
                "Configure a managed database."
            )

    # Model path existence — resolve relative paths against the project root
    # so validation works regardless of the working directory.
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    model_path = Path(settings.model_path)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.exists():
        errors.append(f"MODEL_PATH does not exist: {model_path}")

    # Basic numeric sanity checks (already constrained by config types)
    if settings.batch_size <= 0:
        errors.append("BATCH_SIZE must be > 0.")
    if settings.cache_ttl <= 0:
        errors.append("CACHE_TTL must be > 0.")

    if errors:
        for err in errors:
            logger.error("Config error: %s", err)
        raise RuntimeError(
            "Configuration validation failed. See log for details:\n"
            + "\n".join(f"- {e}" for e in errors)
        )

    logger.info(
        "Configuration validated: env=%s, debug=%s, model_path=%s, db=%s",
        settings.environment.value,
        settings.debug,
        settings.model_path,
        settings.database_url,
    )


def main() -> None:
    """CLI entry point for manual validation."""
    validate_config()
    print("Configuration validation succeeded.")


if __name__ == "__main__":
    main()

