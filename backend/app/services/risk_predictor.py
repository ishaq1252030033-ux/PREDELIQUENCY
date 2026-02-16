"""Risk prediction service wrapping the trained ML model.

This module provides :class:`RiskPredictor`, the central service for
computing pre-delinquency risk scores.  It loads the serialised model,
the pre-computed feature matrix, and a SHAP explainer, then exposes
``predict_risk`` and ``batch_predict`` methods with an in-memory TTL cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from pydantic import ValidationError

from backend.app.models.schemas import RiskLevel, RiskScore
from backend.app.utils.logger import get_logger

# Resolve project root so default paths work from any working directory.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal cache data-class
# ---------------------------------------------------------------------------


@dataclass
class _CacheEntry:
    """A single entry in the in-memory prediction cache.

    Attributes:
        value: The cached :class:`RiskScore`.
        created_at: Timestamp when the entry was stored.
    """

    value: RiskScore
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# RiskPredictor
# ---------------------------------------------------------------------------


class RiskPredictor:
    """Service for computing customer default-risk scores.

    The predictor loads a serialised scikit-learn / XGBoost / LightGBM
    model and the pre-computed feature matrix at init time.  Predictions
    are cached in memory with a configurable TTL.

    Attributes:
        model: The loaded ML model object.
        features_df: Pre-computed feature matrix (one row per customer).
        feature_cols: Ordered list of feature column names used by the model.
    """

    def __init__(
        self,
        model_path: Path | str = _PROJECT_ROOT / "ml" / "models" / "risk_model.pkl",
        features_path: Path | str = (
            _PROJECT_ROOT / "ml" / "data" / "processed" / "features.csv"
        ),
        fi_path: Path | str = (
            _PROJECT_ROOT / "ml" / "reports" / "feature_importance.csv"
        ),
        cache_ttl_seconds: int = 3600,
    ) -> None:
        """Initialise the predictor by loading model, features, and SHAP explainer.

        Args:
            model_path: Filesystem path to the serialised model (``.pkl``).
            features_path: Path to the pre-computed features CSV.
            fi_path: Path to the feature-importance CSV (optional).
            cache_ttl_seconds: Number of seconds a cached prediction remains
                valid.  Defaults to 3 600 (one hour).

        Raises:
            FileNotFoundError: If the model or features file is missing.
            ValueError: If ``features.csv`` lacks a ``customer_id`` column.

        Note:
            For simplicity the implementation reads the *training* feature
            matrix rather than re-computing features in real time.  A
            production deployment would invoke the same
            ``FeatureEngineer`` pipeline used during training.
        """
        self.model_path = Path(model_path)
        self.features_path = Path(features_path)
        self.fi_path = Path(fi_path)
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")

        logger.info("Loading trained model from %s", self.model_path)
        self.model: Any = joblib.load(self.model_path)

        logger.info("Loading features from %s", self.features_path)
        self.features_df: pd.DataFrame = pd.read_csv(self.features_path)
        if "customer_id" not in self.features_df.columns:
            raise ValueError("features.csv must contain a 'customer_id' column.")

        self.feature_cols: list[str] = [
            c for c in self.features_df.columns if c != "customer_id"
        ]

        # Feature importance (optional)
        self._feature_importance: Optional[pd.DataFrame] = None
        if self.fi_path.exists():
            try:
                self._feature_importance = pd.read_csv(self.fi_path)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to load feature importance from %s: %s",
                    self.fi_path,
                    exc,
                )

        # SHAP explainer (background sample capped at 500 rows)
        logger.info("Initialising SHAP explainer for risk model")
        bg = self.features_df[self.feature_cols].copy()
        if len(bg) > 500:
            bg = bg.sample(500, random_state=42)
        try:
            self._explainer: Optional[shap.TreeExplainer] = shap.TreeExplainer(
                self.model, bg.to_numpy()
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to initialise SHAP explainer: %s", exc)
            self._explainer = None

        # In-memory TTL prediction cache
        self._cache: Dict[str, _CacheEntry] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_risk(self, customer_id: str) -> RiskScore:
        """Predict risk for a single customer.

        Args:
            customer_id: The customer identifier to score.

        Returns:
            A :class:`RiskScore` Pydantic model.

        Raises:
            ValueError: If no feature row exists for *customer_id*.
            ValidationError: If the score cannot be serialised.
        """
        cached = self._get_from_cache(customer_id)
        if cached is not None:
            return cached

        row = self.features_df[self.features_df["customer_id"] == customer_id]
        if row.empty:
            logger.error("No features found for customer_id=%s", customer_id)
            raise ValueError(f"No features found for customer_id={customer_id}")

        x_arr: np.ndarray = row[self.feature_cols].to_numpy()

        prob_default = self._predict_proba(x_arr)[0]
        risk_score_value = prob_default * 100.0
        risk_level = self._map_score_to_level(risk_score_value)
        top_factors = self._get_top_risk_factors(x_arr)

        try:
            risk = RiskScore(
                customer_id=customer_id,
                risk_score=risk_score_value,
                risk_level=risk_level,
                prediction_date=datetime.now(),
                top_risk_factors=top_factors,
                recommended_action=self._suggest_action(risk_level),
            )
        except ValidationError as exc:
            logger.error("Failed to construct RiskScore for %s: %s", customer_id, exc)
            raise

        self._store_in_cache(customer_id, risk)
        return risk

    def batch_predict(self, customer_ids: List[str]) -> List[RiskScore]:
        """Predict risk for multiple customers using vectorised inference.

        Cached results are returned immediately; only uncached IDs trigger
        model inference.

        Args:
            customer_ids: List of customer identifiers.

        Returns:
            List of :class:`RiskScore` (may be shorter than input if some
            IDs have no features).
        """
        results: List[RiskScore] = []
        missing_ids: List[str] = []

        for cid in customer_ids:
            cached = self._get_from_cache(cid)
            if cached is not None:
                results.append(cached)
            else:
                missing_ids.append(cid)

        if not missing_ids:
            return results

        subset = self.features_df[self.features_df["customer_id"].isin(missing_ids)]
        if subset.empty:
            logger.warning("No features found for any of the requested customers.")
            return results

        x_arr: np.ndarray = subset[self.feature_cols].to_numpy()
        cids: list[str] = subset["customer_id"].tolist()

        probs = self._predict_proba(x_arr)
        shap_values = self._compute_shap_values(x_arr)

        for idx, cid in enumerate(cids):
            risk_score_value = float(probs[idx] * 100.0)
            level = self._map_score_to_level(risk_score_value)
            top_factors = self._extract_top_factors_for_row(shap_values, idx)
            try:
                risk = RiskScore(
                    customer_id=cid,
                    risk_score=risk_score_value,
                    risk_level=level,
                    prediction_date=datetime.now(),
                    top_risk_factors=top_factors,
                    recommended_action=self._suggest_action(level),
                )
            except ValidationError as exc:
                logger.error("Failed to construct RiskScore for %s: %s", cid, exc)
                continue

            self._store_in_cache(cid, risk)
            results.append(risk)

        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature-importance mapping for explainability dashboards.

        Uses ``ensemble_importance`` from the CSV report if available;
        otherwise falls back to ``model.feature_importances_``.

        Returns:
            Dict mapping feature name to importance score.
        """
        importance: Dict[str, float] = {}

        if self._feature_importance is not None:
            fi = self._feature_importance
            col: Optional[str] = None
            for candidate in (
                "ensemble_importance",
                "xgboost_importance",
                "lightgbm_importance",
            ):
                if candidate in fi.columns:
                    col = candidate
                    break
            if col:
                for _, row in fi.iterrows():
                    importance[str(row["feature"])] = float(row[col])
                return importance

        if hasattr(self.model, "feature_importances_"):
            vals = np.array(self.model.feature_importances_, dtype=float)
            for name, val in zip(self.feature_cols, vals):
                importance[name] = float(val)

        return importance

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return probability-of-default for each row in *x*.

        Args:
            x: 2-D feature array of shape ``(n_samples, n_features)``.

        Returns:
            1-D array of default probabilities.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1].astype(float)
        scores = self.model.decision_function(x)
        return (1 / (1 + np.exp(-scores))).astype(float)

    def _get_from_cache(self, customer_id: str) -> Optional[RiskScore]:
        """Return cached :class:`RiskScore` if present and not expired.

        Args:
            customer_id: The customer to look up.

        Returns:
            The cached score, or ``None`` if missing / expired.
        """
        entry = self._cache.get(customer_id)
        if not entry:
            return None
        if datetime.now() - entry.created_at > self.cache_ttl:
            self._cache.pop(customer_id, None)
            return None
        return entry.value

    def _store_in_cache(self, customer_id: str, value: RiskScore) -> None:
        """Store a :class:`RiskScore` in the TTL cache.

        Args:
            customer_id: Cache key.
            value: The risk score to cache.
        """
        self._cache[customer_id] = _CacheEntry(value=value, created_at=datetime.now())

    @staticmethod
    def _map_score_to_level(score: float) -> RiskLevel:
        """Map a 0-100 numeric score to a discrete :class:`RiskLevel`.

        Args:
            score: Risk score on the 0-100 scale.

        Returns:
            The corresponding risk-level enum member.
        """
        if score <= 40:
            return RiskLevel.LOW
        if score <= 70:
            return RiskLevel.MEDIUM
        if score <= 85:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    @staticmethod
    def _suggest_action(level: RiskLevel) -> str:
        """Return a recommended next-best action string.

        Args:
            level: The customer's risk level.

        Returns:
            A short human-readable recommendation.
        """
        actions: dict[RiskLevel, str] = {
            RiskLevel.LOW: "Monitor account; no immediate action required.",
            RiskLevel.MEDIUM: (
                "Send gentle payment reminder and highlight upcoming dues."
            ),
            RiskLevel.HIGH: (
                "Proactively offer restructuring / partial-payment options "
                "via digital channels."
            ),
            RiskLevel.CRITICAL: (
                "Trigger high-priority outreach via call centre and "
                "collections team with custom recovery plan."
            ),
        }
        return actions.get(level, actions[RiskLevel.CRITICAL])

    def _get_top_risk_factors(self, x: np.ndarray, top_k: int = 3) -> List[str]:
        """Compute the top *top_k* risk factors for a single-row input.

        Args:
            x: 2-D array of shape ``(1, n_features)``.
            top_k: Number of top factors to return.

        Returns:
            List of human-readable factor strings.
        """
        shap_values = self._compute_shap_values(x)
        return self._extract_top_factors_for_row(shap_values, row_index=0, top_k=top_k)

    def _compute_shap_values(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Compute SHAP values for the given feature matrix.

        Args:
            x: 2-D feature array.

        Returns:
            SHAP values array, or ``None`` if the explainer is unavailable.
        """
        if self._explainer is None:
            return None
        try:
            return self._explainer.shap_values(x)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to compute SHAP values: %s", exc)
            return None

    def _extract_top_factors_for_row(
        self,
        shap_values: Any,
        row_index: int,
        top_k: int = 3,
    ) -> List[str]:
        """Extract the top absolute-SHAP contributions for one sample.

        Args:
            shap_values: Raw SHAP output (array or list of arrays).
            row_index: Index of the sample within the batch.
            top_k: Number of factors to extract.

        Returns:
            List of formatted factor strings (e.g.
            ``"feature_name (↑, |SHAP|=0.123)"``).
        """
        if shap_values is None:
            return []

        # Handle per-class list (binary: take class-1)
        if isinstance(shap_values, list):
            arr = np.array(shap_values[1 if len(shap_values) > 1 else 0])
        else:
            arr = np.array(shap_values)

        row_vals = arr if arr.ndim == 1 else arr[row_index]

        abs_vals = np.abs(row_vals)
        n = min(len(self.feature_cols), len(abs_vals))
        top_indices = np.argsort(abs_vals[:n])[::-1][:top_k]

        factors: List[str] = []
        for idx in top_indices:
            fname = self.feature_cols[idx]
            sign = "\u2191" if row_vals[idx] > 0 else "\u2193"
            factors.append(f"{fname} ({sign}, |SHAP|={abs_vals[idx]:.3f})")
        return factors
