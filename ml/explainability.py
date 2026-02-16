"""Explainability utilities using SHAP for the risk model.

Provides:
- ExplainabilityService: computes SHAP values for a given customer
- Force and waterfall plots as base64-encoded PNGs
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

from backend.app.services.risk_predictor import RiskPredictor
from backend.app.utils.logger import get_logger


logger = get_logger(__name__)


class ExplainabilityService:
    """Wraps SHAP explainability for the trained risk model."""

    def __init__(self, predictor: RiskPredictor) -> None:
        self.predictor = predictor
        self.model = predictor.model
        self.features_df = predictor.features_df
        self.feature_cols = predictor.feature_cols

        # Reuse predictor's explainer if available, otherwise create one
        explainer = getattr(predictor, "_explainer", None)
        if explainer is None:
            logger.info("Initializing SHAP explainer inside ExplainabilityService")
            bg = self.features_df[self.feature_cols].copy()
            if len(bg) > 500:
                bg = bg.sample(500, random_state=42)
            explainer = shap.TreeExplainer(self.model, bg.to_numpy())
            setattr(predictor, "_explainer", explainer)
        self.explainer = explainer

    def _get_row(self, customer_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return feature row X and shap values for the given customer."""
        row = self.features_df[self.features_df["customer_id"] == customer_id]
        if row.empty:
            raise ValueError(f"No features found for customer_id={customer_id}")

        X = row[self.feature_cols].to_numpy()
        shap_values = self._compute_shap_values(X)

        # Normalize SHAP output shape
        if isinstance(shap_values, list):
            arr = np.array(shap_values[1 if len(shap_values) > 1 else 0])
        else:
            arr = np.array(shap_values)

        if arr.ndim == 1:
            row_vals = arr
        else:
            row_vals = arr[0]

        return X[0], row_vals

    def _compute_shap_values(self, X: np.ndarray):
        """Use the same pattern as RiskPredictor._compute_shap_values."""
        try:
            values = self.explainer.shap_values(X)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to compute SHAP values in ExplainabilityService: %s", exc)
            raise
        return values

    def _expected_value(self) -> Optional[float]:
        """Extract a scalar expected value from the explainer."""
        try:
            expected = self.explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                if len(expected) > 1:
                    return float(expected[1])
                return float(expected[0])
            return float(expected)
        except Exception:
            return None

    @staticmethod
    def _plot_to_base64() -> str:
        """Save current matplotlib figure to base64 PNG string."""
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(plt.gcf())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _make_force_plot(
        self,
        base_value: Optional[float],
        shap_values_row: np.ndarray,
        x_row: np.ndarray,
    ) -> Optional[str]:
        """Generate a SHAP force plot and return it as a base64 PNG string."""
        try:
            if base_value is None:
                return None
            plt.figure()
            shap.force_plot(
                base_value,
                shap_values_row,
                x_row,
                matplotlib=True,
                show=False,
            )
            return self._plot_to_base64()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to generate SHAP force plot: %s", exc)
            plt.close(plt.gcf())
            return None

    def _make_waterfall_plot(
        self,
        base_value: Optional[float],
        shap_values_row: np.ndarray,
        x_row: np.ndarray,
    ) -> Optional[str]:
        """Generate a SHAP waterfall plot and return it as a base64 PNG string."""
        try:
            if base_value is None:
                return None
            expl = shap.Explanation(
                values=shap_values_row,
                base_values=base_value,
                data=x_row,
                feature_names=self.feature_cols,
            )
            plt.figure()
            shap.plots.waterfall(expl, show=False)
            return self._plot_to_base64()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to generate SHAP waterfall plot: %s", exc)
            plt.close(plt.gcf())
            return None

    def explain_customer(self, customer_id: str) -> Dict[str, object]:
        """
        Generate SHAP explanation for a single customer.

        Returns:
            dict with base_value, shap_values, feature_contributions, and base64 plots.
        """
        x_row, shap_row = self._get_row(customer_id)
        base_value = self._expected_value()

        contributions: List[Dict[str, float]] = []
        for idx, fname in enumerate(self.feature_cols):
            val = float(shap_row[idx])
            contributions.append(
                {
                    "feature": fname,
                    "shap_value": val,
                    "abs_value": abs(val),
                }
            )
        contributions.sort(key=lambda x: x["abs_value"], reverse=True)

        force_b64 = self._make_force_plot(base_value, shap_row, x_row)
        waterfall_b64 = self._make_waterfall_plot(base_value, shap_row, x_row)

        return {
            "customer_id": customer_id,
            "base_value": base_value,
            "feature_names": self.feature_cols,
            "shap_values": [float(v) for v in shap_row],
            "feature_contributions": contributions,
            "force_plot_base64": force_b64,
            "waterfall_plot_base64": waterfall_b64,
        }

