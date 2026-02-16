"""Tests for the RiskPredictor service."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

from backend.app.services.risk_predictor import RiskPredictor
from backend.app.models.schemas import RiskLevel, RiskScore


@pytest.fixture
def dummy_model_and_features(tmp_path) -> Tuple[Path, Path, Path]:
    """Create a tiny logistic regression model and matching features CSV."""
    # Create simple training data
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.9, 1.0],
            [1.0, 0.9],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression()
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    # Features for two dummy customers
    features_df = pd.DataFrame(
        [
            {"customer_id": "C_LOW", "f1": 0.1, "f2": 0.2},
            {"customer_id": "C_HIGH", "f1": 0.9, "f2": 1.0},
        ]
    )
    features_path = tmp_path / "features.csv"
    features_df.to_csv(features_path, index=False)

    # Simple feature importance file
    fi_df = pd.DataFrame(
        {
            "feature": ["f1", "f2"],
            "ensemble_importance": [0.6, 0.4],
        }
    )
    fi_path = tmp_path / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)

    return model_path, features_path, fi_path


@pytest.fixture
def predictor(dummy_model_and_features) -> RiskPredictor:
    model_path, features_path, fi_path = dummy_model_and_features
    return RiskPredictor(
        model_path=model_path,
        features_path=features_path,
        fi_path=fi_path,
        cache_ttl_seconds=60,
    )


def test_model_loading(predictor: RiskPredictor):
    """Model and features should load without error."""
    assert predictor.model is not None
    assert not predictor.features_df.empty
    assert len(predictor.feature_cols) == 2


def test_prediction_output_format(predictor: RiskPredictor):
    """predict_risk should return a RiskScore Pydantic model."""
    result = predictor.predict_risk("C_LOW")
    assert isinstance(result, RiskScore)
    assert isinstance(result.customer_id, str)
    assert isinstance(result.risk_score, float)
    assert isinstance(result.risk_level, RiskLevel)
    assert isinstance(result.prediction_date, type(result.prediction_date))


def test_risk_score_range(predictor: RiskPredictor):
    """Risk score should always be in [0, 100]."""
    for cid in ["C_LOW", "C_HIGH"]:
        r = predictor.predict_risk(cid)
        assert 0.0 <= r.risk_score <= 100.0


@pytest.mark.parametrize(
    "score,expected_level",
    [
        (0.0, RiskLevel.LOW),
        (40.0, RiskLevel.LOW),
        (41.0, RiskLevel.MEDIUM),
        (70.0, RiskLevel.MEDIUM),
        (71.0, RiskLevel.HIGH),
        (85.0, RiskLevel.HIGH),
        (86.0, RiskLevel.CRITICAL),
        (100.0, RiskLevel.CRITICAL),
    ],
)
def test_risk_level_assignment_logic(score: float, expected_level: RiskLevel):
    """_map_score_to_level should respect the configured thresholds."""
    level = RiskPredictor._map_score_to_level(score)
    assert level == expected_level


def test_batch_prediction_consistency(predictor: RiskPredictor):
    """Batch prediction should be consistent with single predictions."""
    single_low = predictor.predict_risk("C_LOW")
    single_high = predictor.predict_risk("C_HIGH")

    batch_results = predictor.batch_predict(["C_LOW", "C_HIGH"])
    by_id = {r.customer_id: r for r in batch_results}

    assert pytest.approx(by_id["C_LOW"].risk_score, rel=1e-6) == single_low.risk_score
    assert pytest.approx(by_id["C_HIGH"].risk_score, rel=1e-6) == single_high.risk_score


def test_feature_importance_output(predictor: RiskPredictor):
    """get_feature_importance should return a non-empty mapping."""
    fi = predictor.get_feature_importance()
    assert isinstance(fi, dict)
    assert fi  # non-empty
    assert set(fi.keys()) == {"f1", "f2"}

