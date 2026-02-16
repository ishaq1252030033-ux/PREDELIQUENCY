"""
Train ML models for pre-delinquency default prediction.

Pipeline:
- Load engineered features and labels
- Train/validation/test split (70/15/15)
- Handle class imbalance with SMOTE
- Train XGBoost, LightGBM, and an ensemble VotingClassifier
- Evaluate and save best model, metrics, and feature importances
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


logger = logging.getLogger("train_model")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def load_data(
    features_path: Path,
    labels_path: Path,
) -> Tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Load features and labels, merge on customer_id, and return X, y, and feature names.
    """
    logger.info("Loading features from %s", features_path)
    features = pd.read_csv(features_path)

    logger.info("Loading labels from %s", labels_path)
    labels = pd.read_csv(labels_path)

    if "customer_id" not in features.columns or "customer_id" not in labels.columns:
        raise ValueError("customer_id must be present in both features and labels.")

    data = features.merge(labels, on="customer_id", how="inner")
    if "default" not in data.columns:
        raise ValueError("'default' column not found after merging features and labels.")

    # Separate features and target
    y = data["default"].astype(int)
    feature_cols = [c for c in data.columns if c not in ("customer_id", "default")]
    X = data[feature_cols].astype(float)

    logger.info("Loaded data: %d rows, %d features", len(X), X.shape[1])
    return X, y, pd.Index(feature_cols)


def split_data(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Split into train/validation/test with proportions 70/15/15 (stratified).
    """
    logger.info("Splitting data into train/val/test (70/15/15)")

    # First split off test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state,
    )

    # Now split remaining into train (70%) and val (15%)
    # Remaining proportion = 0.85 -> validation fraction within temp:
    val_fraction = 0.15 / 0.85  # ~0.176
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to the training data to handle class imbalance.
    """
    logger.info("Applying SMOTE to training data")
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(
        "After SMOTE: train size=%d (class distribution: %s)",
        len(X_res),
        np.bincount(y_res.values),
    )
    return X_res, y_res


def build_models(random_state: int = 42) -> Dict[str, object]:
    """
    Create the three models: XGBoost, LightGBM, and Voting ensemble.
    """
    xgb = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=-1,
    )

    lgbm = LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
    )

    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb),
            ("lgbm", lgbm),
        ],
        voting="soft",
        n_jobs=-1,
    )

    return {
        "xgboost": xgb,
        "lightgbm": lgbm,
        "ensemble": ensemble,
    }


def evaluate_model(
    name: str,
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, object]:
    """
    Evaluate a model on validation or test data and return metrics.
    """
    logger.info("Evaluating model: %s", name)
    y_pred = model.predict(X_val)

    # Probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_val)
        y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)
    else:
        y_prob = y_pred

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
        "classification_report": classification_report(
            y_val, y_pred, zero_division=0, output_dict=True
        ),
    }
    return metrics


def compute_feature_importance(
    xgb_model: XGBClassifier,
    lgbm_model: LGBMClassifier,
    feature_names: pd.Index,
) -> pd.DataFrame:
    """
    Compute feature importances from XGBoost and LightGBM and average them.
    """
    xgb_imp = getattr(xgb_model, "feature_importances_", None)
    lgbm_imp = getattr(lgbm_model, "feature_importances_", None)

    if xgb_imp is None or lgbm_imp is None:
        logger.warning("Could not find feature_importances_ on one or more models.")
        return pd.DataFrame({"feature": feature_names})

    xgb_imp = np.array(xgb_imp, dtype=float)
    lgbm_imp = np.array(lgbm_imp, dtype=float)

    if xgb_imp.sum() > 0:
        xgb_norm = xgb_imp / xgb_imp.sum()
    else:
        xgb_norm = xgb_imp

    if lgbm_imp.sum() > 0:
        lgbm_norm = lgbm_imp / lgbm_imp.sum()
    else:
        lgbm_norm = lgbm_imp

    ensemble_imp = (xgb_norm + lgbm_norm) / 2.0

    fi_df = pd.DataFrame(
        {
            "feature": feature_names,
            "xgboost_importance": xgb_imp,
            "lightgbm_importance": lgbm_imp,
            "ensemble_importance": ensemble_imp,
        }
    ).sort_values("ensemble_importance", ascending=False)

    return fi_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for default prediction")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("ml/data/processed/features.csv"),
        help="Path to features.csv",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("ml/data/labels.csv"),
        help="Path to labels.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("ml/models"),
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("ml/reports"),
        help="Directory to save reports (JSON, CSV)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    models_dir: Path = args.models_dir
    reports_dir: Path = args.reports_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, feature_names = load_data(args.features, args.labels)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Apply SMOTE only to training data
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Build models
    models = build_models()

    # Train models with a simple progress bar
    trained_models: Dict[str, object] = {}
    logger.info("Training models...")
    for name in tqdm(models.keys(), desc="Training models"):
        model = models[name]
        logger.info("Fitting %s", name)
        model.fit(X_train_res, y_train_res)
        trained_models[name] = model

    # Evaluate on validation set
    logger.info("Evaluating models on validation set")
    val_metrics: Dict[str, Dict[str, object]] = {}
    for name, model in tqdm(trained_models.items(), desc="Validating models"):
        val_metrics[name] = evaluate_model(name, model, X_val, y_val)

    # Select best model based on validation ROC-AUC
    best_name = max(val_metrics.keys(), key=lambda n: val_metrics[n]["roc_auc"])
    best_model = trained_models[best_name]
    logger.info("Best model on validation set: %s (ROC-AUC=%.4f)", best_name, val_metrics[best_name]["roc_auc"])

    # Evaluate best model on test set
    logger.info("Evaluating best model on test set")
    test_metrics = evaluate_model(best_name, best_model, X_test, y_test)

    # Compute feature importance from base models (xgboost + lightgbm)
    logger.info("Computing feature importances")
    fi_df = compute_feature_importance(
        trained_models["xgboost"],
        trained_models["lightgbm"],
        feature_names,
    )

    # Save best model
    model_path = models_dir / "risk_model.pkl"
    logger.info("Saving best model (%s) to %s", best_name, model_path)
    joblib.dump(best_model, model_path)

    # Save metrics
    metrics_payload: Dict[str, object] = {
        "best_model": best_name,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "data_shape": {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
        },
        "splits": {
            "train": int(len(X_train)),
            "train_resampled": int(len(X_train_res)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
    }
    metrics_path = reports_dir / "model_performance.json"
    logger.info("Saving metrics to %s", metrics_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # Save feature importances
    fi_path = reports_dir / "feature_importance.csv"
    logger.info("Saving feature importances to %s", fi_path)
    fi_df.to_csv(fi_path, index=False)

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()

