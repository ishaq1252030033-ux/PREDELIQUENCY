"""
Evaluate trained risk model on the test set and generate reports/plots.

Steps:
- Load trained model and engineered features + labels
- Recreate train/val/test split (same random_state as training)
- Run predictions on the test set
- Generate:
  * ROC curve
  * Precision-Recall curve
  * Confusion matrix heatmap
  * Feature importance bar chart (top features)
  * Prediction probability distribution histogram
- Compute business metrics over thresholds and find threshold for ~80% recall
- Save plots to ml/reports/ and print summary statistics
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


logger = logging.getLogger("evaluate_model")


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
    """Load features and labels, merge on customer_id, and return X, y, feature names."""
    logger.info("Loading features from %s", features_path)
    features = pd.read_csv(features_path)

    logger.info("Loading labels from %s", labels_path)
    labels = pd.read_csv(labels_path)

    if "customer_id" not in features.columns or "customer_id" not in labels.columns:
        raise ValueError("customer_id must be present in both features and labels.")

    data = features.merge(labels, on="customer_id", how="inner")
    if "default" not in data.columns:
        raise ValueError("'default' column not found after merging features and labels.")

    y = data["default"].astype(int)
    feature_cols = [c for c in data.columns if c not in ("customer_id", "default")]
    X = data[feature_cols].astype(float)

    logger.info("Loaded data: %d rows, %d features", len(X), X.shape[1])
    return X, y, pd.Index(feature_cols)


def split_data_70_15_15(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Recreate the same 70/15/15 stratified split as in training."""
    logger.info("Recreating train/val/test split (70/15/15)")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=random_state,
    )

    val_fraction = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(
        "Split sizes (recreated): train=%d, val=%d, test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_business_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.8,
) -> Dict[str, object]:
    """
    Compute TPR/FPR over thresholds and choose threshold for target recall.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    rows = []
    for thr, fp, tp in zip(thresholds, fpr, tpr):
        rows.append(
            {
                "threshold": float(thr),
                "tpr": float(tp),
                "fpr": float(fp),
            }
        )

    tpr_arr = np.array([r["tpr"] for r in rows])
    thr_arr = np.array([r["threshold"] for r in rows])
    fpr_arr = np.array([r["fpr"] for r in rows])

    candidates = np.where(tpr_arr >= target_recall)[0]
    if len(candidates) > 0:
        # Among candidates with recall >= target, pick lowest FPR
        idx = candidates[np.argmin(fpr_arr[candidates])]
    else:
        # Fall back to point with max TPR
        idx = int(np.argmax(tpr_arr))

    optimal = {
        "threshold": float(thr_arr[idx]),
        "tpr": float(tpr_arr[idx]),
        "fpr": float(fpr_arr[idx]),
    }

    return {
        "roc_points": rows,
        "optimal_for_target_recall": optimal,
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return float(auc)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label="PR curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Default", "Default"],
        yticklabels=["No Default", "Default"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_feature_importance_bar(
    fi_path: Path,
    out_path: Path,
    top_n: int = 10,
) -> None:
    """Plot top N features from feature_importance.csv produced by train_model."""
    if not fi_path.exists():
        logger.warning("Feature importance file %s not found; skipping bar chart.", fi_path)
        return

    fi_df = pd.read_csv(fi_path)
    importance_col = None
    for col in ["ensemble_importance", "xgboost_importance", "lightgbm_importance"]:
        if col in fi_df.columns:
            importance_col = col
            break
    if importance_col is None:
        logger.warning("No importance columns found in %s; skipping bar chart.", fi_path)
        return

    fi_df_sorted = fi_df.sort_values(importance_col, ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=importance_col,
        y="feature",
        data=fi_df_sorted,
        orient="h",
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Features by {importance_col}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_path: Path,
) -> None:
    """Plot histogram of predicted probabilities, colored by true label."""
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(
        data=df,
        x="y_prob",
        hue="y_true",
        bins=20,
        stat="density",
        common_norm=False,
        palette={0: "tab:blue", 1: "tab:orange"},
        ax=ax,
    )
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Probability Distribution by True Label")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained risk model on test set")
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
        "--model",
        type=Path,
        default=Path("ml/models/risk_model.pkl"),
        help="Path to trained model .pkl",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("ml/reports"),
        help="Directory to save evaluation plots and summaries",
    )
    parser.add_argument(
        "--fi-path",
        type=Path,
        default=Path("ml/reports/feature_importance.csv"),
        help="Path to feature_importance.csv (from training)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    reports_dir: Path = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data and recreate split
    X, y, _ = load_data(args.features, args.labels)
    _, _, X_test, _, _, y_test = split_data_70_15_15(X, y)

    # Load model
    logger.info("Loading trained model from %s", args.model)
    model = joblib.load(args.model)

    # Predictions
    logger.info("Running predictions on test set")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
        y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9)

    y_pred = (y_prob >= 0.5).astype(int)

    # Basic metrics at 0.5 threshold
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, zero_division=0)

    # Business metrics (TPR/FPR over thresholds, optimal threshold for ~80% recall)
    business = compute_business_metrics(y_test.values, y_prob)

    # Plots
    roc_path = reports_dir / "roc_curve.png"
    pr_path = reports_dir / "pr_curve.png"
    cm_path = reports_dir / "confusion_matrix_heatmap.png"
    fi_bar_path = reports_dir / "feature_importance_bar.png"
    pred_hist_path = reports_dir / "prediction_distribution.png"

    plot_roc_curve(y_test.values, y_prob, roc_path)
    plot_pr_curve(y_test.values, y_prob, pr_path)
    plot_confusion_matrix(y_test.values, y_pred, cm_path)
    plot_feature_importance_bar(args.fi_path, fi_bar_path, top_n=10)
    plot_prediction_distribution(y_test.values, y_prob, pred_hist_path)

    # Summary JSON
    summary = {
        "metrics_threshold_0_5": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist(),
        },
        "business_metrics": business,
        "n_test_samples": int(len(y_test)),
    }

    summary_path = reports_dir / "evaluation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary statistics to console
    print("=== Evaluation Summary (threshold=0.5) ===")
    print(f"Accuracy       : {acc:.3f}")
    print(f"Precision      : {prec:.3f}")
    print(f"Recall         : {rec:.3f}")
    print(f"F1-score       : {f1:.3f}")
    print(f"ROC-AUC        : {auc:.3f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(cls_report)

    opt = business["optimal_for_target_recall"]
    print("\n=== Optimal threshold for 80% recall (approx) ===")
    print(f"Threshold      : {opt['threshold']:.3f}")
    print(f"TPR (Recall)   : {opt['tpr']:.3f}")
    print(f"FPR            : {opt['fpr']:.3f}")

    logger.info("Evaluation complete. Reports saved to %s", reports_dir)


if __name__ == "__main__":
    main()

