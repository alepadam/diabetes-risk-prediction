"""Evaluation utilities: metrics beyond accuracy, and standard plots.

Accuracy alone is misleading on this imbalanced target (~86/14 split), so
every model comparison in this project should go through evaluate_model()
rather than a bare .score() call.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Compute the standard metric set for one model and return as a dict row,
    suitable for collecting into a comparison table across models.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    return metrics


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, model.predict(X_test))
    ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_roc_curves(models: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series, save_path: str | None = None):
    """Overlay ROC curves for multiple models on one plot for direct comparison."""
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def build_comparison_table(all_metrics: list[dict]) -> pd.DataFrame:
    """Turn a list of evaluate_model() outputs into the results table for the README."""
    return pd.DataFrame(all_metrics).set_index("model").round(3)
