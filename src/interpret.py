"""Model interpretability via SHAP.

Supports both tree-based models (Random Forest, XGBoost — via TreeExplainer,
fast and exact) and linear models (Logistic Regression — via LinearExplainer).
The notebook picks the right explainer based on which model won in 04, rather
than this module guessing from the model's class.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_shap_values(model, X: pd.DataFrame, model_type: str):
    """Compute SHAP values for a fitted model.

    model_type: 'tree' for Random Forest/XGBoost, 'linear' for Logistic
    Regression. Returns a shap.Explanation object.
    """
    import shap

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        raise ValueError(f"Unknown model_type '{model_type}' — expected 'tree' or 'linear'")

    shap_values = explainer(X)

    # Binary classifiers via TreeExplainer sometimes return values for both
    # classes (shape [n_samples, n_features, 2]) — keep the positive class.
    if shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, save_path: str | None = None, max_display: int = 15):
    """Beeswarm summary plot — shows both feature importance and the
    direction of each feature's effect (high/low values pushing risk up or down).
    """
    import shap

    fig = plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_bar(shap_values, X: pd.DataFrame, save_path: str | None = None, max_display: int = 15):
    """Bar plot of mean |SHAP value| per feature — pure importance ranking,
    without the direction information the beeswarm plot carries.
    """
    import shap

    fig = plt.figure(figsize=(9, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_waterfall(shap_values, index: int, save_path: str | None = None):
    """Explain a single prediction — shows exactly which features pushed
    that individual's risk score up or down from the baseline.
    """
    import shap

    fig = plt.figure(figsize=(9, 6))
    shap.waterfall_plot(shap_values[index], show=False)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def get_top_features_by_shap(shap_values, X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value — a data table version of
    the bar plot, useful for the README results table or further analysis.
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    result = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return result.head(n)


def find_extreme_predictions(model, X: pd.DataFrame, n: int = 2) -> dict:
    """Find indices of the highest and lowest predicted-risk individuals in
    X — useful picks for the waterfall (single-prediction) plots, since a
    random row is often unremarkable.
    """
    proba = model.predict_proba(X)[:, 1]
    order = np.argsort(proba)
    return {
        "highest_risk_idx": order[-n:][::-1].tolist(),
        "lowest_risk_idx": order[:n].tolist(),
    }
