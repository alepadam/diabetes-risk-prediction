"""Unit tests for src/interpret.py.

Uses a small RandomForest fitted on data with an intentionally strong,
known signal in one feature — so we can assert SHAP actually identifies
that feature as most important, not just that the functions run.
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from src.interpret import (
    compute_shap_values,
    find_extreme_predictions,
    get_top_features_by_shap,
    plot_shap_bar,
    plot_shap_summary,
    plot_shap_waterfall,
)
from src.train import train_logistic_regression, train_random_forest


@pytest.fixture
def fitted_tree_model_and_data():
    rng = np.random.default_rng(42)
    n = 300
    # strong_signal fully determines the target; noise_feature is unrelated
    target = rng.binomial(1, 0.3, n)
    X = pd.DataFrame({
        "strong_signal": np.where(target == 1, rng.normal(5, 0.5, n), rng.normal(0, 0.5, n)),
        "noise_feature": rng.normal(0, 1, n),
    })
    y = pd.Series(target)
    model = train_random_forest(X, y, {"n_estimators": 50, "max_depth": 6, "class_weight": "balanced", "n_jobs": -1})
    return model, X, y


def test_compute_shap_values_tree_returns_correct_shape(fitted_tree_model_and_data):
    model, X, _y = fitted_tree_model_and_data
    shap_values = compute_shap_values(model, X, model_type="tree")
    assert shap_values.values.shape == (len(X), X.shape[1])


def test_compute_shap_values_linear_returns_correct_shape():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = pd.Series(rng.binomial(1, 0.3, n))
    model = train_logistic_regression(X, y, {"max_iter": 500, "class_weight": "balanced"})
    shap_values = compute_shap_values(model, X, model_type="linear")
    assert shap_values.values.shape == (len(X), X.shape[1])


def test_compute_shap_values_raises_on_invalid_model_type(fitted_tree_model_and_data):
    model, X, _y = fitted_tree_model_and_data
    with pytest.raises(ValueError, match="Unknown model_type"):
        compute_shap_values(model, X, model_type="not_a_real_type")


def test_get_top_features_by_shap_identifies_strong_signal(fitted_tree_model_and_data):
    model, X, _y = fitted_tree_model_and_data
    shap_values = compute_shap_values(model, X, model_type="tree")
    top_features = get_top_features_by_shap(shap_values, X, n=2)
    # the deliberately strong, target-determining feature should rank first
    assert top_features.iloc[0]["feature"] == "strong_signal"
    assert top_features.iloc[0]["mean_abs_shap"] > top_features.iloc[1]["mean_abs_shap"]


def test_find_extreme_predictions_returns_actual_extremes(fitted_tree_model_and_data):
    model, X, _y = fitted_tree_model_and_data
    extremes = find_extreme_predictions(model, X, n=3)
    proba = model.predict_proba(X)[:, 1]

    highest_proba = proba[extremes["highest_risk_idx"]]
    lowest_proba = proba[extremes["lowest_risk_idx"]]

    assert highest_proba.min() >= np.sort(proba)[-3]  # within the true top 3
    assert lowest_proba.max() <= np.sort(proba)[2]  # within the true bottom 3


def test_shap_plots_run_without_error(fitted_tree_model_and_data):
    model, X, _y = fitted_tree_model_and_data
    shap_values = compute_shap_values(model, X, model_type="tree")

    assert plot_shap_summary(shap_values, X) is not None
    assert plot_shap_bar(shap_values, X) is not None
    assert plot_shap_waterfall(shap_values, index=0) is not None
