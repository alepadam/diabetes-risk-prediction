"""Unit tests for src/evaluate.py."""
import matplotlib

matplotlib.use("Agg")  # headless backend for test environments without a display

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    build_comparison_table,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.train import train_logistic_regression, train_random_forest


@pytest.fixture
def fitted_model_and_data():
    rng = np.random.default_rng(42)
    n = 300
    target = rng.binomial(1, 0.3, n)
    X = pd.DataFrame({
        "f1": np.where(target == 1, rng.normal(1, 1, n), rng.normal(0, 1, n)),
        "f2": rng.normal(0, 1, n),
    })
    y = pd.Series(target)
    model = train_logistic_regression(X, y, {"max_iter": 500, "class_weight": "balanced"})
    return model, X, y


def test_evaluate_model_returns_all_expected_metrics(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    metrics = evaluate_model(model, X, y, model_name="Test Model")
    expected_keys = {"model", "accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics.keys()) == expected_keys
    assert metrics["model"] == "Test Model"
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0.0 <= metrics[key] <= 1.0


def test_evaluate_model_metrics_are_perfect_for_trivially_separable_data():
    # A feature that perfectly determines the label should yield perfect
    # metrics — a sanity check that evaluate_model computes them correctly,
    # not just that it returns some number in range.
    X = pd.DataFrame({"f1": [0, 0, 0, 10, 10, 10]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    model = train_logistic_regression(X, y, {"max_iter": 1000, "class_weight": "balanced"})
    metrics = evaluate_model(model, X, y, model_name="Perfect")
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_build_comparison_table_indexes_by_model_name(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    metrics_a = evaluate_model(model, X, y, model_name="Model A")
    metrics_b = evaluate_model(model, X, y, model_name="Model B")
    table = build_comparison_table([metrics_a, metrics_b])
    assert list(table.index) == ["Model A", "Model B"]
    assert "model" not in table.columns  # became the index, not a column


def test_plot_confusion_matrix_runs_without_error(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    fig = plot_confusion_matrix(model, X, y, model_name="Test Model")
    assert fig is not None


def test_plot_roc_curves_runs_without_error(fitted_model_and_data):
    model, X, y = fitted_model_and_data
    rf_model = train_random_forest(X, y, {"n_estimators": 20, "max_depth": 4, "class_weight": "balanced", "n_jobs": -1})
    fig = plot_roc_curves({"LogReg": model, "RF": rf_model}, X, y)
    assert fig is not None
