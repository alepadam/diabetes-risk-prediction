"""Unit tests for src/train.py — model training, cross-validation, and
save/load round-tripping.
"""
import numpy as np
import pandas as pd
import pytest

from src.train import (
    cross_validate_model,
    load_model,
    save_model,
    train_logistic_regression,
    train_random_forest,
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 300
    target = rng.binomial(1, 0.3, n)
    X = pd.DataFrame({
        "f1": np.where(target == 1, rng.normal(1, 1, n), rng.normal(0, 1, n)),
        "f2": rng.normal(0, 1, n),
    })
    y = pd.Series(target)
    return X, y


def test_train_logistic_regression_returns_fitted_model(synthetic_data):
    X, y = synthetic_data
    model = train_logistic_regression(X, y, {"max_iter": 500, "class_weight": "balanced"})
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_train_random_forest_returns_fitted_model(synthetic_data):
    X, y = synthetic_data
    model = train_random_forest(X, y, {"n_estimators": 20, "max_depth": 4, "class_weight": "balanced", "n_jobs": -1})
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_cross_validate_model_returns_scores_in_valid_range(synthetic_data):
    X, y = synthetic_data
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=500, class_weight="balanced")
    result = cross_validate_model(model, X, y, cv=3)
    assert 0.0 <= result["mean"] <= 1.0
    assert len(result["cv_scores"]) == 3


def test_save_and_load_model_roundtrip(tmp_path, synthetic_data):
    X, y = synthetic_data
    model = train_logistic_regression(X, y, {"max_iter": 500, "class_weight": "balanced"})
    path = tmp_path / "model.joblib"
    save_model(model, str(path))
    assert path.exists()

    reloaded = load_model(str(path))
    assert (reloaded.predict(X) == model.predict(X)).all()
