"""Model training functions for the three baseline models.

Each function takes training data + the relevant config block and returns a
fitted model, so notebooks stay to a couple of lines per model.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", "balanced"),
    )
    model.fit(X_train, y_train)
    logger.info("Logistic Regression trained.")
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any]):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", 12),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=params.get("n_jobs", -1),
        random_state=42,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest trained.")
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, Any],
):
    from xgboost import XGBClassifier

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        eval_metric=params.get("eval_metric", "auc"),
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(X_train, y_train)
    logger.info(f"XGBoost trained (scale_pos_weight={scale_pos_weight:.2f}).")
    return model


def cross_validate_model(model, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5, random_state: int = 42) -> dict:
    """Stratified k-fold CV on the training set, scored on ROC-AUC.

    Run this before trusting a single train/validation split — with an
    imbalanced target, a single split can look better or worse than the
    model's true stability by chance. Uses an unfitted clone of the model
    so nothing here leaks fitted state from a prior .fit() call.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(clone(model), X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1)
    logger.info(f"{cv}-fold CV ROC-AUC: mean={scores.mean():.4f}, std={scores.std():.4f}")
    return {"cv_scores": scores, "mean": scores.mean(), "std": scores.std()}


def save_model(model, path: str) -> None:
    from pathlib import Path

    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    import joblib

    return joblib.load(path)
