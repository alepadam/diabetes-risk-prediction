"""Preprocessing: cleaning, train/val/test splitting, and imbalance handling.

Kept separate from feature engineering (features.py) — this module is about
getting the data into a modeling-ready shape, not creating new signal.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop exact duplicates, enforce dtypes.

    This dataset ships with no missing values (confirmed by UCI's data card),
    but duplicate survey rows are common in BRFSS-derived data and worth
    removing before splitting.
    """
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} duplicate rows ({dropped / before:.2%})")

    int_cols = [c for c in df.columns if c != "BMI"]
    df[int_cols] = df[int_cols].astype(int)
    df["BMI"] = df["BMI"].astype(float)
    return df


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train/val/test split so class balance is preserved in every split.

    val_size is expressed as a fraction of the *original* dataset, matching
    config.yaml's documented meaning, not a fraction of the train split.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )

    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=relative_val_size, stratify=y_train_full, random_state=random_state,
    )

    logger.info(
        f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    """Oversample the minority class in the training set only.

    Only ever call this on the training split — applying SMOTE before
    splitting, or to val/test data, leaks synthetic signal across the split
    and inflates evaluation metrics.
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"SMOTE applied — training size {len(X_train)} -> {len(X_res)}")
    return X_res, y_res
