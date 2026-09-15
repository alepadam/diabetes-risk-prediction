"""Data loading and validation for the CDC Diabetes Health Indicators dataset.

Handles fetching from UCI (via ucimlrepo), caching to data/raw/, and basic
schema/quality validation so downstream steps can trust the data shape.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def fetch_and_cache_raw_data(config: dict) -> pd.DataFrame:
    """Fetch the dataset from UCI (if not already cached) and return it as a DataFrame.

    Caches to the raw_path in config so repeated runs don't re-hit the network.
    """
    raw_path = resolve_path(config["data"]["raw_path"])

    if raw_path.exists():
        logger.info(f"Loading cached raw data from {raw_path}")
        return pd.read_csv(raw_path)

    logger.info("No cached file found — fetching from UCI ML Repository...")
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=config["data"]["uci_dataset_id"])
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw data to {raw_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return df


def validate_raw_data(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the raw dataframe. Raises AssertionError on failure.

    Deliberately strict: catching a schema drift here is much cheaper than
    debugging a silently-wrong model three notebooks later.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    assert not missing_cols, f"Missing expected columns: {missing_cols}"

    assert df.shape[0] > 0, "DataFrame is empty"

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"Found {n_missing} missing values — dataset is expected to be clean")

    target = df["Diabetes_binary"]
    assert set(target.unique()).issubset({0, 1}), "Diabetes_binary should be binary 0/1"

    logger.info("Raw data validation passed.")


def load_raw_data(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    """Convenience entry point: load config, fetch data, validate, return it."""
    config = load_config(config_path)
    df = fetch_and_cache_raw_data(config)
    validate_raw_data(df)
    return df
