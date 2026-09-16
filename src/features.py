"""Feature engineering for the diabetes risk dataset.

Each function takes and returns a DataFrame so they can be chained. Keep
new engineered features here (not inline in notebooks) so they're testable
and reusable across the modeling and interpretability notebooks.
"""
from __future__ import annotations

import pandas as pd


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket BMI into standard clinical categories — often more interpretable
    to stakeholders than raw BMI, and can capture non-linear risk jumps.
    """
    df = df.copy()
    bins = [0, 18.5, 25, 30, 100]
    labels = ["underweight", "normal", "overweight", "obese"]
    df["BMI_category"] = pd.cut(df["BMI"], bins=bins, labels=labels, right=False)
    return df


def add_composite_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine MentHlth + PhysHlth (both 'days unwell in past 30') into one
    composite score of overall unwellness, capped at 30.
    """
    df = df.copy()
    df["UnwellDays"] = (df["MentHlth"] + df["PhysHlth"]).clip(upper=30)
    return df


def add_risk_factor_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many known binary risk factors a respondent has.

    A simple, interpretable aggregate that's useful both as a model feature
    and directly for the risk-segmentation analysis in notebook 06.
    """
    df = df.copy()
    risk_cols = ["HighBP", "HighChol", "Smoker", "Stroke",
                 "HeartDiseaseorAttack", "HvyAlcoholConsump"]
    df["RiskFactorCount"] = df[risk_cols].sum(axis=1)
    return df


def encode_bmi_category(df: pd.DataFrame, drop_original: bool = True) -> pd.DataFrame:
    """One-hot encode BMI_category rather than ordinal-encode it.

    Chosen over ordinal encoding because diabetes risk doesn't necessarily
    increase monotonically with BMI category in a way a single ordinal
    number captures well (e.g. underweight can itself carry elevated risk
    in some populations) — one-hot lets models learn each category's
    effect independently instead of assuming a linear order.
    """
    df = df.copy()
    dummies = pd.get_dummies(df["BMI_category"], prefix="BMI_category", dtype=int)
    df = pd.concat([df, dummies], axis=1)
    if drop_original:
        df = df.drop(columns=["BMI_category"])
    return df


def engineer_features(df: pd.DataFrame, encode: bool = True) -> pd.DataFrame:
    """Apply the full feature engineering pipeline in one call.

    encode=True also one-hot encodes BMI_category, producing a fully
    numeric frame ready for the scaling step in preprocessing. Set
    encode=False if you want to inspect BMI_category as a readable label
    first (e.g. for the risk-segmentation notebook).
    """
    df = add_bmi_category(df)
    df = add_composite_health_score(df)
    df = add_risk_factor_count(df)
    if encode:
        df = encode_bmi_category(df)
    return df
