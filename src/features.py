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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline in one call."""
    df = add_bmi_category(df)
    df = add_composite_health_score(df)
    df = add_risk_factor_count(df)
    return df
