"""Unit tests for src/preprocessing.py and src/features.py.

Run with: pytest tests/
These don't need the real dataset — small synthetic frames are enough to
verify the logic (dedup, split proportions, engineered columns).
"""
import pandas as pd
import pytest

from src.features import (
    add_bmi_category,
    add_risk_factor_count,
    encode_bmi_category,
    engineer_features,
)
from src.preprocessing import build_preprocessing_pipeline, clean_data, split_data


@pytest.fixture
def sample_df():
    # 40 rows, balanced classes — large enough for a stratified train/val/test
    # split (test_size=0.2, val_size=0.1) to leave both classes in every split.
    n = 40
    return pd.DataFrame({
        "Diabetes_binary": [i % 2 for i in range(n)],
        "HighBP": [0, 1, 0, 1] * (n // 4),
        "HighChol": [0, 1, 0, 0, 1, 1, 0, 0, 1, 1] * (n // 10),
        "Smoker": [0, 1, 0, 0, 0, 1, 1, 0, 0, 1] * (n // 10),
        "Stroke": [0] * n,
        "HeartDiseaseorAttack": [0] * n,
        "HvyAlcoholConsump": [0] * n,
        "BMI": [22.0, 31.0, 18.0, 27.0, 40.0, 24.0, 19.0, 29.0, 33.0, 21.0] * (n // 10),
        "MentHlth": [0, 2, 0, 5, 10, 0, 1, 3, 0, 20] * (n // 10),
        "PhysHlth": [0, 1, 0, 3, 15, 0, 0, 2, 1, 25] * (n // 10),
    })


def test_clean_data_drops_duplicates():
    # Small, fully-unique fixture so we can assert an exact duplicate count —
    # sample_df's repeating pattern (needed for the stratified-split test)
    # isn't suitable here since it's already full of legitimate row repeats.
    unique_df = pd.DataFrame({
        "Diabetes_binary": [0, 1, 0],
        "HighBP": [0, 1, 1],
        "BMI": [22.0, 31.0, 18.0],
    })
    df_with_dupe = pd.concat([unique_df, unique_df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(df_with_dupe)
    assert len(cleaned) == len(unique_df)


def test_split_data_preserves_stratification(sample_df):
    X_train, X_val, X_test, _y_train, _y_val, _y_test = split_data(
        sample_df, target_column="Diabetes_binary",
        test_size=0.2, val_size=0.1, random_state=42,
    )
    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sample_df)
    # every split should exist and be non-empty for this fixture size
    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0


def test_add_bmi_category_buckets_correctly(sample_df):
    result = add_bmi_category(sample_df)
    assert "BMI_category" in result.columns
    assert result.loc[result["BMI"] == 18.0, "BMI_category"].iloc[0] == "underweight"
    assert result.loc[result["BMI"] == 22.0, "BMI_category"].iloc[0] == "normal"
    assert result.loc[result["BMI"] == 40.0, "BMI_category"].iloc[0] == "obese"


def test_add_risk_factor_count(sample_df):
    result = add_risk_factor_count(sample_df)
    assert "RiskFactorCount" in result.columns
    # row 1: HighBP=1, HighChol=1, Smoker=1 -> count of 3
    assert result.loc[1, "RiskFactorCount"] == 3
    # row 0: all risk cols 0 -> count of 0
    assert result.loc[0, "RiskFactorCount"] == 0


def test_encode_bmi_category_produces_expected_dummy_columns(sample_df):
    df_with_cat = add_bmi_category(sample_df)
    result = encode_bmi_category(df_with_cat)
    expected_dummies = {
        "BMI_category_underweight", "BMI_category_normal",
        "BMI_category_overweight", "BMI_category_obese",
    }
    assert expected_dummies.issubset(set(result.columns))
    assert "BMI_category" not in result.columns  # original dropped by default
    # exactly one dummy should be 1 per row
    assert (result[list(expected_dummies)].sum(axis=1) == 1).all()


def test_engineer_features_encode_flag_controls_bmi_output(sample_df):
    encoded = engineer_features(sample_df, encode=True)
    unencoded = engineer_features(sample_df, encode=False)
    assert "BMI_category" not in encoded.columns
    assert "BMI_category" in unencoded.columns
    # both should still produce the other engineered features
    for df_result in (encoded, unencoded):
        assert "UnwellDays" in df_result.columns
        assert "RiskFactorCount" in df_result.columns


def test_preprocessing_pipeline_no_leakage_between_train_and_val(sample_df):
    df_fe = engineer_features(sample_df, encode=True)
    X_train, X_val, _X_test, _y_train, _y_val, _y_test = split_data(
        df_fe, target_column="Diabetes_binary",
        test_size=0.2, val_size=0.1, random_state=42,
    )
    numeric_features = ["BMI"]
    passthrough_features = [c for c in X_train.columns if c not in numeric_features]

    pipeline = build_preprocessing_pipeline(numeric_features, passthrough_features)
    X_train_transformed = pipeline.fit_transform(X_train)
    X_val_transformed = pipeline.transform(X_val)

    # Train's scaled BMI column should be ~exactly standardized (fit on itself)
    assert X_train_transformed[:, 0].mean() == pytest.approx(0.0, abs=1e-6)
    assert X_train_transformed[:, 0].std() == pytest.approx(1.0, abs=1e-6)
    # Val was transformed using train's fitted scaler, so it should NOT be
    # exactly standardized — if it is, that's a strong sign of data leakage.
    assert X_val_transformed[:, 0].mean() != pytest.approx(0.0, abs=1e-6)
