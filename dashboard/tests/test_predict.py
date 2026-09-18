"""Unit tests for dashboard/utils/predict.py.

Uses a small real trained model + fitted preprocessor (built the same way
notebooks 03/04 do) rather than mocks, since the risk here is specifically
about whether the dashboard's prediction pipeline stays in sync with the
real training pipeline — a mock could hide exactly that kind of drift.
"""
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_DIR.parent
for _p in (DASHBOARD_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
import pytest

from utils.predict import (
    RAW_FEATURE_COLUMNS,
    build_raw_input_df,
    get_clean_feature_names,
    get_shap_model_type,
    predict_risk,
    prepare_features_for_model,
)


@pytest.fixture(scope="module")
def sample_raw_inputs() -> dict:
    return {
        "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 34.5, "Smoker": 0,
        "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 3, "MentHlth": 2, "PhysHlth": 5, "DiffWalk": 0, "Sex": 1,
        "Age": 9, "Education": 5, "Income": 6,
    }


@pytest.fixture(scope="module")
def trained_model_and_preprocessor():
    """Build a small real model + fitted preprocessor via the actual
    training pipeline (src.preprocessing, src.features, src.train) so
    these tests catch drift between the dashboard and the real pipeline.
    """
    from src.features import engineer_features
    from src.preprocessing import build_preprocessing_pipeline, clean_data, split_data
    from src.train import train_random_forest

    rng = np.random.default_rng(42)
    n = 500
    target = rng.binomial(1, 0.15, n)
    raw_df = pd.DataFrame({
        "Diabetes_binary": target,
        "HighBP": rng.binomial(1, 0.4, n), "HighChol": rng.binomial(1, 0.4, n),
        "CholCheck": rng.binomial(1, 0.9, n), "Smoker": rng.binomial(1, 0.4, n),
        "Stroke": rng.binomial(1, 0.05, n), "HeartDiseaseorAttack": rng.binomial(1, 0.1, n),
        "PhysActivity": rng.binomial(1, 0.7, n), "Fruits": rng.binomial(1, 0.6, n),
        "Veggies": rng.binomial(1, 0.8, n), "HvyAlcoholConsump": rng.binomial(1, 0.05, n),
        "AnyHealthcare": rng.binomial(1, 0.95, n), "NoDocbcCost": rng.binomial(1, 0.1, n),
        "DiffWalk": rng.binomial(1, 0.2, n), "Sex": rng.binomial(1, 0.5, n),
        "GenHlth": rng.integers(1, 6, n), "Age": rng.integers(1, 14, n),
        "Education": rng.integers(1, 7, n), "Income": rng.integers(1, 9, n),
        "BMI": rng.normal(28, 6, n).clip(12, 60),
        "MentHlth": rng.poisson(2, n).clip(0, 30), "PhysHlth": rng.poisson(3, n).clip(0, 30),
    })

    df_clean = clean_data(raw_df)
    df_features = engineer_features(df_clean, encode=True)
    X_train, _X_val, _X_test, y_train, _y_val, _y_test = split_data(
        df_features, target_column="Diabetes_binary", test_size=0.2, val_size=0.1, random_state=42,
    )
    numeric_features = ["BMI", "MentHlth", "PhysHlth", "UnwellDays", "RiskFactorCount"]
    passthrough_features = [c for c in X_train.columns if c not in numeric_features]
    preprocessor = build_preprocessing_pipeline(numeric_features, passthrough_features)
    X_train_t = preprocessor.fit_transform(X_train)
    model = train_random_forest(
        X_train_t, y_train,
        {"n_estimators": 30, "max_depth": 5, "class_weight": "balanced", "n_jobs": -1},
    )
    return model, preprocessor


def test_build_raw_input_df_produces_single_row(sample_raw_inputs):
    df = build_raw_input_df(sample_raw_inputs)
    assert len(df) == 1
    assert list(df.columns) == RAW_FEATURE_COLUMNS


def test_build_raw_input_df_raises_on_missing_field(sample_raw_inputs):
    incomplete = dict(sample_raw_inputs)
    del incomplete["BMI"]
    with pytest.raises(ValueError, match="Missing required input"):
        build_raw_input_df(incomplete)


def test_prepare_features_includes_all_bmi_category_dummies(sample_raw_inputs):
    # Regression test for a real risk that was checked before building this
    # dashboard: a single-row input only "observes" one BMI category, so a
    # naive one-hot encode could produce just 1 dummy column instead of all
    # 4 the trained preprocessor expects.
    raw_df = build_raw_input_df(sample_raw_inputs)
    features_df = prepare_features_for_model(raw_df)
    expected_dummies = {
        "BMI_category_underweight", "BMI_category_normal",
        "BMI_category_overweight", "BMI_category_obese",
    }
    assert expected_dummies.issubset(set(features_df.columns))


def test_predict_risk_returns_expected_shape_and_ranges(sample_raw_inputs, trained_model_and_preprocessor):
    model, preprocessor = trained_model_and_preprocessor
    result = predict_risk(sample_raw_inputs, model, preprocessor)

    assert 0.0 <= result["probability"] <= 1.0
    assert result["predicted_class"] in (0, 1)
    assert result["risk_tier"] in ("Low", "Medium", "High")
    assert len(result["features_transformed_df"]) == 1


def test_predict_risk_no_missing_or_extra_columns_vs_training(sample_raw_inputs, trained_model_and_preprocessor):
    # The core train/serve-skew check: the engineered features produced for
    # a single dashboard prediction must contain exactly the columns the
    # preprocessor was fit on — no more, no less.
    model, preprocessor = trained_model_and_preprocessor
    result = predict_risk(sample_raw_inputs, model, preprocessor)
    expected_columns = set(get_clean_feature_names(preprocessor))
    actual_columns = set(result["features_transformed_df"].columns)
    assert actual_columns == expected_columns


def test_get_clean_feature_names_strips_prefixes(trained_model_and_preprocessor):
    _model, preprocessor = trained_model_and_preprocessor
    clean_names = get_clean_feature_names(preprocessor)
    assert all("__" not in name for name in clean_names)
    assert "BMI" in clean_names


def test_get_shap_model_type_tree_models():
    assert get_shap_model_type("Random Forest") == "tree"
    assert get_shap_model_type("XGBoost") == "tree"


def test_get_shap_model_type_linear_model():
    # This is the exact case that was broken before: Logistic Regression
    # must map to 'linear', not fall through to 'tree'.
    assert get_shap_model_type("Logistic Regression") == "linear"


def test_get_shap_model_type_unknown_defaults_to_linear():
    assert get_shap_model_type("Some Unknown Model") == "linear"
