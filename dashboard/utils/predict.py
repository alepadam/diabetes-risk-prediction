"""Turns raw form input into a prediction, reusing the EXACT SAME
feature-engineering and scaling pipeline the model was trained with
(src.preprocessing.clean_data, src.features.engineer_features, and the
fitted preprocessor from notebook 03) — not a reimplementation of it.

This avoids train/serve skew: if the training pipeline ever changes, this
dashboard automatically stays in sync since it imports the same functions
rather than duplicating their logic.
"""
from __future__ import annotations

import pandas as pd

# Raw feature columns the model expects, in the exact coding the CDC
# dataset uses (UCI id 891) — mirrors src.data_loader.EXPECTED_COLUMNS
# minus the target column.
RAW_FEATURE_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]


def build_raw_input_df(raw_inputs: dict) -> pd.DataFrame:
    """Build a single-row DataFrame from a dict of raw coded values.

    raw_inputs must contain every key in RAW_FEATURE_COLUMNS, already coded
    to match the dataset (0/1 for binary features, the correct bucket code
    for Age/GenHlth/Education/Income, a float for BMI, 0-30 ints for
    MentHlth/PhysHlth). Raises a clear error if anything is missing rather
    than silently producing a malformed row.
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in raw_inputs]
    if missing:
        raise ValueError(f"Missing required input(s): {missing}")
    row = {col: raw_inputs[col] for col in RAW_FEATURE_COLUMNS}
    return pd.DataFrame([row])


def prepare_features_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Run raw input through the same cleaning + feature engineering used
    in notebook 03, producing a frame with every column the fitted
    preprocessor expects (including all 4 one-hot BMI_category columns —
    verified safe for single-row input since pd.cut's Categorical dtype
    always carries the full category list regardless of what's observed).
    """
    from src.features import engineer_features
    from src.preprocessing import clean_data

    df_clean = clean_data(raw_df)
    df_features = engineer_features(df_clean, encode=True)
    return df_features


def get_clean_feature_names(preprocessor) -> list[str]:
    """ColumnTransformer.get_feature_names_out() prefixes every name with
    its transformer's name (e.g. 'scale__BMI', 'passthrough__HighBP'),
    which is accurate but ugly in a user-facing SHAP plot. Strip the
    prefix back off for display purposes only.
    """
    raw_names = preprocessor.get_feature_names_out()
    return [name.split("__", 1)[-1] for name in raw_names]


def get_shap_model_type(best_model_name: str) -> str:
    """Map the winning model's name to the correct SHAP explainer type.

    Mirrors the same tree_model_names check used in
    notebooks/05_interpretability_shap.ipynb — kept as a shared function so
    the dashboard and the notebook can never drift out of sync on which
    models are tree-based.
    """
    tree_model_names = {"Random Forest", "XGBoost"}
    return "tree" if best_model_name in tree_model_names else "linear"


def predict_risk(raw_inputs: dict, model, preprocessor) -> dict:
    """End-to-end: raw form input -> engineered features -> scaled features
    -> prediction. Returns probability, predicted class, and risk tier.
    """
    from src.segmentation import bucket_risk_scores

    raw_df = build_raw_input_df(raw_inputs)
    features_df = prepare_features_for_model(raw_df)

    # ColumnTransformer selects columns by name internally, so passing
    # extra/differently-ordered columns is safe as long as every column it
    # was fit on is present by name — which prepare_features_for_model
    # guarantees by construction (same function used in training).
    features_transformed = preprocessor.transform(features_df)

    probability = float(model.predict_proba(features_transformed)[0, 1])
    predicted_class = int(model.predict(features_transformed)[0])
    risk_tier = bucket_risk_scores([probability]).iloc[0]

    clean_names = get_clean_feature_names(preprocessor)
    features_transformed_df = pd.DataFrame(features_transformed, columns=clean_names)

    return {
        "probability": probability,
        "predicted_class": predicted_class,
        "risk_tier": str(risk_tier),
        "features_df": features_df,  # raw engineered features, before scaling
        "features_transformed_df": features_transformed_df,  # scaled, for SHAP
    }
