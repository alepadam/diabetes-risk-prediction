"""Risk Calculator page — collects lifestyle/health inputs and returns a
personalized risk prediction with a SHAP-based explanation.
"""
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_DIR.parent
for _p in (DASHBOARD_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt
import streamlit as st

from utils.mappings import (
    EDUCATION_OPTIONS,
    GENHLTH_OPTIONS,
    INCOME_OPTIONS,
    age_to_bucket,
    calculate_bmi,
)
from utils.model_loader import (
    artifacts_available,
    load_metadata,
    load_model,
    load_preprocessor,
)
from utils.predict import get_shap_model_type, predict_risk
from utils.styling import render_sidebar, risk_badge_html

render_sidebar(st)

st.title("🎯 Risk Calculator")
st.markdown(
    """
    <div class="disclaimer-box">
    <strong>⚠️ Not a medical diagnostic tool.</strong> This is a statistical
    estimate based on population survey data, for educational purposes only.
    It cannot replace a conversation with a doctor.
    </div>
    """,
    unsafe_allow_html=True,
)

status = artifacts_available()
if not all(status.values()):
    st.error(
        "Model artifacts are missing. Run notebooks 03 (`feature_engineering`) "
        "and 04 (`modeling`) first to generate the required files in `models/`."
    )
    st.stop()

st.markdown("### Your Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    real_age = st.number_input("Age (years)", min_value=18, max_value=99, value=45)
    sex_label = st.radio("Sex", ["Female", "Male"], horizontal=True)
    education_code = st.selectbox(
        "Education level", options=list(EDUCATION_OPTIONS.keys()),
        format_func=lambda k: EDUCATION_OPTIONS[k], index=4,
    )
    income_code = st.selectbox(
        "Annual household income", options=list(INCOME_OPTIONS.keys()),
        format_func=lambda k: INCOME_OPTIONS[k], index=5,
    )

with col2:
    st.markdown("**Body & General Health**")
    weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.5)
    height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=170.0, step=0.5)
    bmi = calculate_bmi(weight_kg, height_cm)
    st.caption(f"Calculated BMI: **{bmi:.1f}**")
    genhlth_code = st.select_slider(
        "General health (self-rated)", options=list(GENHLTH_OPTIONS.keys()),
        value=3, format_func=lambda k: GENHLTH_OPTIONS[k],
    )
    ment_hlth = st.slider("Days of poor mental health (past 30 days)", 0, 30, 2)
    phys_hlth = st.slider("Days of poor physical health (past 30 days)", 0, 30, 2)

with col3:
    st.markdown("**Medical History**")
    high_bp = st.checkbox("High blood pressure")
    high_chol = st.checkbox("High cholesterol")
    chol_check = st.checkbox("Cholesterol checked in past 5 years", value=True)
    stroke = st.checkbox("History of stroke")
    heart_disease = st.checkbox("Heart disease or heart attack")
    diff_walk = st.checkbox("Serious difficulty walking/climbing stairs")

st.markdown("**Lifestyle & Healthcare Access**")
col4, col5, col6 = st.columns(3)
with col4:
    smoker = st.checkbox("Smoked at least 100 cigarettes in lifetime")
    phys_activity = st.checkbox("Physical activity in past 30 days (not job-related)", value=True)
with col5:
    fruits = st.checkbox("Consumes fruit 1+ times/day", value=True)
    veggies = st.checkbox("Consumes vegetables 1+ times/day", value=True)
    hvy_alcohol = st.checkbox("Heavy alcohol consumption")
with col6:
    any_healthcare = st.checkbox("Has any healthcare coverage", value=True)
    no_doc_cost = st.checkbox("Skipped seeing a doctor due to cost (past 12 months)")

st.markdown("---")

if st.button("Calculate My Risk", type="primary", width="stretch"):
    raw_inputs = {
        "HighBP": int(high_bp),
        "HighChol": int(high_chol),
        "CholCheck": int(chol_check),
        "BMI": bmi,
        "Smoker": int(smoker),
        "Stroke": int(stroke),
        "HeartDiseaseorAttack": int(heart_disease),
        "PhysActivity": int(phys_activity),
        "Fruits": int(fruits),
        "Veggies": int(veggies),
        "HvyAlcoholConsump": int(hvy_alcohol),
        "AnyHealthcare": int(any_healthcare),
        "NoDocbcCost": int(no_doc_cost),
        "GenHlth": genhlth_code,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
        "DiffWalk": int(diff_walk),
        "Sex": 1 if sex_label == "Male" else 0,
        "Age": age_to_bucket(real_age),
        "Education": education_code,
        "Income": income_code,
    }

    model = load_model()
    preprocessor = load_preprocessor()

    with st.spinner("Calculating..."):
        result = predict_risk(raw_inputs, model, preprocessor)

    st.markdown("### Your Result")
    st.markdown(risk_badge_html(result["risk_tier"], result["probability"]), unsafe_allow_html=True)
    st.markdown("")

    result_col1, result_col2 = st.columns([1, 2])
    with result_col1:
        st.metric("Predicted probability", f"{result['probability'] * 100:.1f}%")
        st.metric("Risk tier", result["risk_tier"])
        st.caption(
            "Risk tiers: Low (<33%), Medium (33-66%), High (>66%) predicted probability."
        )

    with result_col2:
        st.markdown("**What's driving this prediction?**")
        try:
            from src.interpret import compute_shap_values, plot_shap_waterfall

            metadata = load_metadata()
            shap_model_type = get_shap_model_type(metadata.get("best_model_name", ""))
            shap_values = compute_shap_values(model, result["features_transformed_df"], model_type=shap_model_type)
            fig = plot_shap_waterfall(shap_values, index=0)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Red bars push your risk score up, blue bars push it down. "
                "This explains this specific prediction, not the model in general."
            )
        except Exception as e:  # noqa: BLE001 — SHAP can fail in several model-specific ways; degrade gracefully regardless of cause
            st.info(
                "Individual explanation unavailable for this model type. "
                f"({e})"
            )
