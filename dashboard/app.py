"""Diabetes Risk Prediction Dashboard — Overview page.

Run with: streamlit run app.py (from inside the dashboard/ folder)
"""
import sys
from pathlib import Path

# Make src/ (project root) importable from every page of this app.
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
for _p in (DASHBOARD_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st
from utils.model_loader import artifacts_available, load_metadata
from utils.styling import apply_custom_theme

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)
apply_custom_theme(st)

# --- Sidebar -----------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 Diabetes Risk\nPrediction Dashboard")
    st.markdown("---")
    st.markdown(
        "Built on the **CDC Diabetes Health Indicators** dataset "
        "(BRFSS 2015, 253,680 respondents)."
    )
    st.markdown("Use the pages in the sidebar to navigate:")
    st.markdown(
        "- 🎯 **Risk Calculator** — get a personalized prediction\n"
        "- 📈 **Model Performance** — how the model was evaluated\n"
        "- 🔎 **Data Insights** — what drives diabetes risk in this data"
    )
    st.markdown("---")
    st.caption("Muhammad Aliff Adam bin Sultan · Universiti Malaya")

# --- Main content --------------------------------------------------------
st.title("Diabetes Risk Prediction")
st.markdown(
    "A data science project predicting diabetes/prediabetes risk from "
    "lifestyle and health-survey indicators — with a focus on both "
    "**predictive accuracy** and **explainability**."
)

st.markdown(
    """
    <div class="disclaimer-box">
    <strong>⚠️ Not a medical diagnostic tool.</strong> Predictions are based on
    a statistical model trained on population survey data and are intended for
    educational purposes only. Always consult a qualified healthcare provider
    for medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)

status = artifacts_available()
if not all(status.values()):
    st.warning(
        "⚠️ Some model artifacts are missing. Run notebooks 03 and 04 "
        "first to generate `models/preprocessor.joblib`, "
        "`models/best_model.joblib`, and `models/model_metadata.json`."
    )
else:
    metadata = load_metadata()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", metadata.get("best_model_name", "—"))
    with col2:
        test_metrics = metadata.get("test_metrics", {})
        st.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
    with col3:
        st.metric("Test Recall", f"{test_metrics.get('recall', 0):.3f}")
    with col4:
        st.metric("Test F1", f"{test_metrics.get('f1', 0):.3f}")

st.markdown("---")

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("About This Project")
    st.markdown(
        """
        This dashboard is built on a full data science pipeline:

        1. **Exploratory analysis** — univariate, bivariate, and multivariate
           patterns in the data
        2. **Statistical testing** — formal hypothesis tests with effect
           sizes and multiple-testing correction
        3. **Feature engineering** — BMI categories, composite health
           scores, risk factor counts
        4. **Modeling** — Logistic Regression, Random Forest, and XGBoost,
           compared via cross-validation and a held-out test set
        5. **Interpretability** — SHAP explanations, both global and
           per-individual
        6. **Risk segmentation** — checking whether predicted risk
           concentrates in populations facing healthcare access barriers
        """
    )

with col_right:
    st.subheader("How to Use This Dashboard")
    st.markdown(
        """
        - Head to **Risk Calculator** to enter your own health/lifestyle
          information and get a personalized risk score, along with an
          explanation of what's driving that score.
        - Check **Model Performance** to see exactly how well the model
          does, and how it compares against the alternatives that were tried.
        - Explore **Data Insights** to see what the underlying data reveals
          about diabetes risk factors — including a health-equity finding
          about who tends to be flagged highest-risk.
        """
    )

st.markdown("---")
st.caption(
    "Data source: CDC Diabetes Health Indicators, UCI ML Repository "
    "(dataset ID 891). Colors and styling are custom — this dashboard is "
    "not affiliated with the CDC."
)
