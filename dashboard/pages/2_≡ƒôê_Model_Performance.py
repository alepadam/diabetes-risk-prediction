"""Model Performance page — shows how the best model was selected and
evaluated, pulling directly from notebook 04/05's saved artifacts rather
than recomputing anything.
"""
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_DIR.parent
for _p in (DASHBOARD_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from utils.model_loader import artifacts_available, get_figure_path, load_metadata
from utils.styling import apply_custom_theme

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
apply_custom_theme(st)

st.title("📈 Model Performance")
st.markdown(
    "How the final model was selected and how well it actually performs — "
    "pulled directly from the training pipeline's saved results."
)

status = artifacts_available()
if not status["metadata"]:
    st.warning(
        "⚠️ `models/model_metadata.json` not found. Run "
        "`notebooks/04_modeling.ipynb` first."
    )
    st.stop()

metadata = load_metadata()

# --- Model selection summary --------------------------------------------
st.markdown("### Selected Model")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", metadata.get("best_model_name", "—"))
with col2:
    cv_mean = metadata.get("cv_mean_roc_auc")
    cv_std = metadata.get("cv_std_roc_auc")
    if cv_mean is not None:
        st.metric("5-Fold CV ROC-AUC", f"{cv_mean:.3f}", delta=f"±{cv_std:.3f}", delta_color="off")
with col3:
    val_metrics = metadata.get("validation_metrics", {})
    st.metric("Validation ROC-AUC", f"{val_metrics.get('roc_auc', 0):.3f}")
with col4:
    test_metrics = metadata.get("test_metrics", {})
    st.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")

st.caption(
    "Model selection used the **validation** set; the **test** set was "
    "evaluated exactly once, after selection — so the test numbers above "
    "reflect genuine out-of-sample performance, not a metric the model "
    "was tuned against."
)

st.markdown("---")

# --- Detailed metrics table ----------------------------------------------
st.markdown("### Test Set Metrics")
if test_metrics:
    metrics_display = {
        "Accuracy": test_metrics.get("accuracy"),
        "Precision": test_metrics.get("precision"),
        "Recall": test_metrics.get("recall"),
        "F1 Score": test_metrics.get("f1"),
        "ROC-AUC": test_metrics.get("roc_auc"),
    }
    cols = st.columns(len(metrics_display))
    for col, (name, value) in zip(cols, metrics_display.items()):
        with col:
            st.metric(name, f"{value:.3f}" if value is not None else "—")

    st.info(
        "💡 **Why recall matters here:** with an imbalanced target (~86% "
        "no-diabetes / ~14% diabetes), a false negative — missing someone "
        "who actually has diabetes — is generally more costly than a false "
        "positive. Recall on the diabetic class is worth weighing alongside "
        "ROC-AUC, not just accuracy."
    )

st.markdown("---")

# --- Visualizations from the notebooks ------------------------------------
st.markdown("### Confusion Matrix (Test Set)")
cm_path = get_figure_path("14_confusion_matrix_test_final.png")
if cm_path:
    st.image(str(cm_path), width=500)
else:
    st.info("Confusion matrix figure not found — run notebook 04 to generate it.")

st.markdown("---")

st.markdown("### ROC Curve Comparison (Validation Set)")
roc_path = get_figure_path("12_roc_curves_validation.png")
if roc_path:
    st.image(str(roc_path), width=550)
    st.caption(
        "Compares all three candidate models (Logistic Regression, Random "
        "Forest, XGBoost) on the validation set — this comparison is what "
        "determined the winning model above."
    )
else:
    st.info("ROC curve figure not found — run notebook 04 to generate it.")

st.markdown("---")

# --- SHAP global importance ------------------------------------------------
st.markdown("### What Drives the Model's Predictions Globally")
shap_col1, shap_col2 = st.columns(2)
with shap_col1:
    shap_bar_path = get_figure_path("16_shap_bar_importance.png")
    if shap_bar_path:
        st.image(str(shap_bar_path), caption="Mean |SHAP value| per feature")
    else:
        st.info("Run notebook 05 to generate SHAP importance plots.")
with shap_col2:
    shap_summary_path = get_figure_path("15_shap_summary_beeswarm.png")
    if shap_summary_path:
        st.image(str(shap_summary_path), caption="Direction + magnitude of each feature's effect")
    else:
        st.info("Run notebook 05 to generate SHAP summary plots.")
