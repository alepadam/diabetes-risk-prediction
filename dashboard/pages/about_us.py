"""About Us page — how the model performs and what the data reveals,
combined into one page with two tabs so related "how good/what does it
mean" content lives together instead of being split across separate
top-level pages.
"""
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_DIR.parent
for _p in (DASHBOARD_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st
from utils.model_loader import (
    artifacts_available,
    get_figure_path,
    load_high_risk_summary,
    load_metadata,
    load_statistical_results,
)
from utils.styling import render_sidebar

render_sidebar(st)

st.title("📊 About Us")
st.markdown(
    "How the model was built, how well it performs, and what the "
    "underlying data reveals about diabetes risk factors."
)

tab_performance, tab_insights = st.tabs(["📈 Model Performance", "🔎 Data Insights"])

# =====================================================================
# TAB 1: Model Performance
# =====================================================================
with tab_performance:
    status = artifacts_available()
    if not status["metadata"]:
        st.warning(
            "⚠️ `models/model_metadata.json` not found. Run "
            "`notebooks/04_modeling.ipynb` first."
        )
    else:
        metadata = load_metadata()

        st.markdown("#### Selected Model")
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
        st.markdown("#### Test Set Metrics")
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
        st.markdown("#### Confusion Matrix (Test Set)")
        cm_path = get_figure_path("14_confusion_matrix_test_final.png")
        if cm_path:
            st.image(str(cm_path), width=500)
        else:
            st.info("Confusion matrix figure not found — run notebook 04 to generate it.")

        st.markdown("---")
        st.markdown("#### ROC Curve Comparison (Validation Set)")
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
        st.markdown("#### What Drives the Model's Predictions Globally")
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

# =====================================================================
# TAB 2: Data Insights
# =====================================================================
with tab_insights:
    st.markdown("#### Target Distribution & Key Patterns")
    col1, col2 = st.columns(2)
    with col1:
        target_dist_path = get_figure_path("01_target_distribution.png")
        if target_dist_path:
            st.image(str(target_dist_path), caption="Class balance: diabetes/prediabetes vs. no diabetes")
        else:
            st.info("Run notebook 01 to generate this figure.")
    with col2:
        corr_heatmap_path = get_figure_path("08_correlation_heatmap.png")
        if corr_heatmap_path:
            st.image(str(corr_heatmap_path), caption="Full feature correlation heatmap")
        else:
            st.info("Run notebook 01 to generate this figure.")

    st.markdown("---")
    st.markdown("#### Ranked Feature-Target Associations")
    st.markdown(
        "Using the statistically appropriate measure per feature type "
        "(point-biserial for binary/ordinal, correlation ratio for continuous) "
        "rather than a blanket Pearson correlation, with Benjamini-Hochberg "
        "FDR correction applied across all tests."
    )

    stats_results = load_statistical_results()
    if stats_results is not None:
        ranking_col, table_col = st.columns([1, 1])
        with ranking_col:
            assoc_fig_path = get_figure_path("10_association_ranking.png")
            if assoc_fig_path:
                st.image(str(assoc_fig_path))
        with table_col:
            display_df = stats_results[
                ["feature", "effect_size", "effect_size_label", "significant_after_correction"]
            ].copy()
            display_df.columns = ["Feature", "Effect Size", "Magnitude", "Significant (FDR-corrected)"]
            display_df["Effect Size"] = display_df["Effect Size"].round(3)
            st.dataframe(display_df, width="stretch", hide_index=True, height=400)

        st.caption(
            "Effect size — not p-value — drives this ranking. With n≈253,680, "
            "even trivial differences reach statistical significance, so effect "
            "size is what actually separates a meaningful finding from noise."
        )
    else:
        st.warning("⚠️ Run `notebooks/02_statistical_analysis.ipynb` to generate this data.")

    st.markdown("---")
    st.markdown("#### Health Equity: Who Gets Flagged High-Risk?")
    st.markdown(
        "The sharpest question this project asks: does predicted risk "
        "concentrate in populations that *also* face barriers to acting on "
        "that risk (lower income, no healthcare coverage, skipping care due "
        "to cost)?"
    )

    high_risk_summary = load_high_risk_summary()
    if high_risk_summary is not None:
        overrep_col, figs_col = st.columns([1, 1])

        with overrep_col:
            st.markdown("**Most overrepresented groups in the High-risk tier**")
            top_overrep = high_risk_summary.sort_values("overrepresentation", ascending=False).head(8).copy()
            top_overrep = top_overrep[["subgroup_column", "level", "pct_in_high_risk_group", "pct_in_overall_population", "overrepresentation"]]
            top_overrep.columns = ["Subgroup", "Level", "% of High-Risk Group", "% of Overall Population", "Overrepresentation (pts)"]
            for col in ["% of High-Risk Group", "% of Overall Population", "Overrepresentation (pts)"]:
                top_overrep[col] = top_overrep[col].round(1)
            st.dataframe(top_overrep, width="stretch", hide_index=True, height=320)
            st.caption(
                "A positive 'overrepresentation' means that group makes up a "
                "larger share of the High-risk tier than it does of the overall "
                "test population."
            )

        with figs_col:
            income_fig = get_figure_path("20_risk_by_income.png")
            nodoc_fig = get_figure_path("22_risk_by_nodoc_cost.png")
            if income_fig:
                st.image(str(income_fig), caption="Risk tier by income level")
            if nodoc_fig:
                st.image(str(nodoc_fig), caption="Risk tier by 'skipped doctor due to cost'")
    else:
        st.warning("⚠️ Run `notebooks/06_risk_segmentation.ipynb` to generate this data.")

    st.markdown("---")
    with st.expander("More exploratory figures (univariate & bivariate breakdowns)"):
        fig_names = [
            ("02_univariate_binary.png", "Binary feature distributions"),
            ("03_univariate_ordinal.png", "Ordinal feature distributions"),
            ("04_univariate_continuous.png", "Continuous feature distributions"),
            ("05_bivariate_binary.png", "Binary features vs. diabetes status"),
            ("06_bivariate_ordinal.png", "Ordinal features vs. diabetes status"),
            ("11_cramers_v_heatmap.png", "Cramer's V — feature redundancy check"),
        ]
        for filename, caption in fig_names:
            path = get_figure_path(filename)
            if path:
                st.image(str(path), caption=caption)
