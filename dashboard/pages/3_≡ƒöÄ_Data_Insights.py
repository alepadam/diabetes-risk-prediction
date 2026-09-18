"""Data Insights page — surfaces the statistical findings and the
health-equity risk-segmentation result, rather than leaving them buried in
notebooks nobody opens.
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
    get_figure_path,
    load_high_risk_summary,
    load_statistical_results,
)
from utils.styling import apply_custom_theme

st.set_page_config(page_title="Data Insights", page_icon="🔎", layout="wide")
apply_custom_theme(st)

st.title("🔎 Data Insights")
st.markdown(
    "What the underlying data reveals about diabetes risk factors — from "
    "the exploratory analysis, formal statistical testing, and the "
    "health-equity risk segmentation."
)

st.markdown("---")

# --- Overview figures -----------------------------------------------------
st.markdown("### Target Distribution & Key Patterns")
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

# --- Statistical association ranking ---------------------------------------
st.markdown("### Ranked Feature-Target Associations")
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

# --- Health equity: risk segmentation --------------------------------------
st.markdown("### Health Equity: Who Gets Flagged High-Risk?")
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

# --- Additional EDA figures, collapsed to keep the page scannable ----------
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
