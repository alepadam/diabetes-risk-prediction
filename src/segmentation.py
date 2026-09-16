"""Risk segmentation: turn model probabilities into actionable risk tiers,
and check how those tiers distribute across socioeconomic/access subgroups.

This is what turns a classifier into something with a health-equity angle —
per the project's README goal — rather than stopping at "the model predicts X".
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def bucket_risk_scores(probabilities, low_threshold: float = 0.33, high_threshold: float = 0.66) -> pd.Series:
    """Bucket predicted probabilities into Low/Medium/High risk tiers.

    Thresholds are on the predicted probability itself, not the class label
    — two people both classified as 'diabetic' by the model can still have
    very different risk scores (0.51 vs 0.98), and that distinction matters
    for a screening/triage use case.
    """
    import numpy as np

    probabilities = np.asarray(probabilities)
    labels = pd.cut(
        probabilities,
        bins=[-0.01, low_threshold, high_threshold, 1.0],
        labels=["Low", "Medium", "High"],
    )
    return pd.Series(labels, name="risk_tier")


def crosstab_risk_by_subgroup(risk_tier: pd.Series, subgroup: pd.Series, normalize: str = "index") -> pd.DataFrame:
    """Cross-tabulate risk tier against a subgroup variable (e.g. Income,
    Education, NoDocbcCost), normalized to percentages within each subgroup
    so subgroup sizes don't distort the comparison.
    """
    return pd.crosstab(subgroup, risk_tier, normalize=normalize) * 100


def plot_risk_by_subgroup(crosstab_df: pd.DataFrame, title: str, save_path: str | None = None):
    """Stacked bar chart of risk tier composition across a subgroup's levels."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Low": "#4c72b0", "Medium": "#dd8452", "High": "#c44e52"}
    plot_colors = [colors.get(c, "#999999") for c in crosstab_df.columns]
    crosstab_df.plot(kind="bar", stacked=True, ax=ax, color=plot_colors)
    ax.set_ylabel("% within subgroup")
    ax.set_title(title)
    ax.legend(title="Risk Tier", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def summarize_high_risk_group(df: pd.DataFrame, risk_tier: pd.Series, subgroup_cols: list[str]) -> pd.DataFrame:
    """For the High risk tier specifically, summarize its composition across
    a list of subgroup columns — a compact table for the README/report
    rather than one plot per subgroup.
    """
    high_risk_mask = risk_tier == "High"
    rows = []
    for col in subgroup_cols:
        high_risk_dist = df.loc[high_risk_mask, col].value_counts(normalize=True) * 100
        overall_dist = df[col].value_counts(normalize=True) * 100
        for level in overall_dist.index:
            rows.append({
                "subgroup_column": col,
                "level": level,
                "pct_in_high_risk_group": high_risk_dist.get(level, 0.0),
                "pct_in_overall_population": overall_dist.get(level, 0.0),
            })
    return pd.DataFrame(rows)
