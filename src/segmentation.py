"""Risk segmentation: turn model probabilities into actionable risk tiers,
and check how those tiers distribute across socioeconomic/access subgroups.

This is what turns a classifier into something with a health-equity angle —
per the project's README goal — rather than stopping at "the model predicts X".

IMPORTANT — index-alignment safety:
`bucket_risk_scores()` returns a Series with a fresh 0..n-1 index (since it's
built from a plain numpy probability array, which has no index of its own).
If a caller then combines that result with a DataFrame that has a *different*
index — e.g. `X_test` after a train_test_split, which keeps its original,
non-contiguous row positions — pandas' default index-alignment behavior on
things like `pd.crosstab()` or `df.loc[boolean_series]` will silently
mismatch rows against each other and produce wrong numbers with no error or
warning. (This happened in practice — see the regression tests in
tests/test_segmentation.py for the reproduced scenario.)

To make this impossible regardless of what the caller passes in, every
function below that combines a risk-tier Series with another array-like
converts both to plain numpy arrays via `_to_array()` before combining them,
and explicitly validates the lengths match first. This assumes — and
enforces — pure *positional* correspondence: row i of `risk_tier` must
describe row i of `subgroup`/`df`, which is guaranteed as long as both were
derived from the same underlying data in the same order (e.g. both from
`X_test` in the order it was loaded). Pandas index values are never
consulted for alignment anywhere in this module.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _to_array(series_like) -> np.ndarray:
    """Convert a Series/array-like to a plain numpy array, stripping any
    pandas index so it can never be used for (mis)alignment downstream.
    """
    return np.asarray(series_like)


def _validate_same_length(a, b, name_a: str, name_b: str) -> None:
    """Fail loudly and immediately if two array-likes don't correspond
    row-for-row, instead of letting a silent misalignment happen later.
    """
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} and {name_b} must have the same length (they are "
            f"assumed to correspond row-for-row, in order) — got "
            f"len({name_a})={len(a)} vs len({name_b})={len(b)}."
        )


def bucket_risk_scores(probabilities, low_threshold: float = 0.33, high_threshold: float = 0.66) -> pd.Series:
    """Bucket predicted probabilities into Low/Medium/High risk tiers.

    Thresholds are on the predicted probability itself, not the class label
    — two people both classified as 'diabetic' by the model can still have
    very different risk scores (0.51 vs 0.98), and that distinction matters
    for a screening/triage use case.

    Returns a Series with a fresh default (0..n-1) index — this is expected
    and safe as long as downstream code in this module is used, since every
    other function here works positionally rather than by index alignment.
    Do not rely on this Series' index to match another DataFrame's index.
    """
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

    risk_tier and subgroup are matched purely by position (row i to row i),
    never by their pandas index — see the module docstring for why.
    """
    _validate_same_length(risk_tier, subgroup, "risk_tier", "subgroup")
    risk_arr = _to_array(risk_tier)
    subgroup_arr = _to_array(subgroup)
    subgroup_name = getattr(subgroup, "name", None) or "subgroup"
    risk_name = getattr(risk_tier, "name", None) or "risk_tier"

    subgroup_series = pd.Series(subgroup_arr, name=subgroup_name)
    risk_series = pd.Series(risk_arr, name=risk_name)
    return pd.crosstab(subgroup_series, risk_series, normalize=normalize) * 100


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

    df and risk_tier are matched purely by position (row i to row i), never
    by their pandas index — see the module docstring for why. This uses a
    positional boolean numpy mask rather than `df.loc[boolean_series]`,
    which is what silently breaks when risk_tier's index doesn't match df's.
    """
    _validate_same_length(df, risk_tier, "df", "risk_tier")
    high_risk_mask = _to_array(risk_tier) == "High"

    rows = []
    for col in subgroup_cols:
        col_values = df[col].to_numpy()
        high_risk_values = col_values[high_risk_mask]

        high_risk_dist = pd.Series(high_risk_values).value_counts(normalize=True) * 100
        overall_dist = pd.Series(col_values).value_counts(normalize=True) * 100

        for level in overall_dist.index:
            rows.append({
                "subgroup_column": col,
                "level": level,
                "pct_in_high_risk_group": high_risk_dist.get(level, 0.0),
                "pct_in_overall_population": overall_dist.get(level, 0.0),
            })
    return pd.DataFrame(rows)
