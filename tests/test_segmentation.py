"""Unit tests for src/segmentation.py."""
import pandas as pd
import pytest

from src.segmentation import (
    bucket_risk_scores,
    crosstab_risk_by_subgroup,
    summarize_high_risk_group,
)


def test_bucket_risk_scores_assigns_correct_tiers():
    probabilities = [0.05, 0.32, 0.34, 0.65, 0.67, 0.99]
    tiers = bucket_risk_scores(probabilities, low_threshold=0.33, high_threshold=0.66)
    assert list(tiers) == ["Low", "Low", "Medium", "Medium", "High", "High"]


def test_bucket_risk_scores_handles_boundary_values():
    # 0.0 and 1.0 are the extremes of a valid probability — must not be dropped as NaN
    tiers = bucket_risk_scores([0.0, 1.0])
    assert tiers.isna().sum() == 0
    assert tiers.iloc[0] == "Low"
    assert tiers.iloc[1] == "High"


def test_crosstab_risk_by_subgroup_normalizes_to_100():
    risk_tier = pd.Series(["Low", "Medium", "High", "Low", "Medium"])
    subgroup = pd.Series([1, 1, 1, 2, 2])
    ct = crosstab_risk_by_subgroup(risk_tier, subgroup)
    # Each row (subgroup level) should sum to ~100%
    row_sums = ct.sum(axis=1)
    for total in row_sums:
        assert total == pytest.approx(100.0)


def test_summarize_high_risk_group_flags_overrepresentation():
    # Construct a case where subgroup level 1 is clearly overrepresented in
    # the High risk tier relative to its share of the overall population.
    subgroup_col = pd.Series([1] * 10 + [2] * 90)  # level 1 is 10% of population
    df = pd.DataFrame({"group": subgroup_col})

    # 8 of the 10 "High" risk rows belong to level 1 — heavily overrepresented
    # relative to level 1's 10% share of the overall population.
    risk_tier = pd.Series(["High"] * 8 + ["Low"] * 2 + ["Low"] * 90, name="risk_tier")
    summary = summarize_high_risk_group(df, risk_tier, ["group"])

    level_1_row = summary[(summary["subgroup_column"] == "group") & (summary["level"] == 1)]
    assert not level_1_row.empty
    pct_high = level_1_row["pct_in_high_risk_group"].iloc[0]
    pct_overall = level_1_row["pct_in_overall_population"].iloc[0]
    assert pct_high > pct_overall  # level 1 is overrepresented among High risk


# --- Regression tests for the index-misalignment bug -----------------------
#
# In practice, `risk_tier` (built from a plain numpy probability array via
# bucket_risk_scores) always gets a fresh 0..n-1 index, while a DataFrame
# like X_test loaded after a train_test_split keeps its original,
# non-contiguous row index. Combining the two via pandas' default
# index-alignment (pd.crosstab, df.loc[boolean_series]) silently produces
# wrong numbers — no error, no warning. These tests reproduce that exact
# scenario and assert the functions are correct regardless of index.

def test_crosstab_risk_by_subgroup_correct_with_mismatched_index():
    # risk_tier has a default 0..3 index; subgroup has a deliberately
    # different, non-contiguous index — exactly the real-world scenario.
    risk_tier = pd.Series(["Low", "High", "Low", "High"])
    subgroup = pd.Series([1, 1, 2, 2], index=[100, 205, 7, 300])

    ct = crosstab_risk_by_subgroup(risk_tier, subgroup)

    # Row-for-row (positional) correspondence: row0=(1,Low), row1=(1,High),
    # row2=(2,Low), row3=(2,High) — so group 1 should be 50/50 Low/High,
    # and group 2 should also be 50/50 Low/High.
    assert ct.loc[1, "Low"] == pytest.approx(50.0)
    assert ct.loc[1, "High"] == pytest.approx(50.0)
    assert ct.loc[2, "Low"] == pytest.approx(50.0)
    assert ct.loc[2, "High"] == pytest.approx(50.0)


def test_summarize_high_risk_group_correct_with_mismatched_index():
    # df has a non-contiguous index; risk_tier has a default 0..3 index.
    df = pd.DataFrame({"group": [10, 10, 20, 20]}, index=[50, 12, 900, 3])
    risk_tier = pd.Series(["High", "Low", "High", "Low"])

    summary = summarize_high_risk_group(df, risk_tier, ["group"])

    # Positionally: row0=(group10,High), row1=(group10,Low),
    # row2=(group20,High), row3=(group20,Low) — group 10 and group 20 should
    # each be 50% of the High-risk tier (1 of 2 High-risk rows each).
    group_10_row = summary[summary["level"] == 10].iloc[0]
    group_20_row = summary[summary["level"] == 20].iloc[0]
    assert group_10_row["pct_in_high_risk_group"] == pytest.approx(50.0)
    assert group_20_row["pct_in_high_risk_group"] == pytest.approx(50.0)


def test_crosstab_risk_by_subgroup_raises_on_length_mismatch():
    # A genuine length mismatch (not just a differing index) must fail
    # loudly rather than silently truncating or padding with NaN.
    risk_tier = pd.Series(["Low", "High", "Low"])
    subgroup = pd.Series([1, 2])
    with pytest.raises(ValueError, match="same length"):
        crosstab_risk_by_subgroup(risk_tier, subgroup)


def test_summarize_high_risk_group_raises_on_length_mismatch():
    df = pd.DataFrame({"group": [1, 2, 3]})
    risk_tier = pd.Series(["Low", "High"])
    with pytest.raises(ValueError, match="same length"):
        summarize_high_risk_group(df, risk_tier, ["group"])
