"""Formal hypothesis testing utilities — assumption checks, effect sizes, and
multiple-testing correction.

src/eda.py already runs chi-square and Mann-Whitney tests for exploration.
This module adds what's needed to treat those as a proper hypothesis-testing
pass rather than exploratory signal:

- normality/variance assumption checks, so the test choice is justified
  rather than assumed
- effect sizes alongside p-values (a p-value alone doesn't say how big or
  practically meaningful an association is)
- multiple-testing correction, since running ~20 tests at once inflates the
  chance of false positives if each is judged at p < 0.05 in isolation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def check_normality(series: pd.Series, sample_size: int = 5000, random_state: int = 42) -> dict:
    """D'Agostino-Pearson normality test. Samples down large series first —
    with n > ~5000, even trivial deviations from normality become
    'significant', making the test uninformative at full dataset size.
    """
    data = series.sample(n=min(len(series), sample_size), random_state=random_state) if len(series) > sample_size else series
    stat, p = stats.normaltest(data)
    return {"stat": stat, "p_value": p, "is_normal": bool(p >= 0.05), "n_used": len(data)}


def check_variance_homogeneity(group_a: pd.Series, group_b: pd.Series) -> dict:
    """Levene's test for equal variance between two groups — the assumption
    an independent-samples t-test relies on. Used here to justify why
    Mann-Whitney (which doesn't assume equal variance or normality) is the
    safer default for this dataset's skewed health-day features.
    """
    stat, p = stats.levene(group_a, group_b)
    return {"stat": stat, "p_value": p, "equal_variance": bool(p >= 0.05)}


def rank_biserial_from_mannwhitney(u_stat: float, n1: int, n2: int) -> float:
    """Effect size for a Mann-Whitney U test, on a comparable -1 to 1 scale
    to Pearson/point-biserial r. u_stat is the U statistic for group 1.
    """
    return 1 - (2 * u_stat) / (n1 * n2)


def cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Standardized mean difference — reported for context alongside the
    non-parametric test, but rank-biserial (above) is the primary effect
    size here since it doesn't assume normal/equal-variance data.
    """
    n1, n2 = len(group_a), len(group_b)
    pooled_std = np.sqrt(((n1 - 1) * group_a.var() + (n2 - 1) * group_b.var()) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group_a.mean() - group_b.mean()) / pooled_std


def apply_multiple_testing_correction(p_values: list[float], method: str = "fdr_bh", alpha: float = 0.05) -> pd.DataFrame:
    """Correct a batch of p-values for multiple comparisons.

    Default is Benjamini-Hochberg FDR (method='fdr_bh') — less conservative
    than Bonferroni, appropriate here since we care about controlling the
    rate of false discoveries across ~20 exploratory tests, not guaranteeing
    zero false positives at any cost.
    """
    from statsmodels.stats.multitest import multipletests

    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return pd.DataFrame({
        "p_value_raw": p_values,
        "p_value_corrected": p_corrected,
        "significant_after_correction": reject,
    })


def effect_size_label(value: float, measure: str) -> str:
    """Rough qualitative label for an effect size, using conventional cutoffs.

    These thresholds (Cohen's rules of thumb) are context-independent
    conventions, not a substitute for domain judgment about what counts as
    a meaningful effect in a health-risk context.
    """
    v = abs(value)
    if measure in ("point_biserial", "rank_biserial", "cohens_d", "correlation_ratio"):
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "small"
        elif v < 0.5:
            return "medium"
        else:
            return "large"
    elif measure == "cramers_v":
        if v < 0.1:
            return "negligible"
        elif v < 0.2:
            return "small"
        elif v < 0.4:
            return "medium"
        else:
            return "large"
    return "unknown"
