"""Statistical helpers for exploratory data analysis.

This dataset is mostly binary and ordinal features with only BMI as a truly
continuous variable, so plain Pearson correlation (what df.corr() gives you)
is the wrong tool for most feature-target relationships — it only strictly
applies to two continuous/binary variables. This module picks the right
association measure per variable-type pair:

- binary vs binary / binary vs continuous  -> point-biserial correlation
  (mathematically equivalent to Pearson here, but named for clarity)
- categorical (nominal, >2 levels) vs categorical -> Cramer's V
- categorical (nominal) vs continuous -> correlation ratio (eta)

All functions return values in comparable [0, 1] (or [-1, 1] where sign is
meaningful) ranges so results can be ranked against each other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def point_biserial(binary_series: pd.Series, continuous_series: pd.Series) -> float:
    """Correlation between a binary variable and a continuous/binary variable.

    Returns the signed correlation coefficient (-1 to 1).
    """
    r, _p = stats.pointbiserialr(binary_series, continuous_series)
    return r


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Association strength between two categorical variables, bias-corrected.

    Returns a value in [0, 1] — no sign, since nominal categories have no
    inherent order. Uses the Bergsma (2013) bias correction, which matters
    for smaller contingency tables.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0:
        return 0.0
    return np.sqrt(phi2_corr / denom)


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Eta — association between a categorical variable and a continuous one.

    Returns a value in [0, 1]: how much of the continuous variable's
    variance is explained by category membership.
    """
    categories = np.asarray(categories)
    measurements = np.asarray(measurements, dtype=float)
    unique_cats = np.unique(categories)

    cat_means = np.array([measurements[categories == c].mean() for c in unique_cats])
    cat_counts = np.array([np.sum(categories == c) for c in unique_cats])
    overall_mean = measurements.mean()

    ss_between = np.sum(cat_counts * (cat_means - overall_mean) ** 2)
    ss_total = np.sum((measurements - overall_mean) ** 2)
    if ss_total == 0:
        return 0.0
    return np.sqrt(ss_between / ss_total)


def chi_square_test(x: pd.Series, y: pd.Series) -> dict:
    """Chi-square test of independence between two categorical variables.

    Returns chi2 statistic, p-value, and degrees of freedom.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, _expected = stats.chi2_contingency(confusion_matrix)
    return {"chi2": chi2, "p_value": p, "dof": dof}


def mann_whitney_test(group_a: pd.Series, group_b: pd.Series) -> dict:
    """Non-parametric test comparing a continuous variable between two groups.

    Preferred over a t-test here since features like MentHlth/PhysHlth are
    heavily right-skewed (most respondents report 0 days), violating the
    normality assumption a t-test relies on.
    """
    stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return {"u_stat": stat, "p_value": p}


def compute_target_associations(
    df: pd.DataFrame,
    target: str,
    binary_cols: list[str],
    ordinal_cols: list[str],
    continuous_cols: list[str],
) -> pd.DataFrame:
    """Compute the appropriate association measure between every feature and
    the target, returning one ranked table instead of a misleading blanket
    Pearson correlation across mixed variable types.
    """
    rows = []

    for col in binary_cols:
        r = point_biserial(df[target], df[col])
        rows.append({"feature": col, "type": "binary", "measure": "point-biserial", "value": r, "abs_value": abs(r)})

    for col in ordinal_cols:
        r = point_biserial(df[target], df[col])
        rows.append({"feature": col, "type": "ordinal", "measure": "point-biserial", "value": r, "abs_value": abs(r)})

    for col in continuous_cols:
        eta = correlation_ratio(df[target], df[col])
        rows.append({"feature": col, "type": "continuous", "measure": "correlation ratio (eta)", "value": eta, "abs_value": abs(eta)})

    result = pd.DataFrame(rows).sort_values("abs_value", ascending=False).reset_index(drop=True)
    return result


def compute_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Variance Inflation Factor per feature — flags multicollinearity.

    Rule of thumb: VIF > 5 warrants a closer look, VIF > 10 is a real
    problem for models sensitive to collinearity (e.g. plain logistic
    regression coefficient interpretation).
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df[columns].assign(_intercept=1.0)
    vif_data = pd.DataFrame({
        "feature": columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(len(columns))],
    })
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
