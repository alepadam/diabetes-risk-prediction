"""Unit tests for src/hypothesis_testing.py."""
import numpy as np
import pandas as pd
import pytest

from src.hypothesis_testing import (
    apply_multiple_testing_correction,
    check_normality,
    check_variance_homogeneity,
    cohens_d,
    effect_size_label,
    rank_biserial_from_mannwhitney,
)


def test_check_normality_detects_normal_data():
    rng = np.random.default_rng(42)
    data = pd.Series(rng.normal(0, 1, 3000))
    result = check_normality(data)
    assert result["is_normal"] is True


def test_check_normality_detects_skewed_data():
    rng = np.random.default_rng(42)
    data = pd.Series(rng.exponential(2, 3000))
    result = check_normality(data)
    assert result["is_normal"] is False


def test_check_variance_homogeneity_equal_variance():
    rng = np.random.default_rng(42)
    group_a = pd.Series(rng.normal(0, 5, 500))
    group_b = pd.Series(rng.normal(10, 5, 500))
    result = check_variance_homogeneity(group_a, group_b)
    assert result["equal_variance"] is True


def test_cohens_d_zero_for_identical_groups():
    group = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cohens_d(group, group) == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_sign_reflects_direction():
    group_a = pd.Series([1.0, 2.0, 3.0])
    group_b = pd.Series([4.0, 5.0, 6.0])
    assert cohens_d(group_a, group_b) < 0
    assert cohens_d(group_b, group_a) > 0


def test_rank_biserial_bounds():
    # U=0 (group 1 always ranked lower) -> effect size should be at the +1 extreme
    r = rank_biserial_from_mannwhitney(u_stat=0, n1=10, n2=10)
    assert r == pytest.approx(1.0)


def test_multiple_testing_correction_flags_fewer_significant():
    # A batch where several p-values are just under 0.05 should end up with
    # fewer (or equal) significant results after FDR correction, never more.
    pvals = [0.001, 0.04, 0.045, 0.049, 0.5]
    result = apply_multiple_testing_correction(pvals)
    n_raw_significant = sum(p < 0.05 for p in pvals)
    n_corrected_significant = result["significant_after_correction"].sum()
    assert n_corrected_significant <= n_raw_significant


def test_effect_size_label_thresholds():
    assert effect_size_label(0.05, "point_biserial") == "negligible"
    assert effect_size_label(0.9, "point_biserial") == "large"
    assert effect_size_label(0.05, "cramers_v") == "negligible"
    assert effect_size_label(0.5, "cramers_v") == "large"
