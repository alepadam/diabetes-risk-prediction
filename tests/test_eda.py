"""Unit tests for src/eda.py — the association-measure functions the EDA
notebook relies on. Uses small hand-constructed examples with a known
answer (perfect correlation, perfect independence) rather than the real
dataset, so these run fast and assert exact expected behavior.
"""
import numpy as np
import pandas as pd
import pytest

from src.eda import (
    chi_square_test,
    correlation_ratio,
    cramers_v,
    mann_whitney_test,
    point_biserial,
)


def test_point_biserial_perfect_positive_correlation():
    binary = pd.Series([0, 0, 0, 1, 1, 1])
    continuous = pd.Series([1, 1, 1, 2, 2, 2])
    r = point_biserial(binary, continuous)
    assert r == pytest.approx(1.0, abs=1e-6)


def test_point_biserial_no_correlation():
    # Balanced so the continuous variable carries no information about the group
    binary = pd.Series([0, 0, 1, 1])
    continuous = pd.Series([1, 2, 1, 2])
    r = point_biserial(binary, continuous)
    assert r == pytest.approx(0.0, abs=1e-6)


def test_cramers_v_independent_variables_near_zero():
    rng = np.random.default_rng(42)
    x = pd.Series(rng.integers(0, 2, 2000))
    y = pd.Series(rng.integers(0, 2, 2000))  # generated independently of x
    v = cramers_v(x, y)
    assert 0 <= v < 0.1  # should be close to 0 for independent variables


def test_cramers_v_perfect_association():
    x = pd.Series([0, 0, 1, 1, 2, 2])
    y = pd.Series([0, 0, 1, 1, 2, 2])  # y is fully determined by x
    v = cramers_v(x, y)
    assert v == pytest.approx(1.0, abs=1e-6)


def test_correlation_ratio_bounds():
    categories = pd.Series([0, 0, 1, 1, 2, 2])
    measurements = pd.Series([10, 11, 20, 21, 30, 31])
    eta = correlation_ratio(categories, measurements)
    assert 0 <= eta <= 1
    assert eta > 0.9  # categories almost fully explain the measurement variance


def test_chi_square_test_returns_expected_keys():
    x = pd.Series([0, 0, 1, 1, 0, 1])
    y = pd.Series([0, 1, 1, 1, 0, 0])
    result = chi_square_test(x, y)
    assert set(result.keys()) == {"chi2", "p_value", "dof"}
    assert result["p_value"] >= 0


def test_mann_whitney_detects_shifted_distribution():
    group_a = pd.Series([1, 2, 3, 4, 5])
    group_b = pd.Series([10, 11, 12, 13, 14])  # clearly shifted higher
    result = mann_whitney_test(group_a, group_b)
    assert result["p_value"] < 0.05
