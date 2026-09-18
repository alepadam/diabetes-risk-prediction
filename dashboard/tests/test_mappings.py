"""Unit tests for dashboard/utils/mappings.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from utils.mappings import AGE_BUCKETS, age_to_bucket, calculate_bmi


def test_age_to_bucket_boundaries():
    # Every bucket boundary, verified against AGE_BUCKETS directly rather
    # than hardcoded expectations, so this stays correct if AGE_BUCKETS
    # is ever edited.
    for code, _label, min_age, max_age in AGE_BUCKETS:
        assert age_to_bucket(min_age) == code
        assert age_to_bucket(max_age) == code


def test_age_to_bucket_below_range_clamps_to_first_bucket():
    assert age_to_bucket(0) == 1
    assert age_to_bucket(17) == 1


def test_age_to_bucket_above_range_clamps_to_last_bucket():
    assert age_to_bucket(80) == 13
    assert age_to_bucket(150) == 13


def test_age_to_bucket_covers_every_bucket_with_no_gaps():
    # Every integer age from 18 to 90 must map to some bucket 1-13 —
    # a gap here would mean a real age silently gets no valid input.
    seen_buckets = {age_to_bucket(age) for age in range(18, 91)}
    assert seen_buckets == set(range(1, 14))


def test_calculate_bmi_known_value():
    # Standard reference value: 70kg at 175cm ≈ 22.86 BMI
    assert calculate_bmi(weight_kg=70, height_cm=175) == pytest.approx(22.857, abs=0.01)


def test_calculate_bmi_raises_on_zero_height():
    with pytest.raises(ValueError):
        calculate_bmi(weight_kg=70, height_cm=0)


def test_calculate_bmi_raises_on_negative_height():
    with pytest.raises(ValueError):
        calculate_bmi(weight_kg=70, height_cm=-10)
