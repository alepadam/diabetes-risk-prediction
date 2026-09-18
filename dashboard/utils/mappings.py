"""Maps human-friendly inputs (actual age, height/weight) to the coded
categories the CDC Diabetes Health Indicators dataset — and therefore the
trained model — actually expects. Coding follows the BRFSS 2015
questionnaire, matching UCI dataset ID 891.
"""
from __future__ import annotations

# (bucket_code, label, min_age_inclusive, max_age_inclusive)
AGE_BUCKETS = [
    (1, "18-24", 18, 24),
    (2, "25-29", 25, 29),
    (3, "30-34", 30, 34),
    (4, "35-39", 35, 39),
    (5, "40-44", 40, 44),
    (6, "45-49", 45, 49),
    (7, "50-54", 50, 54),
    (8, "55-59", 55, 59),
    (9, "60-64", 60, 64),
    (10, "65-69", 65, 69),
    (11, "70-74", 70, 74),
    (12, "75-79", 75, 79),
    (13, "80+", 80, 120),
]

GENHLTH_OPTIONS = {
    1: "Excellent",
    2: "Very good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}

EDUCATION_OPTIONS = {
    1: "Never attended school / kindergarten only",
    2: "Elementary (grades 1-8)",
    3: "Some high school (grades 9-11)",
    4: "High school graduate",
    5: "Some college or technical school",
    6: "College graduate",
}

INCOME_OPTIONS = {
    1: "Less than $10,000",
    2: "$10,000 to $14,999",
    3: "$15,000 to $19,999",
    4: "$20,000 to $24,999",
    5: "$25,000 to $34,999",
    6: "$35,000 to $49,999",
    7: "$50,000 to $74,999",
    8: "$75,000 or more",
}


def age_to_bucket(age: int) -> int:
    """Convert a real age in years to the dataset's 1-13 age bucket code.

    Ages below 18 clamp to bucket 1; ages above 80 clamp to bucket 13 — the
    dataset's youngest/oldest buckets, matching how BRFSS itself
    top/bottom-codes age.
    """
    if age < 18:
        return 1
    for code, _label, min_age, max_age in AGE_BUCKETS:
        if min_age <= age <= max_age:
            return code
    return 13  # anything above the last bucket's max


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    """Standard BMI formula: weight(kg) / height(m)^2."""
    height_m = height_cm / 100
    if height_m <= 0:
        raise ValueError("Height must be greater than 0.")
    return weight_kg / (height_m ** 2)
