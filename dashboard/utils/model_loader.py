"""Cached loading of the trained model, preprocessor, and metadata.

Streamlit reruns the whole script on every interaction, so anything
expensive (loading a joblib model, reading a file) must be wrapped in
@st.cache_resource / @st.cache_data or it reloads on every click.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def get_project_root() -> Path:
    """dashboard/utils/model_loader.py -> parents[2] = project root."""
    return Path(__file__).resolve().parents[2]


@st.cache_resource
def load_model():
    root = get_project_root()
    return joblib.load(root / "models" / "best_model.joblib")


@st.cache_resource
def load_preprocessor():
    root = get_project_root()
    return joblib.load(root / "models" / "preprocessor.joblib")


@st.cache_data
def load_metadata() -> dict:
    root = get_project_root()
    metadata_path = root / "models" / "model_metadata.json"
    with open(metadata_path) as f:
        return json.load(f)


@st.cache_data
def load_statistical_results() -> pd.DataFrame | None:
    root = get_project_root()
    path = root / "reports" / "statistical_test_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_high_risk_summary() -> pd.DataFrame | None:
    root = get_project_root()
    path = root / "reports" / "high_risk_group_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_figure_path(filename: str) -> Path | None:
    """Return the path to a saved figure from the notebooks, or None if it
    doesn't exist yet (e.g. the corresponding notebook hasn't been run).
    """
    root = get_project_root()
    path = root / "reports" / "figures" / filename
    return path if path.exists() else None


def artifacts_available() -> dict[str, bool]:
    """Check which required artifacts exist, so pages can show a clear
    'run notebook X first' message instead of a raw file-not-found crash.
    """
    root = get_project_root()
    return {
        "model": (root / "models" / "best_model.joblib").exists(),
        "preprocessor": (root / "models" / "preprocessor.joblib").exists(),
        "metadata": (root / "models" / "model_metadata.json").exists(),
    }
