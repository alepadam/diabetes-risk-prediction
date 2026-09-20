"""Diabetes Risk Prediction Dashboard — entry point.

Uses st.navigation() with position="top" for a top navigation bar rather
than the default sidebar page list. Page icons are set here as a parameter
(not baked into filenames) — this also sidesteps a real problem: emoji
characters in filenames survive fine in most tools, but Windows' built-in
zip "Extract All" often mangles UTF-8 filenames into garbage (CP437
fallback), which is exactly what happened before this was refactored.

Run with: streamlit run app.py (from inside the dashboard/ folder)
"""
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

import streamlit as st

from utils.styling import apply_custom_theme

# set_page_config and the custom CSS injection must happen exactly once,
# here in the true entry script — calling either again inside a page file
# would raise an error (page config) or just be redundant (CSS).
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
)
apply_custom_theme(st)

pages = [
    st.Page("pages/overview.py", title="Overview", icon="🩺", default=True),
    st.Page("pages/risk_calculator.py", title="Risk Calculator", icon="🎯"),
    st.Page("pages/about_us.py", title="About Us", icon="📊"),
    st.Page("pages/diabetes_info_clinics.py", title="Diabetes Info & Clinics", icon="🏥"),
]

nav = st.navigation(pages, position="top")
nav.run()
