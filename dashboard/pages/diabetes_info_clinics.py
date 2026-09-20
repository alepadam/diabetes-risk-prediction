"""Diabetes Info & Clinics page — general educational content plus a
simple clinic locator built on a Google Maps search embed (no API key
required, since this is a student/portfolio project rather than a
production deployment with billing set up).
"""
import sys
import urllib.parse
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_DIR.parent
for _p in (DASHBOARD_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from utils.styling import render_sidebar

render_sidebar(st)

st.title("🏥 Diabetes Info & Clinics")
st.markdown(
    """
    <div class="disclaimer-box">
    <strong>⚠️ General education only.</strong> The information below is a
    plain-language summary, not medical advice. Diagnosis and treatment
    decisions should always come from a qualified healthcare provider.
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Educational content
# =====================================================================
st.markdown("### Understanding Diabetes")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Types of Diabetes")
    st.markdown(
        """
        - **Type 1** — the body produces little or no insulin. Usually
          diagnosed in children/young adults; requires lifelong insulin.
        - **Type 2** — the body doesn't use insulin effectively (insulin
          resistance). The most common form, strongly linked to weight,
          activity level, and family history — and the type this
          project's model predicts risk for.
        - **Prediabetes** — blood sugar is higher than normal but not yet
          high enough to be classified as diabetes. Often reversible with
          lifestyle changes.
        - **Gestational diabetes** — develops during pregnancy, usually
          resolves after birth, but raises the mother's future Type 2 risk.
        """
    )

    st.markdown("#### Common Symptoms")
    st.markdown(
        """
        - Increased thirst and frequent urination
        - Unexplained fatigue
        - Blurred vision
        - Slow-healing cuts or sores
        - Unexplained weight loss (more common in Type 1)
        - Tingling or numbness in hands/feet

        Type 2 diabetes symptoms often develop gradually and can go
        unnoticed for years — which is part of why risk screening (like
        the calculator on this dashboard) has value even without symptoms.
        """
    )

with col2:
    st.markdown("#### Risk Factors")
    st.markdown(
        """
        Many of these are exactly what the Risk Calculator on this
        dashboard asks about:

        - Being overweight or obese (higher BMI)
        - Physical inactivity
        - Family history of diabetes
        - High blood pressure or high cholesterol
        - Age 45 and older
        - History of gestational diabetes
        - Limited access to healthcare or regular checkups
        """
    )

    st.markdown("#### Prevention & Management")
    st.markdown(
        """
        - Regular physical activity (even brisk walking helps)
        - A diet with plenty of vegetables and fruit, limited refined sugar
        - Maintaining a healthy weight
        - Routine health checkups, especially if risk factors are present
        - Not smoking, and limiting alcohol
        - For those already diagnosed: consistent blood sugar monitoring
          and following a care plan set by a doctor
        """
    )

st.markdown(
    """
    <div class="disclaimer-box">
    <strong>When to see a doctor:</strong> if you notice persistent symptoms
    above, have several risk factors, or haven't had a checkup in a while —
    it's worth getting screened. Early detection makes management
    significantly easier.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# =====================================================================
# Clinic locator
# =====================================================================
st.markdown("### Find a Nearby Clinic")
st.markdown(
    "Enter your city or area to search for nearby clinics and diabetes "
    "care centers on Google Maps."
)

location_input = st.text_input(
    "City or area", placeholder="e.g. Petaling Jaya, Selangor",
)

if st.button("Find Nearby Clinics", type="primary"):
    if not location_input.strip():
        st.warning("Please enter a city or area first.")
    else:
        search_query = f"diabetes clinic near {location_input.strip()}"
        encoded_query = urllib.parse.quote(search_query)

        maps_search_url = f"https://www.google.com/maps/search/{encoded_query}"
        maps_embed_url = f"https://www.google.com/maps?q={encoded_query}&output=embed"

        st.markdown(f"**Results for:** {search_query}")
        try:
            st.iframe(maps_embed_url, height=450)
        except Exception:  # noqa: BLE001 — embedding can fail for reasons outside our control (network, browser policy); the link below always works
            st.info("Map preview couldn't load — use the link below instead.")

        st.link_button("Open full results in Google Maps ↗", maps_search_url)
        st.caption(
            "This uses a public Google Maps search link — it doesn't verify "
            "clinic hours, credentials, or current availability. Always call "
            "ahead to confirm."
        )
