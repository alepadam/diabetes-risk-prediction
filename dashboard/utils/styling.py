"""Shared visual identity for the dashboard: a muted, warm palette instead
of the default bright blue/purple Streamlit look. Deep teal as the primary
accent, warm off-white background, earthy risk-tier colors instead of
neon red/green.

Every page imports COLORS from here rather than hardcoding hex values, so
the palette only needs to change in one place.
"""

COLORS = {
    "primary": "#3B6E77",       # deep teal — main accent
    "primary_dark": "#2A4F56",  # darker teal — headers, emphasis
    "secondary": "#8C6E4A",     # warm taupe — secondary accent
    "background": "#FAF8F5",    # warm off-white
    "surface": "#EFEAE1",       # card/sidebar background
    "border": "#DDD6C9",        # subtle borders/dividers
    "text": "#2B2B28",          # near-black warm text
    "text_muted": "#6B6558",    # secondary/caption text
    "low_risk": "#4C7A5E",      # muted forest green
    "medium_risk": "#C98A2C",   # muted ochre/amber
    "high_risk": "#A8453A",     # muted brick red
}

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {COLORS['background']};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {COLORS['text']};
    }}

    /* Headers */
    h1, h2, h3 {{
        color: {COLORS['primary_dark']};
        font-weight: 600;
    }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background-color: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 10px;
        padding: 1rem;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {COLORS['text_muted']};
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['primary_dark']};
        color: white;
    }}

    /* Risk tier badges */
    .risk-badge {{
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.2rem;
        color: white;
        text-align: center;
    }}
    .risk-badge-low {{ background-color: {COLORS['low_risk']}; }}
    .risk-badge-medium {{ background-color: {COLORS['medium_risk']}; }}
    .risk-badge-high {{ background-color: {COLORS['high_risk']}; }}

    /* Disclaimer box */
    .disclaimer-box {{
        background-color: {COLORS['surface']};
        border-left: 4px solid {COLORS['secondary']};
        border-radius: 4px;
        padding: 0.8rem 1.2rem;
        font-size: 0.9rem;
        color: {COLORS['text_muted']};
        margin-bottom: 1.5rem;
    }}

    /* Divider */
    hr {{
        border-color: {COLORS['border']};
    }}
</style>
"""


def apply_custom_theme(st) -> None:
    """Inject the shared CSS. Call once near the top of every page.

    Takes the `st` module as an argument rather than importing streamlit
    directly here, so this module has zero hard dependency on streamlit
    being installed — makes it importable/testable in plain Python too.
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar(st) -> None:
    """Render the shared branding sidebar. Call from EVERY page, not just
    one — under st.navigation(), only the active page's script body
    executes per render, so sidebar content defined in a single page file
    would only appear on that one page and silently vanish everywhere
    else. This was an actual bug (fixed): the sidebar previously only
    lived in pages/overview.py's own script.
    """
    with st.sidebar:
        st.markdown("### 🩺 Diabetes Risk\nPrediction Dashboard")
        st.markdown("---")
        st.markdown(
            "Built on the **CDC Diabetes Health Indicators** dataset "
            "(BRFSS 2015, 253,680 respondents)."
        )
        st.markdown("---")
        st.caption("Muhammad Aliff Adam bin Sultan · Universiti Malaya")


def risk_badge_html(risk_tier: str, probability: float) -> str:
    """Build the HTML for a colored risk-tier badge."""
    tier_class = {
        "Low": "risk-badge-low",
        "Medium": "risk-badge-medium",
        "High": "risk-badge-high",
    }.get(risk_tier, "risk-badge-medium")
    return (
        f'<div class="risk-badge {tier_class}">'
        f'{risk_tier} Risk — {probability * 100:.1f}% predicted probability'
        f'</div>'
    )
