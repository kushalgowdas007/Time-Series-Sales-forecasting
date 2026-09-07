"""
sidebar.py — Fixed left-navigation sidebar component.
Uses CSS-driven active states for pixel-perfect styling.
"""

import streamlit as st


# ── Page registry ──────────────────────────────────────────────────────────────
NAV_SECTIONS = [
    {
        "label": "OVERVIEW",
        "items": [
            {"key": "overview",      "icon": "◉",  "label": "Overview"},
        ],
    },
    {
        "label": "FORECASTING",
        "items": [
            {"key": "forecast",      "icon": "📈", "label": "Demand Forecast"},
        ],
    },
    {
        "label": "INVENTORY",
        "items": [
            {"key": "inventory",     "icon": "📦", "label": "Inventory Health"},
            {"key": "replenishment", "icon": "↻",  "label": "Replenishment"},
        ],
    },
    {
        "label": "INTELLIGENCE",
        "items": [
            {"key": "risk",          "icon": "⚠",  "label": "Risk Intelligence"},
            {"key": "copilot",       "icon": "🤖", "label": "AI Copilot"},
        ],
    },
    {
        "label": "SIMULATION",
        "items": [
            {"key": "digital_twin",  "icon": "◈",  "label": "Digital Twin"},
            {"key": "simulator",     "icon": "⧫",  "label": "What-If Simulator"},
        ],
    },
    {
        "label": "OPERATIONS",
        "items": [
            {"key": "mlops",         "icon": "⚙",  "label": "MLOps Monitoring"},
        ],
    },
]


def render_sidebar() -> str:
    """
    Renders the sidebar navigation and returns the currently selected page key.
    Uses HTML buttons with CSS active class for precise styling.
    """
    if "page" not in st.session_state:
        st.session_state.page = "overview"

    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────────────
        brand_html = (
            '<div class="sidebar-brand">'
            '<div class="sidebar-logo">⬡</div>'
            '<div class="sidebar-brand-name">AI RETAIL</div>'
            '<div class="sidebar-brand-sub">Decision Intelligence</div>'
            '</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)


        # ── Navigation sections ────────────────────────────────────────────────
        for section in NAV_SECTIONS:
            st.markdown(
                f'<div class="sidebar-section-label">{section["label"]}</div>',
                unsafe_allow_html=True
            )
            for item in section["items"]:
                is_active = st.session_state.page == item["key"]
                # Use Streamlit buttons — styled via CSS overrides in dashboard.css
                # The type="primary" flag triggers our CSS active state rule
                if st.button(
                    f'{item["icon"]}  {item["label"]}',
                    key=f'nav_{item["key"]}',
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.page = item["key"]
                    st.rerun()

        # ── Footer area ─────────────────────────────────────────────────────────
        st.markdown('<hr class="dash-divider">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-section-label" style="opacity:0.45">⚙ Settings &nbsp;·&nbsp; v1.0.0</div>',
            unsafe_allow_html=True
        )

    return st.session_state.page
