"""
executive_dashboard.py
AI Retail Decision Intelligence Platform — Enterprise SaaS Dashboard v2.0

Run with:
    streamlit run app/executive_dashboard.py
    (from the AI-Retail-Decision-Intelligence-Platform/ directory)

FastAPI backend must be running:
    uvicorn api.app:app --reload
"""

import sys
import os
from datetime import datetime

import streamlit as st
import requests

# ── Path setup: allow imports from app and project root ───────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── Component imports ─────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.kpi_cards          import render_kpi_cards, render_business_health
from components.forecast_chart     import render_forecast_chart
from components.inventory_card     import render_inventory_card
from components.risk_card          import render_risk_card, render_supply_chain_card, render_demand_intel_card
from components.replenishment_card import render_replenishment_card
from components.copilot_panel      import render_copilot_panel
from components.simulation_panel   import render_simulation_panel
from components.digital_twin       import render_digital_twin
from components.mlops_panel        import render_mlops_panel

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Retail Decision Intelligence Platform",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────────────────────────────────────
_css_path = os.path.join(os.path.dirname(__file__), "styles", "dashboard.css")
if os.path.exists(_css_path):
    with open(_css_path, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/forecast"


@st.cache_data(ttl=30)   # Refresh every 30 s
def fetch_data() -> dict | None:
    try:
        resp = requests.get(API_URL, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _api_error_banner():
    banner_html = (
        f'<div class="alert-banner critical" style="margin-bottom:1.25rem">'
        f'⚡ <strong>Cannot reach FastAPI backend</strong> at <code>{API_URL}</code><br>'
        f'<span style="font-size:0.72rem;opacity:0.85">'
        f'Start it with: <code>uvicorn api.app:app --reload</code>'
        f'&nbsp;·&nbsp; Make sure you are in the project root directory.'
        f'</span>'
        f'</div>'
    )
    st.markdown(banner_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
page = render_sidebar()

# ─────────────────────────────────────────────────────────────────────────────
# TOP HEADER  (sticky)
# ─────────────────────────────────────────────────────────────────────────────
now_str = datetime.now().strftime("%b %d, %Y   %I:%M %p")

# Determine model status label from data (shown in header)
_header_model_status = "ARIMA/SARIMA"

header_html = (
    f'<div class="dash-header">'
    f'<div class="dash-header-brand">'
    f'<div class="dash-header-title">AI Retail Decision Intelligence Platform</div>'
    f'<div class="dash-header-sub">Forecast · Optimize · Predict · Decide</div>'
    f'</div>'
    f'<div class="dash-header-right">'
    f'<div class="header-meta">'
    f'<div class="header-datetime">{now_str}</div>'
    f'<div class="status-dot">System Operational</div>'
    f'</div>'
    f'<span class="badge badge-ai" style="font-size:0.6rem">{_header_model_status}</span>'
    f'</div>'
    f'</div>'
)
st.markdown(header_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────────────────────────────────────
data = fetch_data()

# ── Refresh button row ────────────────────────────────────────────────────────
col_spacer, col_last, col_refresh = st.columns([5, 2, 1])
with col_last:
    last_upd = datetime.now().strftime("Last updated: %I:%M %p")
    st.markdown(
        f'<div style="font-size:0.68rem;color:var(--text-muted);'
        f'text-align:right;padding-top:0.4rem">{last_upd}</div>',
        unsafe_allow_html=True
    )
with col_refresh:
    if st.button("↻ Refresh", help="Reload all data from FastAPI backend", type="secondary"):
        st.cache_data.clear()
        st.rerun()

if data is None:
    _api_error_banner()
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# VERTICAL KPI LIST (used in AI Copilot sidebar column)
# ─────────────────────────────────────────────────────────────────────────────

def _render_kpi_vertical(data: dict) -> None:
    """Compact vertical key-metric list for secondary columns."""
    forecast   = data.get("forecast", 0)
    revenue    = data.get("revenue", 0)
    profit     = data.get("profit", 0)
    inventory  = data.get("inventory", 0)
    risk_cat   = data.get("risk_category", "LOW")

    risk_style = {"LOW": "success", "MEDIUM": "warning", "HIGH": "critical"}.get(risk_cat, "neutral")

    rows = [
        ("FORECAST",  f"{forecast} units",  "var(--color-primary)"),
        ("REVENUE",   f"₹{revenue:,}",       "var(--color-ai)"),
        ("PROFIT",    f"₹{profit:,}",        "var(--color-success)"),
        ("INVENTORY", f"{inventory} units",  "var(--text-primary)"),
        ("RISK",      risk_cat,              f"var(--color-{risk_style.replace('critical','critical').replace('neutral','text-secondary')})"),
    ]
    rows_html = "".join(
        f'<div class="metric-row">'
        f'<span class="metric-row-label">{lbl}</span>'
        f'<span class="metric-row-value" style="color:{clr}">{val}</span>'
        f'</div>'
        for lbl, val, clr in rows
    )
    st.markdown(f'<div class="dash-card">{rows_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
if page == "overview":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">EXECUTIVE KPIs</div>', unsafe_allow_html=True)
    render_kpi_cards(data)

    st.markdown("<div style='margin-top:0.85rem'></div>", unsafe_allow_html=True)

    # Row 2: Business Health + Demand Intelligence | Forecast Chart
    col_left, col_forecast = st.columns([1, 2])
    with col_left:
        render_business_health(data)
        st.markdown("<div style='margin-top:0.65rem'></div>", unsafe_allow_html=True)
        render_demand_intel_card(data)
    with col_forecast:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        render_forecast_chart(data)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

    # Row 3: Inventory | Risk | Replenishment
    col_inv, col_risk, col_rep = st.columns(3)
    with col_inv:
        render_inventory_card(data)
    with col_risk:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        render_risk_card(data)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_rep:
        render_replenishment_card(data)

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

    # Row 4: AI Copilot | Supply Chain
    col_cop, col_sc = st.columns([2, 1])
    with col_cop:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        render_copilot_panel(data)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_sc:
        render_supply_chain_card(data)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "forecast":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">DEMAND FORECAST</div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    render_forecast_chart(data)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

    with st.expander("📋 Forecast Series Data", expanded=False):
        import pandas as pd
        fs = data.get("forecast_series", {})
        if fs:
            df_fc = pd.DataFrame({
                "Date": list(fs.keys()),
                "Forecast (units)": [round(v, 2) for v in fs.values()],
            })
            st.dataframe(df_fc, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top:0.65rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        render_demand_intel_card(data)
    with col_b:
        render_replenishment_card(data)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "inventory":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">INVENTORY HEALTH</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        render_inventory_card(data)
    with col_b:
        render_replenishment_card(data)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "replenishment":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">AUTONOMOUS REPLENISHMENT</div>', unsafe_allow_html=True)
    render_replenishment_card(data)

    st.markdown("<div style='margin-top:0.65rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        render_inventory_card(data)
    with col_b:
        render_supply_chain_card(data)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "risk":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">RISK INTELLIGENCE</div>', unsafe_allow_html=True)

    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    render_risk_card(data)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:0.65rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        render_supply_chain_card(data)
    with col_b:
        render_demand_intel_card(data)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "copilot":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">AI BUSINESS COPILOT</div>', unsafe_allow_html=True)

    col_cop, col_kpi = st.columns([2, 1])
    with col_cop:
        st.markdown('<div class="dash-card">', unsafe_allow_html=True)
        render_copilot_panel(data)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_kpi:
        _render_kpi_vertical(data)
        st.markdown("<div style='margin-top:0.65rem'></div>", unsafe_allow_html=True)
        render_business_health(data)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "digital_twin":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">DIGITAL TWIN — STATE COMPARISON</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    render_digital_twin(data)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "simulator":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">WHAT-IF SIMULATOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="dash-card">', unsafe_allow_html=True)
    render_simulation_panel(data)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
elif page == "mlops":
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">MLOPS MONITORING</div>', unsafe_allow_html=True)
    render_mlops_panel(data)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
footer_html = (
    '<div class="dash-footer">'
    '<strong>AI Retail Decision Intelligence Platform</strong><br>'
    'Built with Python &bull; FastAPI &bull; Streamlit &bull; ARIMA/SARIMA'
    '&nbsp;&nbsp;|&nbsp;&nbsp;'
    'v1.0.0'
    '</div>'
)
st.markdown(footer_html, unsafe_allow_html=True)

