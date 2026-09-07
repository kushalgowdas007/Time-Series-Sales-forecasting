"""
digital_twin.py — Digital Twin visual state comparison component.
Renders CURRENT STATE vs SIMULATED STATE side by side.
"""

import streamlit as st
import requests


API_BASE = "http://127.0.0.1:8000"


def _status_badge(is_stockout: bool) -> str:
    if is_stockout:
        return '<span class="badge badge-critical">⚡ STOCKOUT</span>'
    return '<span class="badge badge-success">✓ SAFE</span>'


def render_digital_twin(data: dict) -> None:
    """
    Digital Twin visual comparison: Current State ←→ Simulated State.
    Allows adjusting inventory increase % to see the simulated state.
    """
    if not isinstance(data, dict):
        data = {}

    try:
        current_inventory = int(data.get("inventory", 100))
    except (ValueError, TypeError):
        current_inventory = 100

    try:
        forecast_value = float(data.get("forecast", 32))
    except (ValueError, TypeError):
        forecast_value = 32.0

    header_html = (
        '<div class="dash-card-header" style="margin-bottom:0.75rem">'
        '<span class="dash-card-icon">◇</span>'
        '<span class="dash-card-title">DIGITAL TWIN — STATE COMPARISON</span>'
        '<span style="margin-left:auto">'
        '<span class="badge badge-ai">Simulation</span>'
        '</span>'
        '</div>'
        '<div style="font-size:0.76rem;color:var(--text-secondary);margin-bottom:0.85rem">'
        'Compare the current operational state with a simulated inventory scenario.'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # ── Simulation control ─────────────────────────────────────────────────────
    inv_increase = st.slider(
        "Simulated Inventory Increase (%)",
        min_value=0, max_value=100, value=20, step=10,
        key="dt_inv_increase",
        help="Simulate increasing the inventory by this percentage."
    )

    # Compute simulated values locally (matches digital_twin.py logic)
    sim_inventory  = int(current_inventory * (1 + inv_increase / 100))
    cost_before    = current_inventory * 10
    cost_after     = sim_inventory * 10
    stockout_before = forecast_value > current_inventory
    stockout_after  = forecast_value > sim_inventory

    # ── Visual comparison ──────────────────────────────────────────────────────
    col_cur, col_arrow, col_sim = st.columns([5, 1, 5])

    with col_cur:
        cur_html = (
            f'<div class="twin-box">'
            f'<div class="twin-box-label">◉ CURRENT STATE</div>'
            f'<div style="margin-bottom:0.75rem">'
            f'<div class="twin-metric">{current_inventory}</div>'
            f'<div class="twin-metric-label">units in inventory</div>'
            f'</div>'
            f'<div style="margin-bottom:0.75rem">'
            f'<div class="twin-metric">₹{cost_before:,}</div>'
            f'<div class="twin-metric-label">holding cost</div>'
            f'</div>'
            f'<div>'
            f'<div class="twin-metric-label">Stockout Status</div>'
            f'{_status_badge(stockout_before)}'
            f'</div>'
            f'</div>'
        )
        st.markdown(cur_html, unsafe_allow_html=True)

    with col_arrow:
        arrow_html = '<div class="twin-arrow" style="padding-top:3.5rem;font-size:1.8rem;color:#3B82F6">→</div>'
        st.markdown(arrow_html, unsafe_allow_html=True)

    with col_sim:
        delta_inv  = sim_inventory - current_inventory
        delta_cost = cost_after - cost_before
        sim_html = (
            f'<div class="twin-box simulated">'
            f'<div class="twin-box-label" style="color:var(--color-ai)">◈ SIMULATED STATE</div>'
            f'<div style="margin-bottom:0.75rem">'
            f'<div class="twin-metric" style="color:var(--color-ai)">{sim_inventory}</div>'
            f'<div class="twin-metric-label">units (+{delta_inv})</div>'
            f'</div>'
            f'<div style="margin-bottom:0.75rem">'
            f'<div class="twin-metric">₹{cost_after:,}</div>'
            f'<div class="twin-metric-label">holding cost (+₹{delta_cost:,})</div>'
            f'</div>'
            f'<div>'
            f'<div class="twin-metric-label">Stockout Status</div>'
            f'{_status_badge(stockout_after)}'
            f'</div>'
            f'</div>'
        )
        st.markdown(sim_html, unsafe_allow_html=True)

    # ── Interpretation ─────────────────────────────────────────────────────────
    if not stockout_before and not stockout_after:
        msg = f"Both states are safe. Increasing inventory adds ₹{delta_cost:,} holding cost with no stockout benefit."
        style = "info"
    elif stockout_before and not stockout_after:
        msg = f"Simulation resolves stockout risk. Recommend increasing inventory by {inv_increase}%."
        style = "success"
    elif not stockout_before and stockout_after:
        msg = "Warning: simulated scenario introduces stockout risk despite higher inventory."
        style = "warning"
    else:
        msg = "Both states carry stockout risk. Consider demand reduction or further inventory increase."
        style = "critical"

    alert_html = f'<div class="alert-banner {style}" style="margin-top:0.65rem;font-size:0.74rem">◇ {msg}</div>'
    st.markdown(alert_html, unsafe_allow_html=True)

