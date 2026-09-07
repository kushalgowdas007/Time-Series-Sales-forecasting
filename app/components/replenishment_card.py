"""
replenishment_card.py — Autonomous Replenishment operations card.
Uses real data from the /forecast endpoint replenishment block.
Enhanced: inventory comparison bar, clearer action badges.
"""

import streamlit as st


def _progress_bar(pct: float, style: str = "primary") -> str:
    pct = max(0, min(pct, 100))
    return f'<div class="progress-track"><div class="progress-fill {style}" style="width:{pct:.1f}%"></div></div>'


def render_replenishment_card(data: dict) -> None:
    """
    Professional replenishment status card with inventory comparison bar.
    """
    rep = data.get("replenishment", {})

    action          = rep.get("action", "NO_ACTION")
    required_stock  = rep.get("required_stock", 85)
    current_inv     = rep.get("current_inventory", 100)
    reorder_qty     = rep.get("reorder_quantity", 0)
    order_date      = rep.get("recommended_order_date", "—")
    lead_time       = rep.get("lead_time_days", 5)
    confidence      = rep.get("confidence", 99)

    action_style = "success" if action == "NO_ACTION" else "critical"
    action_icon  = "✓" if action == "NO_ACTION" else "⚡"
    action_label = "NO ACTION" if action == "NO_ACTION" else "REORDER NOW"

    rationale = (
        "Current inventory exceeds required stock. No reorder needed."
        if action == "NO_ACTION"
        else f"Stock deficit detected. Reorder {reorder_qty} units immediately."
    )

    # Inventory coverage: current vs required
    coverage_pct = min((current_inv / max(required_stock, 1)) * 100, 100)
    bar_style = "success" if current_inv >= required_stock else "critical"

    current_inv_color = "var(--color-success)" if current_inv >= required_stock else "var(--color-critical)"

    html_content = (
        f'<div class="dash-card">'
        f'<div class="dash-card-header">'
        f'<span class="dash-card-icon">↻</span>'
        f'<span class="dash-card-title">AUTONOMOUS REPLENISHMENT</span>'
        f'<span style="margin-left:auto">'
        f'<span class="badge badge-{action_style}">{action_icon} {action_label}</span>'
        f'</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Required Stock</span>'
        f'<span class="metric-row-value">{required_stock} units</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Current Inventory</span>'
        f'<span class="metric-row-value" style="color:{current_inv_color}">{current_inv} units</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Reorder Quantity</span>'
        f'<span class="metric-row-value">{reorder_qty} units</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Order Date</span>'
        f'<span class="metric-row-value" style="font-size:0.78rem">{order_date}</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Lead Time</span>'
        f'<span class="metric-row-value">{lead_time} days</span>'
        f'</div>'
        f'<div class="metric-row" style="border-bottom:none">'
        f'<span class="metric-row-label">Confidence</span>'
        f'<span class="metric-row-value" style="color:var(--color-success)">{confidence}%</span>'
        f'</div>'
        f'<div style="margin-top:0.85rem">'
        f'<div style="display:flex;justify-content:space-between;font-size:0.68rem;color:var(--text-muted);margin-bottom:3px">'
        f'<span>Stock Coverage (Current vs Required)</span>'
        f'<span>{coverage_pct:.0f}%</span>'
        f'</div>'
        f'{_progress_bar(coverage_pct, bar_style)}'
        f'</div>'
        f'<div class="alert-banner {action_style}" style="margin-top:0.65rem;font-size:0.74rem">'
        f'{action_icon} {rationale}'
        f'</div>'
        f'</div>'
    )

    st.markdown(html_content, unsafe_allow_html=True)

