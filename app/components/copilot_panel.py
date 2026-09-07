"""
copilot_panel.py — AI Business Copilot insight panel.
Enhanced: richer insight cards, profit detail, spike ratio inline,
          styled recommended action box with gradient.
"""

import streamlit as st


# ── Maps known recommendation text → (icon, style, detail) ───────────────────
_REC_MAP = {
    "No reorder required": (
        "✓", "success",
        "Current inventory exceeds required stock level. Supply chain is stable."
    ),
    "Reorder inventory immediately": (
        "⚡", "critical",
        "Stockout risk is HIGH. Place a reorder immediately to avoid disruption."
    ),
    "Reduce excess inventory": (
        "↓", "warning",
        "Excess units above recommended level. Reduce stock to optimise holding costs."
    ),
    "Monitor demand spike closely": (
        "⚠", "warning",
        "Recent demand is significantly above baseline. Maintain safety stock buffer."
    ),
    "High risk product detected": (
        "⚠", "critical",
        "Product classified as HIGH risk. Review demand patterns and supply reliability."
    ),
    "Business remains profitable": (
        "✓", "success",
        "Positive profit margin maintained. Revenue and profit targets are being met."
    ),
}


def _insight_card(icon: str, title: str, detail: str, style: str = "info") -> str:
    css_class = style if style in ("success", "warning", "critical", "ai") else ""
    return (
        f'<div class="insight-card {css_class}">'
        f'<div class="insight-title">{icon}&nbsp; {title}</div>'
        f'<div class="insight-body">{detail}</div>'
        f'</div>'
    )


def render_copilot_panel(data: dict) -> None:
    """
    Renders the AI Business Copilot with structured insight cards and recommended action.
    """
    if not isinstance(data, dict):
        data = {}

    recommendations = data.get("recommendations", [])
    forecast   = data.get("forecast", 0)
    revenue    = data.get("revenue", 0)
    profit     = data.get("profit", 0)
    margin     = data.get("margin", 0.25)
    risk_cat   = data.get("risk_category", "LOW")
    excess     = data.get("excess_inventory", 0)
    spike      = data.get("spike_detected", False)
    spike_ratio = data.get("spike_ratio", 0.0)
    margin_pct = int(margin * 100)

    header_html = (
        '<div class="dash-card-header" style="margin-bottom:0.85rem">'
        '<span class="dash-card-icon">🤖</span>'
        '<span class="dash-card-title">AI BUSINESS COPILOT</span>'
        '<span style="margin-left:auto">'
        '<span class="badge badge-ai">● ACTIVE</span>'
        '</span>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # ── Insight cards ──────────────────────────────────────────────────────────
    cards_html = ""
    for rec in recommendations:
        if rec in _REC_MAP:
            icon, style, detail = _REC_MAP[rec]
            # Enrich spike detail with actual ratio
            if rec == "Monitor demand spike closely" and spike:
                detail = f"Demand spike ratio: <strong>{spike_ratio:.2f}×</strong> above baseline. Ensure safety stock is adequate."
        else:
            icon, style, detail = "●", "info", rec
        cards_html += _insight_card(icon, rec, detail, style)

    # Add profit card if not already covered
    if profit > 0 and "Business remains profitable" not in recommendations:
        cards_html += _insight_card(
            "✓", "Business remains profitable",
            f"Expected profit: <strong>₹{profit:,}</strong> at {margin_pct}% margin.", "success"
        )

    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Recommended Action ─────────────────────────────────────────────────────
    if excess > 0 and spike:
        action_text = (
            f"Reduce excess inventory ({excess} units above recommended) "
            f"while monitoring the detected demand spike (ratio: {spike_ratio:.2f}×). "
            "Maintain safety stock to guard against stockout."
        )
    elif excess > 0:
        action_text = (
            f"Reduce excess inventory of {excess} units to optimise holding costs. "
            "Current stock significantly exceeds forecasted demand."
        )
    elif spike:
        action_text = (
            f"Monitor demand spike closely. Spike ratio: {spike_ratio:.2f}×. "
            "Ensure adequate safety stock is maintained to prevent stockout."
        )
    else:
        action_text = (
            "Business is operating within normal parameters. "
            "Continue monitoring forecast accuracy, inventory levels, and supply chain health."
        )

    action_html = (
        f'<div style="margin-top:0.85rem">'
        f'<div style="font-size:0.62rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:var(--text-muted);margin-bottom:0.5rem">'
        f'✦ RECOMMENDED ACTION'
        f'</div>'
        f'<div class="outcome-card">'
        f'<div style="font-size:0.82rem;color:var(--text-primary);line-height:1.6">'
        f'{action_text}'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(action_html, unsafe_allow_html=True)


    # ── Expandable business summary ────────────────────────────────────────────
    with st.expander("📋 Full Business Summary", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Forecast Demand | {forecast} units |
| Expected Revenue | ₹{revenue:,} |
| Expected Profit | ₹{profit:,} |
            """)
        with col_b:
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Gross Margin | {margin_pct}% |
| Risk Level | {risk_cat} |
| Demand Spike | {'Yes — {:.2f}×'.format(spike_ratio) if spike else 'No'} |
            """)
