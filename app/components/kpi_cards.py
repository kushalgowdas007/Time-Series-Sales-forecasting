"""
kpi_cards.py — Five executive KPI cards for the Overview section.
Uses real data from the FastAPI /forecast response.
Visual upgrade: larger values, gradient accents, trend indicators.
"""

import streamlit as st


def _badge(text: str, style: str = "neutral") -> str:
    return f'<span class="badge badge-{style}">{text}</span>'


def _kpi_card(
    label: str,
    value: str,
    sub: str = "",
    accent: str = "primary",
    badge: str = "",
    value_color: str = "",
) -> str:
    color_style = f'style="color:{value_color}"' if value_color else ""
    badge_html = f'<div style="margin-top:0.45rem">{badge}</div>' if badge else ""
    return (
        f'<div class="kpi-card {accent}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" {color_style}>{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'{badge_html}'
        f'</div>'
    )


def render_kpi_cards(data: dict) -> None:
    """
    Renders 5 KPI cards in a single responsive row using real backend data.
    """
    if not isinstance(data, dict):
        data = {}

    forecast   = data.get("forecast", 0)
    revenue    = data.get("revenue", 0)
    profit     = data.get("profit", 0)
    margin     = data.get("margin", 0.25)
    inventory  = data.get("inventory", 0)
    excess     = data.get("excess_inventory", 0)
    risk_cat   = data.get("risk_category", "LOW")
    risk_score = data.get("risk_score", 0)

    risk_style = {"LOW": "success", "MEDIUM": "warning", "HIGH": "critical"}.get(risk_cat, "neutral")
    margin_pct = int(margin * 100)
    excess_label = f"{excess} excess" if excess > 0 else "Optimal level"
    excess_style = "warning" if excess > 0 else "success"

    cards_html = ""

    # Card 1 — Demand Forecast
    cards_html += _kpi_card(
        label="DEMAND FORECAST",
        value=f"{forecast:,}",
        sub="units · next period",
        accent="primary",
        badge=_badge("ARIMA/SARIMA", "ai"),
    )

    # Card 2 — Expected Revenue
    cards_html += _kpi_card(
        label="EXPECTED REVENUE",
        value=f"₹{revenue:,}",
        sub="based on forecast × price",
        accent="ai-accent",
        badge=_badge("PROJECTED", "info"),
    )

    # Card 3 — Expected Profit
    cards_html += _kpi_card(
        label="EXPECTED PROFIT",
        value=f"₹{profit:,}",
        sub=f"{margin_pct}% gross margin",
        accent="success",
        badge=_badge(f"{margin_pct}% MARGIN", "success"),
    )

    # Card 4 — Current Inventory
    cards_html += _kpi_card(
        label="CURRENT INVENTORY",
        value=f"{inventory:,}",
        sub="units currently on hand",
        accent="warning" if excess > 0 else "success",
        badge=_badge(excess_label, excess_style),
    )

    # Card 5 — Business Risk
    cards_html += _kpi_card(
        label="BUSINESS RISK",
        value=risk_cat,
        sub=f"Score: {risk_score} / 100",
        accent=risk_style,
        badge=_badge(f"SCORE {risk_score}", risk_style),
    )

    # Wrap in a 5-column flex grid
    grid_html = f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.75rem;margin-bottom:0.1rem">{cards_html}</div>'
    st.markdown(grid_html, unsafe_allow_html=True)


def render_business_health(data: dict) -> None:
    """
    Compact business health summary card with dynamic status and top reason.
    """
    if not isinstance(data, dict):
        data = {}

    status  = data.get("status", "HEALTHY")
    recs    = data.get("recommendations", [])
    spike   = data.get("spike_detected", False)
    risk    = data.get("risk_category", "LOW")
    excess  = data.get("excess_inventory", 0)

    if status == "HEALTHY" and risk in ("LOW", "MEDIUM") and not spike:
        health_style = "success"
        health_icon  = "✓"
        summary = recs[0] if recs else "No immediate business-critical issues detected."
    else:
        health_style = "warning"
        health_icon  = "⚠"
        summary = recs[0] if recs else "Attention required — review risk and inventory."

    # Build secondary detail line
    detail_parts = []
    if spike:
        detail_parts.append("Demand spike active")
    if excess > 0:
        detail_parts.append(f"{excess} units excess inventory")
    if risk == "HIGH":
        detail_parts.append("High risk classification")
    detail = " · ".join(detail_parts) if detail_parts else "All systems nominal"

    badge_html = f'<span class="badge badge-{health_style}">{health_icon} {status}</span>'

    card_html = (
        f'<div class="dash-card" style="min-height:110px">'
        f'<div class="dash-card-header">'
        f'<span class="dash-card-icon">🏢</span>'
        f'<span class="dash-card-title">BUSINESS HEALTH</span>'
        f'</div>'
        f'<div style="margin-bottom:0.6rem">{badge_html}</div>'
        f'<div style="font-size:0.78rem;color:var(--text-primary);font-weight:500;margin-bottom:0.3rem">{summary}</div>'
        f'<div style="font-size:0.68rem;color:var(--text-muted)">{detail}</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

