"""
risk_card.py — Compact Risk Intelligence + Supply Chain + Demand Intel cards.
Enhanced: mini gauge inside card, better badge layout, risk progress bar.
Robust handling of missing/null values, valid Plotly colors, clean HTML rendering.
"""

import streamlit as st
import plotly.graph_objects as go


def _risk_color(category: str) -> str:
    cat = str(category or "").strip().upper()
    return {"LOW": "#22C55E", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"}.get(cat, "#94A3B8")


def _risk_badge_style(category: str) -> str:
    cat = str(category or "").strip().upper()
    return {"LOW": "success", "MEDIUM": "warning", "HIGH": "critical"}.get(cat, "neutral")


def _mini_gauge(score: float, category: str) -> go.Figure:
    """Compact arc gauge — fits inside a card column."""
    cat = str(category or "LOW").strip().upper()
    if cat not in ("LOW", "MEDIUM", "HIGH"):
        cat = "LOW"
    color = _risk_color(cat)

    try:
        val = float(score) if score is not None else 0.0
    except (ValueError, TypeError):
        val = 0.0
    val = max(0.0, min(val, 100.0))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={
            "font": {"size": 26, "color": "#F8FAFC", "family": "Inter"},
            "suffix": "",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 0,
                "tickcolor": "rgba(0,0,0,0)",
                "visible": False,
            },
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "#1A2535",
            "borderwidth": 0,
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,  40], "color": "rgba(34,197,94,0.07)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.07)"},
                {"range": [70, 100], "color": "rgba(239,68,68,0.07)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 2},
                "thickness": 0.7,
                "value": val,
            },
        },
    ))
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#F8FAFC"),
    )
    return fig


def render_risk_card(data: dict) -> None:
    """
    Compact risk intelligence card — gauge + metric rows + status badges.
    """
    if not isinstance(data, dict):
        data = {}

    raw_cat = data.get("risk_category") or data.get("risk") or "LOW"
    risk_category = str(raw_cat).strip().upper()
    if risk_category not in ("LOW", "MEDIUM", "HIGH"):
        risk_category = "LOW"

    raw_score = data.get("risk_score")
    if raw_score is None:
        raw_score = 15 if risk_category == "LOW" else (50 if risk_category == "MEDIUM" else 85)
    try:
        risk_score = float(raw_score)
    except (ValueError, TypeError):
        risk_score = 0.0
    risk_score = max(0.0, min(risk_score, 100.0))

    spike_detected = bool(data.get("spike_detected", False))
    try:
        spike_ratio = float(data.get("spike_ratio", 1.0))
    except (ValueError, TypeError):
        spike_ratio = 1.0

    supply_chain = data.get("supply_chain") if isinstance(data.get("supply_chain"), dict) else {}
    sc_risk = str(supply_chain.get("risk", "LOW")).strip().upper()
    if sc_risk not in ("LOW", "MEDIUM", "HIGH"):
        sc_risk = "LOW"

    status = str(data.get("status", "HEALTHY")).strip().upper()

    badge_cat    = _risk_badge_style(risk_category)
    badge_sc     = _risk_badge_style(sc_risk)
    spike_style  = "warning" if spike_detected else "success"
    spike_label  = "DETECTED" if spike_detected else "NONE"
    status_style = "success" if status == "HEALTHY" else "warning"
    cat_color    = _risk_color(risk_category)

    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        header_html = (
            '<div class="dash-card-header">'
            '<span class="dash-card-icon">⚠</span>'
            '<span class="dash-card-title">RISK INTELLIGENCE</span>'
            '</div>'
        )
        st.markdown(header_html, unsafe_allow_html=True)

        fig = _mini_gauge(risk_score, risk_category)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        badge_html = (
            f'<div style="text-align:center;margin-top:-6px">'
            f'<span class="badge badge-{badge_cat}">{risk_category} RISK</span>'
            f'</div>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)

    with col_metrics:
        score_display = int(risk_score) if risk_score.is_integer() else f"{risk_score:.1f}"
        metrics_html = (
            f'<div style="padding-top:0.25rem">'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Risk Score</span>'
            f'<span class="metric-row-value" style="color:{cat_color};font-size:1rem;font-weight:700">'
            f'{score_display} <span style="font-size:0.72rem;color:var(--text-muted);font-weight:400">/ 100</span>'
            f'</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Demand Spike</span>'
            f'<span class="metric-row-value">'
            f'<span class="badge badge-{spike_style}">{spike_label}</span>'
            f'</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Spike Ratio</span>'
            f'<span class="metric-row-value">{spike_ratio:.2f}×</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Supply Chain</span>'
            f'<span class="metric-row-value">'
            f'<span class="badge badge-{badge_sc}">{sc_risk}</span>'
            f'</span>'
            f'</div>'
            f'<div class="metric-row" style="border-bottom:none">'
            f'<span class="metric-row-label">Overall Status</span>'
            f'<span class="metric-row-value">'
            f'<span class="badge badge-{status_style}">{status}</span>'
            f'</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(metrics_html, unsafe_allow_html=True)


def render_supply_chain_card(data: dict) -> None:
    """
    Compact supply chain risk alert card with delay detail.
    """
    if not isinstance(data, dict):
        data = {}

    sc = data.get("supply_chain") if isinstance(data.get("supply_chain"), dict) else {}
    try:
        delay = int(sc.get("supplier_delay", 10))
    except (ValueError, TypeError):
        delay = 10

    risk = str(sc.get("risk", "HIGH")).strip().upper()
    if risk not in ("LOW", "MEDIUM", "HIGH"):
        risk = "HIGH"

    try:
        safety_rec = int(sc.get("recommended_safety_stock", 16))
    except (ValueError, TypeError):
        safety_rec = 16

    risk_style = _risk_badge_style(risk)
    reason = (
        f"Supplier delay of {delay} days exceeds 7-day threshold."
        if delay > 7
        else "Supply chain within normal parameters."
    )
    alert_class = "critical" if risk == "HIGH" else ("warning" if risk == "MEDIUM" else "success")

    # Delay bar (out of 30 days max)
    delay_pct = min((delay / 30) * 100, 100)
    bar_style  = "critical" if delay > 14 else ("warning" if delay > 7 else "success")
    delay_color = "var(--color-critical)" if delay > 7 else "var(--color-success)"

    card_html = (
        f'<div class="dash-card">'
        f'<div class="dash-card-header">'
        f'<span class="dash-card-icon">🚚</span>'
        f'<span class="dash-card-title">SUPPLY CHAIN RISK</span>'
        f'<span style="margin-left:auto">'
        f'<span class="badge badge-{risk_style}">{risk}</span>'
        f'</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Supplier Delay</span>'
        f'<span class="metric-row-value" style="color:{delay_color}">{delay} days</span>'
        f'</div>'
        f'<div class="metric-row" style="border-bottom:none">'
        f'<span class="metric-row-label">Rec. Safety Stock</span>'
        f'<span class="metric-row-value">{safety_rec} units</span>'
        f'</div>'
        f'<div style="margin-top:0.7rem">'
        f'<div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:3px">'
        f'Delay Severity (vs 30-day max)'
        f'</div>'
        f'<div class="progress-track">'
        f'<div class="progress-fill {bar_style}" style="width:{delay_pct:.1f}%"></div>'
        f'</div>'
        f'</div>'
        f'<div class="alert-banner {alert_class}" style="margin-top:0.65rem;font-size:0.74rem">'
        f'🚚 {reason}'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def render_demand_intel_card(data: dict) -> None:
    """
    Compact demand intelligence / spike detection card with ratio progress.
    """
    if not isinstance(data, dict):
        data = {}

    spike_detected = bool(data.get("spike_detected", False))
    try:
        spike_ratio = float(data.get("spike_ratio", 1.0))
    except (ValueError, TypeError):
        spike_ratio = 1.0

    style  = "warning" if spike_detected else "success"
    label  = "DEMAND SPIKE DETECTED" if spike_detected else "DEMAND NORMAL"
    icon   = "⚠" if spike_detected else "✓"
    detail = (
        f"Recent demand is <strong>{spike_ratio:.2f}×</strong> above baseline. "
        "Monitor closely for stockout risk."
        if spike_detected
        else "Demand is within expected baseline range."
    )

    # Spike ratio bar (capped at 4× = 100%)
    ratio_pct = min((spike_ratio / 4.0) * 100, 100)
    ratio_color = "var(--color-warning)" if spike_detected else "var(--color-success)"

    card_html = (
        f'<div class="dash-card">'
        f'<div class="dash-card-header">'
        f'<span class="dash-card-icon">📊</span>'
        f'<span class="dash-card-title">DEMAND INTELLIGENCE</span>'
        f'<span style="margin-left:auto">'
        f'<span class="badge badge-{style}">{icon} {label}</span>'
        f'</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Spike Ratio</span>'
        f'<span class="metric-row-value" style="color:{ratio_color}">{spike_ratio:.2f}×</span>'
        f'</div>'
        f'<div style="margin-top:0.7rem">'
        f'<div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:3px">'
        f'Demand Intensity (vs 4× baseline)'
        f'</div>'
        f'<div class="progress-track">'
        f'<div class="progress-fill {style}" style="width:{ratio_pct:.1f}%"></div>'
        f'</div>'
        f'</div>'
        f'<div class="alert-banner {style}" style="margin-top:0.65rem;font-size:0.74rem">'
        f'{detail}'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

