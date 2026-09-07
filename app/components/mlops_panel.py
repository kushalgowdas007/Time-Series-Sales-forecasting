"""
mlops_panel.py — MLOps Monitoring panel.
Three cards: Model Health, Data Drift, Retraining Action.
Fixed: gauge inside card, drift bar chart, better layout.
"""

import streamlit as st
import plotly.graph_objects as go


def _status_color(status: str) -> str:
    return {"GOOD": "#22C55E", "WARNING": "#F59E0B", "POOR": "#EF4444"}.get(status, "#94A3B8")


def _status_badge(status: str) -> str:
    style = {"GOOD": "success", "WARNING": "warning", "POOR": "critical"}.get(status, "neutral")
    icon  = {"GOOD": "●", "WARNING": "⚠", "POOR": "✕"}.get(status, "●")
    return f'<span class="badge badge-{style}">{icon} {status}</span>'


def _mae_gauge(mae: float) -> go.Figure:
    """Compact MAE gauge that renders cleanly inside a card."""
    try:
        val = float(mae) if mae is not None else 4.2
    except (ValueError, TypeError):
        val = 4.2
    val = max(0.0, min(val, 20.0))

    color = "#22C55E" if val <= 5 else ("#F59E0B" if val <= 10 else "#EF4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={
            "font": {"size": 20, "color": "#F8FAFC", "family": "Inter"},
            "suffix": "",
        },
        gauge={
            "axis": {
                "range": [0, 20],
                "tickwidth": 0,
                "tickcolor": "rgba(0,0,0,0)",
                "visible": False,
            },
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#1A2535",
            "borderwidth": 0,
            "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,   5], "color": "rgba(34,197,94,0.07)"},
                {"range": [5,  10], "color": "rgba(245,158,11,0.07)"},
                {"range": [10, 20], "color": "rgba(239,68,68,0.07)"},
            ],
        },
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=5, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#F8FAFC"),
    )
    return fig



def _drift_bar(drift_score: float, threshold: float = 20.0) -> go.Figure:
    """Horizontal bar chart showing drift score vs threshold."""
    color = "#F59E0B" if drift_score > threshold else "#22C55E"
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[drift_score],
        y=["Drift Score"],
        orientation="h",
        marker_color=color,
        marker_line_width=0,
        hovertemplate=f"Drift Score: {drift_score:.1f}<extra></extra>",
        width=0.4,
    ))

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="rgba(148,163,184,0.5)",
        line_width=1.5,
        annotation_text="Threshold",
        annotation_font_size=9,
        annotation_font_color="#475569",
        annotation_position="top right",
    )

    fig.update_layout(
        height=90,
        margin=dict(l=4, r=4, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111B2A",
        xaxis=dict(
            range=[0, max(drift_score * 1.2, threshold * 1.5, 50)],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
        bargap=0,
    )
    return fig


def render_mlops_panel(data: dict) -> None:
    """
    Three-card MLOps monitoring layout:
    Model Health | Data Drift | Retraining Action
    """
    mlops = data.get("mlops", {})

    model_status   = mlops.get("model_status",   "GOOD")
    mae            = mlops.get("mae",             4.2)
    rmse           = mlops.get("rmse",            5.8)
    drift_detected = mlops.get("drift_detected",  True)
    drift_score    = mlops.get("drift_score",     42.8)
    retrain_action = mlops.get("retrain_action",  "RETRAIN_MODEL")

    drift_style    = "warning" if drift_detected else "success"
    drift_label    = "DETECTED" if drift_detected else "NONE"
    retrain_style  = "critical" if retrain_action == "RETRAIN_MODEL" else "success"
    retrain_icon   = "↻" if retrain_action == "RETRAIN_MODEL" else "✓"
    retrain_label  = "RETRAIN" if retrain_action == "RETRAIN_MODEL" else "MODEL OK"

    col1, col2, col3 = st.columns(3)

    # ── MODEL HEALTH ─────────────────────────────────────────────────────────
    with col1:
        card1_html = (
            f'<div class="dash-card">'
            f'<div class="dash-card-header">'
            f'<span class="dash-card-icon">⚙</span>'
            f'<span class="dash-card-title">MODEL HEALTH</span>'
            f'<span style="margin-left:auto">{_status_badge(model_status)}</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">MAE</span>'
            f'<span class="metric-row-value" style="color:{_status_color(model_status)}">{mae}</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">RMSE</span>'
            f'<span class="metric-row-value">{rmse}</span>'
            f'</div>'
            f'<div class="metric-row" style="border-bottom:none">'
            f'<span class="metric-row-label">Model Type</span>'
            f'<span class="metric-row-value">ARIMA</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card1_html, unsafe_allow_html=True)

        # Gauge renders here — directly below the card, still within col1
        fig_mae = _mae_gauge(mae)
        st.plotly_chart(fig_mae, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div style="text-align:center;margin-top:-12px;font-size:0.65rem;'
            'color:var(--text-muted)">MAE threshold: 5.0</div>',
            unsafe_allow_html=True
        )

    # ── DATA DRIFT ────────────────────────────────────────────────────────────
    with col2:
        drift_icon = '⚠' if drift_detected else '✓'
        drift_desc = '⚠ Data drift detected. Distribution shift from historical baseline.' if drift_detected else '✓ No significant drift detected.'
        card2_html = (
            f'<div class="dash-card">'
            f'<div class="dash-card-header">'
            f'<span class="dash-card-icon">📡</span>'
            f'<span class="dash-card-title">DATA DRIFT</span>'
            f'<span style="margin-left:auto">'
            f'<span class="badge badge-{drift_style}">{drift_icon} {drift_label}</span>'
            f'</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Drift Score</span>'
            f'<span class="metric-row-value" style="color:var(--color-{"warning" if drift_detected else "success"})">{drift_score}</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Threshold</span>'
            f'<span class="metric-row-value">20.0</span>'
            f'</div>'
            f'<div class="metric-row" style="border-bottom:none">'
            f'<span class="metric-row-label">Status</span>'
            f'<span class="metric-row-value">{"Above threshold" if drift_detected else "Within range"}</span>'
            f'</div>'
            f'<div class="alert-banner {drift_style}" style="margin-top:0.65rem;font-size:0.74rem">'
            f'{drift_desc}'
            f'</div>'
            f'</div>'
        )
        st.markdown(card2_html, unsafe_allow_html=True)

        fig_drift = _drift_bar(drift_score)
        st.plotly_chart(fig_drift, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            '<div style="text-align:center;margin-top:-14px;font-size:0.65rem;'
            'color:var(--text-muted)">Score vs Threshold (20.0)</div>',
            unsafe_allow_html=True
        )

    # ── RETRAINING ACTION ─────────────────────────────────────────────────────
    with col3:
        explanation = (
            "Data drift has exceeded the threshold. Model retraining is recommended "
            "to restore forecast accuracy."
            if retrain_action == "RETRAIN_MODEL"
            else "Model performance is within acceptable bounds. No retraining required."
        )
        card3_html = (
            f'<div class="dash-card">'
            f'<div class="dash-card-header">'
            f'<span class="dash-card-icon">{retrain_icon}</span>'
            f'<span class="dash-card-title">MODEL ACTION</span>'
            f'<span style="margin-left:auto">'
            f'<span class="badge badge-{retrain_style}">{retrain_icon} {retrain_label}</span>'
            f'</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Action</span>'
            f'<span class="metric-row-value" style="font-size:0.78rem">{retrain_action}</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Trigger</span>'
            f'<span class="metric-row-value">{"Drift > 20.0" if retrain_action == "RETRAIN_MODEL" else "None"}</span>'
            f'</div>'
            f'<div class="metric-row" style="border-bottom:none">'
            f'<span class="metric-row-label">Priority</span>'
            f'<span class="metric-row-value">'
            f'<span class="badge badge-{retrain_style}">{"HIGH" if retrain_action == "RETRAIN_MODEL" else "NONE"}</span>'
            f'</span>'
            f'</div>'
            f'<div class="alert-banner {retrain_style}" style="margin-top:0.65rem;font-size:0.74rem">'
            f'{retrain_icon} {explanation}'
            f'</div>'
            f'</div>'
        )
        st.markdown(card3_html, unsafe_allow_html=True)


    # ── Technical detail expander ─────────────────────────────────────────────
    st.markdown("<div style='margin-top:0.65rem'></div>", unsafe_allow_html=True)
    with st.expander("🔬 Technical MLOps Detail", expanded=False):
        st.markdown(f"""
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| MAE | {mae} | 5.0 | {model_status} |
| RMSE | {rmse} | — | — |
| Drift Score | {drift_score} | 20.0 | {'DRIFT' if drift_detected else 'OK'} |
| Retraining | {retrain_action} | — | — |

**Drift Detection Method:** Mean absolute difference between historical and recent sales distribution.  
**Retraining Trigger:** Automatic when drift score exceeds 20.0 threshold.  
**Model Architecture:** ARIMA (AutoRegressive Integrated Moving Average)
        """)
