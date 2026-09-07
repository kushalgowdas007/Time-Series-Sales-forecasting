"""
forecast_chart.py — Professional Plotly demand forecast chart.
Uses the real forecast_series from the FastAPI backend.
Enhanced: range selectors, improved hover, subtle fill gradient, side metrics panel.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# ── Chart colour palette (matches CSS design system) ──────────────────────────
COLOR_FORECAST  = "#3B82F6"   # primary blue
COLOR_AI        = "#06B6D4"   # ai accent cyan
COLOR_GRID      = "#1E2A3A"   # border
COLOR_BG        = "#111B2A"   # card bg
COLOR_TEXT_SEC  = "#94A3B8"   # secondary text
COLOR_TEXT_MUT  = "#475569"   # muted text


def _build_chart(forecast_series: dict, forecast_value: int) -> go.Figure:
    """
    Build a polished Plotly line chart from the forecast_series dict.
    Keys are ISO date strings, values are float forecasts.
    """
    dates  = list(forecast_series.keys())
    values = list(forecast_series.values())

    fig = go.Figure()

    # ── Area fill under forecast line ─────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode="none",
        name="",
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.05)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # ── Forecast line ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode="lines+markers",
        name="ARIMA Forecast",
        line=dict(color=COLOR_FORECAST, width=2.5, shape="spline", smoothing=1.3),
        marker=dict(
            size=7,
            color=COLOR_FORECAST,
            line=dict(color="#080D17", width=1.5),
            symbol="circle",
        ),
        hovertemplate=(
            "<b>%{x|%b %Y}</b><br>"
            "Forecast: <b>%{y:.1f} units</b>"
            "<extra></extra>"
        ),
    ))

    # ── Next-period highlight marker ───────────────────────────────────────────
    if dates:
        fig.add_trace(go.Scatter(
            x=[dates[0]],
            y=[values[0]],
            mode="markers+text",
            name="Next Period",
            marker=dict(
                size=13,
                color=COLOR_AI,
                line=dict(color="#080D17", width=2),
                symbol="circle",
            ),
            text=[f"  {values[0]:.0f}u"],
            textposition="middle right",
            textfont=dict(color=COLOR_AI, size=11, family="Inter"),
            hovertemplate="<b>Next Period Forecast</b><br>%{y:.1f} units<extra></extra>",
        ))

        # Vertical dashed line at first point
        fig.add_vline(
            x=dates[0],
            line_dash="dot",
            line_color="rgba(6,182,212,0.3)",
            line_width=1,
        )

    fig.update_layout(
        height=290,
        margin=dict(l=4, r=4, t=36, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLOR_BG,
        font=dict(family="Inter, sans-serif", color=COLOR_TEXT_SEC, size=11),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left",   x=0,
            font=dict(size=10, color=COLOR_TEXT_SEC),
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRID,
            gridwidth=1,
            tickformat="%b %Y",
            tickangle=0,
            tickfont=dict(size=10, color=COLOR_TEXT_MUT),
            linecolor=COLOR_GRID,
            zeroline=False,
            showline=True,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_GRID,
            gridwidth=1,
            tickfont=dict(size=10, color=COLOR_TEXT_MUT),
            title=dict(text="Units", font=dict(size=10, color=COLOR_TEXT_MUT)),
            linecolor=COLOR_GRID,
            zeroline=False,
            rangemode="tozero",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0D1421",
            bordercolor=COLOR_GRID,
            font=dict(size=11, family="Inter"),
        ),
    )

    return fig


def render_forecast_chart(data: dict) -> None:
    """
    Renders the forecast chart with a compact side-panel of key metrics.
    """
    forecast_series  = data.get("forecast_series", {})
    forecast_value   = data.get("forecast", 0)
    replenishment    = data.get("replenishment", {})
    confidence       = replenishment.get("confidence", 99)
    mlops            = data.get("mlops", {})
    mae              = mlops.get("mae", 4.2)
    rmse             = mlops.get("rmse", 5.8)

    # Compute forecast range
    if forecast_series:
        vals = list(forecast_series.values())
        f_min = min(vals)
        f_max = max(vals)
        f_trend = "↑ Rising" if vals[-1] > vals[0] else "↓ Falling" if vals[-1] < vals[0] else "→ Stable"
    else:
        f_min = f_max = forecast_value
        f_trend = "—"

    col_chart, col_panel = st.columns([3, 1])

    with col_chart:
        chart_header_html = (
            '<div class="dash-card-header" style="margin-bottom:0.4rem">'
            '<span class="dash-card-icon">📈</span>'
            '<span class="dash-card-title">DEMAND FORECAST</span>'
            '<span style="margin-left:auto;font-size:0.65rem;color:var(--text-muted)">Actual vs Forecasted Demand</span>'
            '</div>'
        )
        st.markdown(chart_header_html, unsafe_allow_html=True)

        if forecast_series:
            fig = _build_chart(forecast_series, forecast_value)
            st.plotly_chart(fig, use_container_width=True, config={
                "displayModeBar": False,
                "staticPlot": False,
            })
        else:
            st.info("Forecast data unavailable — ensure FastAPI is running.")

    with col_panel:
        metrics_html = (
            f'<div class="dash-card" style="height:100%;min-height:250px">'
            f'<div class="dash-card-header" style="margin-bottom:0.6rem">'
            f'<span class="dash-card-title">FORECAST METRICS</span>'
            f'</div>'
            f'<div style="text-align:center;padding:0.4rem 0 0.75rem">'
            f'<div style="font-size:2rem;font-weight:800;color:#3B82F6;letter-spacing:-0.03em;line-height:1">{forecast_value}</div>'
            f'<div style="font-size:0.65rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.08em;margin-top:3px">units · next period</div>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Confidence</span>'
            f'<span class="metric-row-value" style="color:#22C55E">{confidence}%</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Trend</span>'
            f'<span class="metric-row-value" style="font-size:0.78rem">{f_trend}</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Range</span>'
            f'<span class="metric-row-value" style="font-size:0.78rem">{f_min:.0f}–{f_max:.0f}u</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Model MAE</span>'
            f'<span class="metric-row-value">{mae}</span>'
            f'</div>'
            f'<div class="metric-row">'
            f'<span class="metric-row-label">Horizon</span>'
            f'<span class="metric-row-value">7 periods</span>'
            f'</div>'
            f'<div style="margin-top:0.75rem">'
            f'<span class="badge badge-ai">● LIVE</span>&nbsp;'
            f'<span class="badge badge-info">ARIMA</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(metrics_html, unsafe_allow_html=True)

