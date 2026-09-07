"""
inventory_card.py

Professional inventory health card for the
AI Retail Decision Intelligence Platform.

Displays:
- Current inventory
- Recommended stock
- Safety stock
- Excess inventory
- Stockout risk
- Inventory utilisation
"""

import streamlit as st


def _progress_bar(
    percentage: float,
    label: str = "",
    bar_class: str = "primary"
) -> None:
    """
    Render a progress bar using Streamlit HTML.
    """

    percentage = max(
        0.0,
        min(float(percentage), 100.0)
    )

    label_html = ""

    if label:
        label_html = f"""
        <div
            style="
                font-size:0.68rem;
                color:var(--text-muted);
                margin-bottom:4px;
            "
        >
            {label}
        </div>
        """

    html = f"""
    <div style="width:100%;">

        {label_html}

        <div class="progress-track">

            <div
                class="progress-fill {bar_class}"
                style="width:{percentage:.1f}%"
            ></div>

        </div>

        <div
            style="
                font-size:0.65rem;
                color:var(--text-secondary);
                text-align:right;
                margin-top:3px;
            "
        >
            {percentage:.0f}%
        </div>

    </div>
    """

    st.markdown(
        html,
        unsafe_allow_html=True
    )


def render_inventory_card(data: dict) -> None:
    """
    Render the inventory health card.

    Parameters
    ----------
    data : dict
        Data returned by the FastAPI /forecast endpoint.
    """

    # ------------------------------------------------------------------
    # Read API values
    # ------------------------------------------------------------------

    inventory = data.get(
        "inventory",
        100
    )

    recommended_stock = data.get(
        "recommended_stock",
        38
    )

    safety_stock = data.get(
        "safety_stock",
        53
    )

    excess = data.get(
        "excess_inventory",
        0
    )

    stockout_risk = str(
        data.get(
            "stockout_risk",
            "LOW"
        )
    ).upper()


    # ------------------------------------------------------------------
    # Calculate percentages
    # ------------------------------------------------------------------

    inventory_base = max(
        float(inventory),
        1.0
    )

    utilisation_pct = (
        float(recommended_stock)
        / inventory_base
    ) * 100

    excess_pct = (
        float(excess)
        / inventory_base
    ) * 100

    utilisation_pct = min(
        max(utilisation_pct, 0),
        100
    )

    excess_pct = min(
        max(excess_pct, 0),
        100
    )


    # ------------------------------------------------------------------
    # Determine styles
    # ------------------------------------------------------------------

    if excess > 0:
        inventory_badge = "warning"
        inventory_badge_text = "⚠ EXCESS"
    else:
        inventory_badge = "success"
        inventory_badge_text = "✓ OPTIMAL"


    stockout_styles = {
        "LOW": "success",
        "MEDIUM": "warning",
        "HIGH": "critical"
    }

    stockout_style = stockout_styles.get(
        stockout_risk,
        "neutral"
    )


    if excess > 20:
        utilisation_style = "warning"
    else:
        utilisation_style = "success"


    # ------------------------------------------------------------------
    # Alert message
    # ------------------------------------------------------------------

    if excess > 0:

        alert_class = "warning"

        alert_text = (
            f"⚠ Excess inventory of "
            f"<strong>{int(excess)} units</strong> detected. "
            f"Consider reducing stock."
        )

    else:

        alert_class = "success"

        alert_text = (
            "✓ Inventory levels are optimal."
        )


    # ------------------------------------------------------------------
    # Main inventory card
    # ------------------------------------------------------------------

    card_html = (
        f'<div class="dash-card">'
        f'<div class="dash-card-header">'
        f'<span class="dash-card-icon">📦</span>'
        f'<span class="dash-card-title">INVENTORY HEALTH</span>'
        f'<span style="margin-left:auto;">'
        f'<span class="badge badge-{inventory_badge}">{inventory_badge_text}</span>'
        f'</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Current Inventory</span>'
        f'<span class="metric-row-value" style="color:var(--text-primary);">{int(inventory)} units</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Recommended Level</span>'
        f'<span class="metric-row-value">{int(recommended_stock)} units</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Safety Stock</span>'
        f'<span class="metric-row-value">{int(safety_stock)} units</span>'
        f'</div>'
        f'<div class="metric-row">'
        f'<span class="metric-row-label">Excess Inventory</span>'
        f'<span class="metric-row-value" style="color:var(--color-warning);">{int(excess)} units</span>'
        f'</div>'
        f'<div class="metric-row" style="border-bottom:none;">'
        f'<span class="metric-row-label">Stockout Risk</span>'
        f'<span class="metric-row-value">'
        f'<span class="badge badge-{stockout_style}">{stockout_risk}</span>'
        f'</span>'
        f'</div>'
        f'<div style="margin-top:0.85rem;">'
        f'<div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:4px;">Inventory Utilisation</div>'
        f'<div class="progress-track">'
        f'<div class="progress-fill {utilisation_style}" style="width:{utilisation_pct:.1f}%"></div>'
        f'</div>'
        f'<div style="font-size:0.65rem;color:var(--text-secondary);text-align:right;margin-top:3px;">{utilisation_pct:.0f}%</div>'
        f'</div>'
        f'<div style="margin-top:0.65rem;">'
        f'<div style="font-size:0.68rem;color:var(--text-muted);margin-bottom:4px;">Excess Inventory</div>'
        f'<div class="progress-track">'
        f'<div class="progress-fill warning" style="width:{excess_pct:.1f}%"></div>'
        f'</div>'
        f'<div style="font-size:0.65rem;color:var(--text-secondary);text-align:right;margin-top:3px;">{excess_pct:.0f}%</div>'
        f'</div>'
        f'<div class="alert-banner {alert_class}" style="margin-top:0.65rem;font-size:0.74rem;">{alert_text}</div>'
        f'</div>'
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Streamlit must interpret the string as HTML.
    # ------------------------------------------------------------------

    st.markdown(
        card_html,
        unsafe_allow_html=True
    )