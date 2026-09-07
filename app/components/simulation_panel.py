"""
simulation_panel.py — What-If Simulator with Streamlit sliders.
Calls the FastAPI /simulate POST endpoint using existing src modules.
"""

import streamlit as st
import requests


API_BASE = "http://127.0.0.1:8000"


def render_simulation_panel(data: dict) -> None:
    """
    What-If Simulator — interactive sliders + POST to /simulate → outcome card.
    """
    header_html = (
        '<div class="dash-card-header" style="margin-bottom:0.75rem">'
        '<span class="dash-card-icon">◇</span>'
        '<span class="dash-card-title">WHAT-IF SIMULATOR</span>'
        '<span style="margin-left:auto">'
        '<span class="badge badge-ai">Interactive</span>'
        '</span>'
        '</div>'
        '<div style="font-size:0.76rem;color:var(--text-secondary);margin-bottom:0.85rem">'
        'Adjust parameters and run a simulation to see projected outcomes.'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    col_sliders, col_result = st.columns([1, 1])

    with col_sliders:
        discount = st.slider(
            "💰 Price Discount (%)",
            min_value=0, max_value=50, value=10, step=5,
            help="Percentage price discount applied to the base price."
        )
        supplier_delay = st.slider(
            "🚚 Supplier Delay (days)",
            min_value=0, max_value=30, value=10, step=1,
            help="Simulated supplier delay in days."
        )
        inv_increase = st.slider(
            "📦 Inventory Increase (%)",
            min_value=0, max_value=100, value=20, step=10,
            help="Simulated percentage increase in inventory for the Digital Twin."
        )

        run_btn = st.button("▶ Run Simulation", use_container_width=True)

    with col_result:
        if run_btn:
            try:
                resp = requests.post(
                    f"{API_BASE}/simulate",
                    json={
                        "discount_percent": float(discount),
                        "supplier_delay_days": int(supplier_delay),
                        "inventory_increase_percent": float(inv_increase),
                    },
                    timeout=8,
                )
                result = resp.json()

                proj_demand  = result.get("projected_demand", 0)
                exp_revenue  = result.get("expected_revenue", 0)
                exp_profit   = result.get("expected_profit", 0)
                disc_price   = result.get("discounted_price", 0)
                stockout_r   = result.get("stockout_risk", "LOW")
                sc           = result.get("supply_chain", {})
                sc_risk      = sc.get("risk", "LOW")

                stockout_style = "success" if stockout_r == "LOW" else "critical"
                sc_style       = _risk_style(sc_risk)

                outcome_html = (
                    f'<div class="outcome-card">'
                    f'<div class="outcome-card-title">◇ SIMULATION RESULT</div>'
                    f'<div class="metric-row">'
                    f'<span class="metric-row-label">Projected Demand</span>'
                    f'<span class="metric-row-value" style="color:#3B82F6">{proj_demand} units</span>'
                    f'</div>'
                    f'<div class="metric-row">'
                    f'<span class="metric-row-label">Discounted Price</span>'
                    f'<span class="metric-row-value">₹{disc_price:,.2f}</span>'
                    f'</div>'
                    f'<div class="metric-row">'
                    f'<span class="metric-row-label">Expected Revenue</span>'
                    f'<span class="metric-row-value">₹{exp_revenue:,.0f}</span>'
                    f'</div>'
                    f'<div class="metric-row">'
                    f'<span class="metric-row-label">Expected Profit</span>'
                    f'<span class="metric-row-value" style="color:#22C55E">₹{exp_profit:,.0f}</span>'
                    f'</div>'
                    f'<div class="metric-row">'
                    f'<span class="metric-row-label">Stockout Risk</span>'
                    f'<span class="metric-row-value">'
                    f'<span class="badge badge-{stockout_style}">{stockout_r}</span>'
                    f'</span>'
                    f'</div>'
                    f'<div class="metric-row">'
                    f'<span class="metric-row-label">Supply Chain Risk</span>'
                    f'<span class="metric-row-value">'
                    f'<span class="badge badge-{sc_style}">{sc_risk}</span>'
                    f'</span>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(outcome_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Simulation failed: {e}. Is the FastAPI backend running?")
        else:
            placeholder_html = (
                '<div class="dash-card" style="text-align:center;padding:2rem 1rem">'
                '<div style="font-size:2rem;margin-bottom:0.5rem">◇</div>'
                '<div style="font-size:0.8rem;color:var(--text-secondary)">'
                'Adjust the sliders and click<br><strong>Run Simulation</strong> to see results.'
                '</div>'
                '</div>'
            )
            st.markdown(placeholder_html, unsafe_allow_html=True)



def _risk_style(category: str) -> str:
    return {"LOW": "success", "MEDIUM": "warning", "HIGH": "critical"}.get(category, "neutral")
