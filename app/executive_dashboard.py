import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Retail Control Tower",
    layout="wide"
)

st.title("🏢 AI Retail Decision Intelligence Platform")

# ==========================================
# FETCH DATA FROM FASTAPI
# ==========================================

try:
    response = requests.get(
        "http://127.0.0.1:8000/forecast",
        timeout=5
    )

    data = response.json()

    forecast = data.get("forecast", 0)
    revenue = data.get("revenue", 0)
    profit = data.get("profit", 0)
    risk = data.get("risk", "LOW")
    status = data.get("status", "HEALTHY")

except Exception as e:

    st.error(f"FastAPI Error: {e}")
    st.stop()

# ==========================================
# SAMPLE VALUES
# ==========================================

inventory = 100
recommended_stock = 38
safety_stock = 53
reorder_quantity = 0
confidence = 99

# ==========================================
# KPI CARDS
# ==========================================

st.subheader("📊 Executive Control Tower")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Forecast", forecast)

with col2:
    st.metric("Revenue", f"₹{revenue}")

with col3:
    st.metric("Profit", f"₹{profit}")

with col4:
    st.metric("Inventory", inventory)

with col5:
    st.metric("Safety Stock", safety_stock)

# ==========================================
# BUSINESS HEALTH
# ==========================================

st.divider()

st.subheader("🏢 Business Health")

if status == "HEALTHY":
    st.success(status)
else:
    st.warning(status)

# ==========================================
# RISK GAUGE
# ==========================================

st.divider()

st.subheader("⚠ Business Risk")

risk_score = 20

if risk == "MEDIUM":
    risk_score = 60
elif risk == "HIGH":
    risk_score = 90

fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Risk Score"}
    )
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# INVENTORY HEALTH
# ==========================================

st.divider()

st.subheader("📦 Inventory Health")

inventory_df = pd.DataFrame({
    "Metric": [
        "Current Inventory",
        "Recommended Inventory",
        "Safety Stock"
    ],
    "Value": [
        inventory,
        recommended_stock,
        safety_stock
    ]
})

st.dataframe(
    inventory_df,
    use_container_width=True
)

bar_fig = go.Figure()

bar_fig.add_bar(
    x=[
        "Current",
        "Recommended",
        "Safety Stock"
    ],
    y=[
        inventory,
        recommended_stock,
        safety_stock
    ]
)

st.plotly_chart(
    bar_fig,
    use_container_width=True
)

# ==========================================
# AUTONOMOUS REPLENISHMENT
# ==========================================

st.divider()

st.subheader("🤖 Autonomous Replenishment")

st.info(f"""
Current Inventory: {inventory}

Required Stock: 85

Reorder Quantity: {reorder_quantity}

Confidence: {confidence}%
""")

# ==========================================
# DIGITAL TWIN
# ==========================================

st.divider()

st.subheader("🧪 Digital Twin")

digital_df = pd.DataFrame({
    "Metric": [
        "Inventory Before",
        "Inventory After",
        "Cost Before",
        "Cost After"
    ],
    "Value": [
        100,
        120,
        1000,
        1200
    ]
})

st.dataframe(
    digital_df,
    use_container_width=True
)

# ==========================================
# WHAT IF SIMULATOR
# ==========================================

st.divider()

st.subheader("📈 What-If Simulator")

what_if_df = pd.DataFrame({
    "Metric": [
        "Discount",
        "New Forecast",
        "Expected Revenue"
    ],
    "Value": [
        "10%",
        36,
        3888
    ]
})

st.dataframe(
    what_if_df,
    use_container_width=True
)

# ==========================================
# EXECUTIVE RECOMMENDATIONS
# ==========================================

st.divider()

st.subheader("🧠 Executive Recommendations")

recommendations = [
    "No reorder required",
    "Reduce excess inventory",
    "Monitor demand spike closely",
    "Business remains profitable"
]

for rec in recommendations:
    st.success(rec)

# ==========================================
# SUPPLY CHAIN ALERTS
# ==========================================

st.divider()

st.subheader("🚚 Supply Chain Risk")

st.error(
    "Supplier Delay = 10 Days | Risk = HIGH"
)

# ==========================================
# DEMAND ALERTS
# ==========================================

st.divider()

st.subheader("📈 Demand Intelligence")

st.warning(
    "Demand Spike Detected | Spike Ratio = 2.97"
)

# ==========================================
# BUSINESS COPILOT
# ==========================================

st.divider()

st.subheader("🤖 AI Business Copilot")

summary = f"""
Forecast Demand: {forecast}

Expected Revenue: ₹{revenue}

Expected Profit: ₹{profit}

Risk Level: {risk}

Inventory Available: {inventory}

Safety Stock: {safety_stock}

Business Status: {status}

Recommendation:
Reduce excess inventory and continue monitoring demand spikes.
"""

st.info(summary)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Retail Decision Intelligence Platform v1.0"
)

st.divider()

st.subheader("🧠 MLOps Monitoring")

mlops_df = pd.DataFrame({

    "Metric": [

        "Model Status",
        "MAE",
        "RMSE",
        "Drift Detected",
        "Retraining Action"

    ],

    "Value": [

        "GOOD",
        4.2,
        5.8,
        "YES",
        "RETRAIN_MODEL"

    ]

})

st.dataframe(
    mlops_df,
    use_container_width=True
)

st.success(
    "Model Performance: GOOD"
)

st.warning(
    "Data Drift Detected"
)

st.error(
    "Retraining Recommended"
)