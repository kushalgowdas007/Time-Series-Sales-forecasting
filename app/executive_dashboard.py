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
        "http://127.0.0.1:8000/forecast"
    )

    data = response.json()

    forecast = data["forecast"]
    revenue = data["revenue"]
    profit = data["profit"]
    risk = data["risk"]
    status = data["status"]

except:

    st.error(
        "FastAPI server not running."
    )

    st.stop()

# ==========================================
# SAMPLE VALUES
# (Replace later with API values)
# ==========================================

inventory = 100

recommended_stock = 38

safety_stock = 53

reorder_quantity = 0

confidence = 99

# ==========================================
# KPI SECTION
# ==========================================

st.subheader("📊 Executive Control Tower")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Forecast",
        forecast
    )

with col2:
    st.metric(
        "Revenue",
        f"₹{revenue}"
    )

with col3:
    st.metric(
        "Profit",
        f"₹{profit}"
    )

with col4:
    st.metric(
        "Inventory",
        inventory
    )

with col5:
    st.metric(
        "Safety Stock",
        safety_stock
    )

# ==========================================
# BUSINESS HEALTH
# ==========================================

st.divider()

st.subheader("🏢 Business Health")

if status == "HEALTHY":
    st.success(status)
else:
    st.error(status)

# ==========================================
# RISK GAUGE
# ==========================================

st.divider()

st.subheader("⚠ Business Risk Indicator")

risk_score = 20

if risk == "LOW":
    risk_score = 20

elif risk == "MEDIUM":
    risk_score = 60

else:
    risk_score = 90

gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={"text": "Risk Score"}
    )
)

st.plotly_chart(
    gauge,
    use_container_width=True
)

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

# ==========================================
# INVENTORY CHART
# ==========================================

inventory_chart = go.Figure()

inventory_chart.add_bar(

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
    inventory_chart,
    use_container_width=True
)

# ==========================================
# AUTONOMOUS REPLENISHMENT
# ==========================================

st.divider()

st.subheader("🤖 Autonomous Replenishment")

replenishment_df = pd.DataFrame({

    "Metric": [

        "Current Inventory",

        "Required Stock",

        "Reorder Quantity",

        "Confidence"

    ],

    "Value": [

        inventory,

        85,

        reorder_quantity,

        f"{confidence}%"

    ]

})

st.dataframe(
    replenishment_df,
    use_container_width=True
)

if reorder_quantity == 0:

    st.success(
        "No reorder required."
    )

else:

    st.warning(
        f"Reorder {reorder_quantity} units immediately."
    )

# ==========================================
# DIGITAL TWIN
# ==========================================

st.divider()

st.subheader("🧪 Digital Twin Simulation")

digital_twin_df = pd.DataFrame({

    "Metric": [

        "Inventory Before",

        "Inventory After",

        "Inventory Cost Before",

        "Inventory Cost After"

    ],

    "Value": [

        100,

        120,

        1000,

        1200

    ]

})

st.dataframe(
    digital_twin_df,
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

for i, rec in enumerate(
        recommendations,
        start=1):

    st.success(
        f"{i}. {rec}"
    )

# ==========================================
# SUPPLY CHAIN RISK
# ==========================================

st.divider()

st.subheader("🚚 Supply Chain Risk")

st.error(
    "Supplier Delay = 10 Days | Risk = HIGH"
)

# ==========================================
# DEMAND INTELLIGENCE
# ==========================================

st.divider()

st.subheader("📈 Demand Intelligence")

st.warning(
    "Demand Spike Detected | Spike Ratio = 2.97"
)

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Retail Decision Intelligence Platform v1.0"
)