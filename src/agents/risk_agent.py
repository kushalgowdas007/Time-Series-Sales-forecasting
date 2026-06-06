from src.intelligence.demand_spike_detector import detect_spike
from src.intelligence.product_risk_classifier import classify_risk
from src.inventory.safety_stock_optimizer import calculate_safety_stock

import pandas as pd

def run_risk_agent(
        sales_history,
        inventory,
        forecast):

    sales = pd.Series(sales_history)

    spike = detect_spike(sales)

    safety_stock = calculate_safety_stock(
        sales
    )

    risk = classify_risk(
        inventory,
        forecast,
        confidence=99
    )

    return {
        "spike": spike,
        "safety_stock": safety_stock,
        "risk": risk
    }