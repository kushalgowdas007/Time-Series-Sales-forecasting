import pandas as pd

from src.intelligence.demand_spike_detector import detect_spike
from src.intelligence.forecast_failure_monitor import forecast_confidence
from src.inventory.safety_stock_optimizer import calculate_safety_stock
from src.intelligence.product_risk_classifier import classify_risk

sales = pd.Series([
    30, 32, 31, 34, 33,
    80, 90, 100, 95, 110
])

print("\n===== DEMAND SPIKE =====")
print(detect_spike(sales))

print("\n===== FORECAST CONFIDENCE =====")
print(
    forecast_confidence(
        [30, 35, 32, 36],
        [31, 34, 33, 35]
    )
)

print("\n===== SAFETY STOCK =====")
print(
    calculate_safety_stock(sales)
)



risk = classify_risk(
    inventory=100,
    forecast=32,
    confidence=99
)

print("\n===== PRODUCT RISK =====")
print(risk)