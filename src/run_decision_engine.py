from inventory.stockout_predictor import predict_stockout
from inventory.inventory_optimizer import optimize_inventory
from inventory.replenishment_engine import recommend_replenishment
from intelligence.product_risk_classifier import classify_risk
from finance.revenue_forecast import forecast_revenue
from src.inventory.stockout_predictor import predict_stockout

# Example forecast from ARIMA
forecast = 150

# Example business inputs
current_inventory = 100
selling_price = 120
profit_margin = 0.25

print("\n===== AI RETAIL DECISION ENGINE =====\n")

# Stockout Risk
stockout = predict_stockout(
    current_inventory,
    forecast
)

print("Stockout Analysis")
print(stockout)

# Inventory Optimization
inventory = optimize_inventory(
    current_inventory,
    forecast
)

print("\nInventory Optimization")
print(inventory)

# Replenishment
replenishment = recommend_replenishment(
    current_inventory,
    forecast
)

print("\nReplenishment Recommendation")
print(replenishment)

# Risk Classification
risk = classify_risk(
    current_inventory,
    forecast
)

print("\nRisk Classification")
print(risk)

# Revenue Forecast
finance = forecast_revenue(
    forecast,
    selling_price,
    profit_margin
)

print("\nRevenue Forecast")
print(finance)