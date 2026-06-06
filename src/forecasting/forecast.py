import pickle
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

model_path = os.path.join(
    BASE_DIR,
    "models",
    "forecasting",
    "arima_model.pkl"
)

with open(model_path, "rb") as f:
    model = pickle.load(f)

forecast = model.forecast(steps=7)

forecast_value = int(forecast.iloc[0])

print("\n===== FORECAST =====")
print(forecast)

print("\nFirst Forecast Value:")
print(forecast_value)

from src.inventory.stockout_predictor import predict_stockout
from src.inventory.inventory_optimizer import optimize_inventory
from src.inventory.replenishment_engine import recommend_replenishment
from src.finance.revenue_forecast import forecast_revenue

stockout = predict_stockout(
    current_inventory=100,
    forecast_demand=forecast_value
)

inventory = optimize_inventory(
    current_stock=100,
    forecast_demand=forecast_value
)

reorder = recommend_replenishment(
    inventory=100,
    forecast=forecast_value
)

finance = forecast_revenue(
    forecast_units=forecast_value,
    selling_price=120,
    profit_margin=0.25
)

print("\n===== STOCKOUT ANALYSIS =====")
print(stockout)

print("\n===== INVENTORY OPTIMIZATION =====")
print(inventory)

print("\n===== REPLENISHMENT =====")
print(reorder)

print("\n===== REVENUE FORECAST =====")
print(finance)

def get_forecast():

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    forecast = model.forecast(steps=7)

    return forecast

if __name__ == "__main__":

    forecast = get_forecast()

    print(forecast)