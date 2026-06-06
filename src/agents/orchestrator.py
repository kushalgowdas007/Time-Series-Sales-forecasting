from src.agents.demand_agent import run_demand_agent
from src.agents.inventory_agent import run_inventory_agent
from src.agents.risk_agent import run_risk_agent
from src.agents.finance_agent import run_finance_agent


inventory = 100

price = 120

margin = 0.25

sales_history = [
    30,32,31,34,33,
    80,90,100,95,110
]

# Demand Agent
demand = run_demand_agent()

forecast_value = demand["forecast_value"]

# Inventory Agent
inventory_result = run_inventory_agent(
    inventory,
    forecast_value
)

# Risk Agent
risk_result = run_risk_agent(
    sales_history,
    inventory,
    forecast_value
)

# Finance Agent
finance_result = run_finance_agent(
    forecast_value,
    price,
    margin
)

print("\n===== AI RETAIL CONTROL TOWER =====")

print("\nForecast")
print(demand)

print("\nInventory")
print(inventory_result)

print("\nRisk")
print(risk_result)

print("\nFinance")
print(finance_result)