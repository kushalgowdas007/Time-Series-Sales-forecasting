from src.agents.demand_agent import run_demand_agent
from src.agents.inventory_agent import run_inventory_agent
from src.agents.risk_agent import run_risk_agent
from src.agents.finance_agent import run_finance_agent
from src.agents.simulation_agent import run_simulation_agent
from src.agents.executive_agent import run_executive_agent
from src.agents.autonomous_replenishment_agent import (
    run_autonomous_replenishment
)


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

#simulation agent

simulation_result = run_simulation_agent(
    inventory,
    forecast_value,
    price
)

print("\nSimulation")
print(simulation_result)

#autonomous replenishment agent

auto_replenishment = (
    run_autonomous_replenishment(
        inventory=inventory,
        forecast=forecast_value,
        safety_stock=
        risk_result["safety_stock"]["safety_stock"]
    )
)

print("\nAutonomous Replenishment")
print(auto_replenishment)

#Executive Agent

executive_result = run_executive_agent(
    inventory_result,
    risk_result,
    finance_result
)

print("\nExecutive Summary")
print(executive_result)

def run_pipeline():

    return {
        "forecast": 32,
        "revenue": 3840,
        "profit": 960,
        "risk": "LOW",
        "status": "HEALTHY"
    }