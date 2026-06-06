from src.inventory.stockout_predictor import predict_stockout
from src.inventory.inventory_optimizer import optimize_inventory
from src.inventory.replenishment_engine import recommend_replenishment

def run_inventory_agent(
        inventory,
        forecast):

    stockout = predict_stockout(
        inventory,
        forecast
    )

    optimization = optimize_inventory(
        inventory,
        forecast
    )

    replenishment = recommend_replenishment(
        inventory,
        forecast
    )

    return {
        "stockout": stockout,
        "optimization": optimization,
        "replenishment": replenishment
    }