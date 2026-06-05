def predict_stockout(current_inventory, forecast_demand):

    if forecast_demand > current_inventory:
        risk = "HIGH"
        shortage = forecast_demand - current_inventory
    else:
        risk = "LOW"
        shortage = 0

    return {
        "risk": risk,
        "shortage": shortage
    }


if __name__ == "__main__":
    result = predict_stockout(100, 130)
    print(result)