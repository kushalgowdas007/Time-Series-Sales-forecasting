def optimize_inventory(current_stock, forecast_demand):

    recommended_stock = int(forecast_demand * 1.2)

    excess_inventory = max(
        current_stock - recommended_stock,
        0
    )

    return {
        "recommended_stock": recommended_stock,
        "excess_inventory": excess_inventory
    }


if __name__ == "__main__":
    print(optimize_inventory(300, 50))