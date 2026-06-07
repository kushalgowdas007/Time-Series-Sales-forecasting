def run_digital_twin(
        current_inventory,
        forecast,
        increase_percent):

    new_inventory = current_inventory * (
        1 + increase_percent / 100
    )

    stockout_before = forecast > current_inventory

    stockout_after = forecast > new_inventory

    inventory_cost_before = current_inventory * 10

    inventory_cost_after = new_inventory * 10

    return {
        "inventory_before": current_inventory,
        "inventory_after": int(new_inventory),
        "stockout_before": stockout_before,
        "stockout_after": stockout_after,
        "inventory_cost_before": inventory_cost_before,
        "inventory_cost_after": inventory_cost_after
    }


if __name__ == "__main__":

    print(
        run_digital_twin(
            current_inventory=100,
            forecast=150,
            increase_percent=20
        )
    )