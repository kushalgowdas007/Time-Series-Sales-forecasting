def balance_inventory(
        store_a_inventory,
        store_b_inventory,
        store_a_forecast,
        store_b_forecast):

    excess = max(
        store_a_inventory -
        store_a_forecast,
        0
    )

    shortage = max(
        store_b_forecast -
        store_b_inventory,
        0
    )

    transfer = min(
        excess,
        shortage
    )

    return {
        "transfer_units": transfer
    }