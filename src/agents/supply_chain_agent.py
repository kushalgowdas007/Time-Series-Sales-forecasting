from src.multi_store.inventory_balancer import balance_inventory

def run_supply_chain_agent():

    result = balance_inventory(
        store_a_inventory=200,
        store_b_inventory=20,
        store_a_forecast=50,
        store_b_forecast=100
    )

    return result