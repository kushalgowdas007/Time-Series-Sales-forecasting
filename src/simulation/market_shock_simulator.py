def simulate_market_shock(
        forecast,
        supplier_delay_days):

    risk = "LOW"

    if supplier_delay_days > 7:
        risk = "HIGH"

    safety_stock = (
        forecast * 0.5
    )

    return {
        "supplier_delay": supplier_delay_days,
        "risk": risk,
        "recommended_safety_stock":
            int(safety_stock)
    }


if __name__ == "__main__":

    print(
        simulate_market_shock(
            forecast=100,
            supplier_delay_days=10
        )
    )