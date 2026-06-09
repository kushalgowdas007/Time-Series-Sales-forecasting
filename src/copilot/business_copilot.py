def generate_summary(
        forecast,
        revenue,
        profit,
        risk,
        inventory,
        safety_stock,
        status):

    summary = []

    # Forecast

    summary.append(
        f"Demand forecast is {forecast} units."
    )

    # Revenue

    summary.append(
        f"Expected revenue is ₹{revenue}."
    )

    # Profit

    summary.append(
        f"Expected profit is ₹{profit}."
    )

    # Risk

    if risk == "LOW":

        summary.append(
            "Business risk remains low."
        )

    elif risk == "MEDIUM":

        summary.append(
            "Business risk requires monitoring."
        )

    else:

        summary.append(
            "Business risk is high."
        )

    # Inventory

    if inventory > forecast:

        summary.append(
            "Inventory is sufficient for expected demand."
        )

    else:

        summary.append(
            "Inventory may not meet demand."
        )

    # Safety stock

    summary.append(
        f"Recommended safety stock is {safety_stock} units."
    )

    # Executive recommendation

    if inventory > forecast:

        summary.append(
            "Recommendation: Reduce excess inventory."
        )

    else:

        summary.append(
            "Recommendation: Replenish inventory."
        )

    summary.append(
        f"Overall Business Health: {status}"
    )

    return "\n\n".join(summary)