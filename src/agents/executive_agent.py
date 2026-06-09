def run_executive_agent(
        inventory_result,
        risk_result,
        finance_result):

    recommendations = []

    # Stockout
    if inventory_result["stockout"]["risk"] == "HIGH":
        recommendations.append(
            "Reorder inventory immediately"
        )
    else:
        recommendations.append(
            "No reorder required"
        )

    # Excess inventory
    if inventory_result["optimization"]["excess_inventory"] > 20:
        recommendations.append(
            "Reduce excess inventory"
        )

    # Demand spike
    if risk_result["spike"]["spike_detected"]:
        recommendations.append(
            "Monitor demand spike closely"
        )

    # Product risk
    if risk_result["risk"]["category"] == "HIGH":
        recommendations.append(
            "High risk product detected"
        )

    # Profit
    if finance_result["profit"] > 0:
        recommendations.append(
            "Business remains profitable"
        )

    return {
        "status": "HEALTHY",
        "recommendations": recommendations
    }