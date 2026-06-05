def classify_risk(
        inventory,
        forecast):

    risk_score = 0

    if forecast > inventory:
        risk_score += 50

    demand_ratio = forecast / max(
        inventory,
        1
    )

    risk_score += min(
        int(demand_ratio * 25),
        50
    )

    if risk_score >= 75:
        category = "HIGH"

    elif risk_score >= 40:
        category = "MEDIUM"

    else:
        category = "LOW"

    return {
        "risk_score": risk_score,
        "category": category
    }


if __name__ == "__main__":

    print(
        classify_risk(
            inventory=100,
            forecast=200
        )
    )