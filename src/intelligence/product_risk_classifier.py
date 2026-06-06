def classify_risk(
        inventory,
        forecast,
        confidence):

    score = 0

    if forecast > inventory:
        score += 40

    if confidence < 70:
        score += 30

    demand_ratio = forecast / max(
        inventory,
        1
    )

    score += min(
        int(demand_ratio * 20),
        30
    )

    if score >= 70:
        category = "HIGH"

    elif score >= 40:
        category = "MEDIUM"

    else:
        category = "LOW"

    return {
        "score": score,
        "category": category
    }