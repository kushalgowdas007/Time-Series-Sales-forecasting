def monitor_store_risk(
        inventory,
        forecast):

    if forecast > inventory:
        return "HIGH"

    return "LOW"