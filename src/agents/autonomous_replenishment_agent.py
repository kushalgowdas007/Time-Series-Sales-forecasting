from datetime import datetime


def run_autonomous_replenishment(
        inventory,
        forecast,
        safety_stock,
        lead_time_days=5):

    required_stock = (
        forecast +
        safety_stock
    )

    reorder_quantity = max(
        required_stock - inventory,
        0
    )

    if reorder_quantity > 0:
        action = "REORDER"
        confidence = 95
    else:
        action = "NO_ACTION"
        confidence = 99

    return {
        "action": action,
        "required_stock": required_stock,
        "current_inventory": inventory,
        "reorder_quantity": reorder_quantity,
        "recommended_order_date":
            datetime.today().strftime("%Y-%m-%d"),
        "lead_time_days": lead_time_days,
        "confidence": confidence
    }


if __name__ == "__main__":

    result = run_autonomous_replenishment(
        inventory=40,
        forecast=80,
        safety_stock=20
    )

    print(result)