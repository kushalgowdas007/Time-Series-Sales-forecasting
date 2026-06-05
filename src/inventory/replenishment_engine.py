def recommend_replenishment(
        inventory,
        forecast,
        safety_stock=20):

    reorder_quantity = (
        forecast +
        safety_stock -
        inventory
    )

    reorder_quantity = max(
        reorder_quantity,
        0
    )

    return {
        "reorder_quantity": reorder_quantity,
        "reorder_date": "Today"
    }


if __name__ == "__main__":
    print(
        recommend_replenishment(
            inventory=100,
            forecast=150
        )
    )