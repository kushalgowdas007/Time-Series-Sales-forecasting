import numpy as np


def calculate_safety_stock(
        demand_history):

    std_dev = np.std(
        demand_history
    )

    safety_stock = round(
        1.65 * std_dev
    )

    return {
        "safety_stock": safety_stock
    }


if __name__ == "__main__":

    history = [
        30,35,40,
        28,33,45
    ]

    print(
        calculate_safety_stock(
            history
        )
    )

    