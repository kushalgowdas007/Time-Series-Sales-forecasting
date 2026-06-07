def run_what_if(
        forecast,
        price,
        discount_percent):

    demand_increase = (
        discount_percent * 1.5
    )

    new_forecast = int(
        forecast *
        (1 + demand_increase / 100)
    )

    discounted_price = (
        price *
        (1 - discount_percent / 100)
    )

    revenue = (
        new_forecast *
        discounted_price
    )

    return {
        "new_forecast": new_forecast,
        "discounted_price": discounted_price,
        "expected_revenue": revenue
    }


if __name__ == "__main__":

    print(
        run_what_if(
            forecast=100,
            price=120,
            discount_percent=10
        )
    )