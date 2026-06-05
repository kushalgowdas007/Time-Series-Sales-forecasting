def forecast_revenue(
        forecast_units,
        selling_price,
        profit_margin):

    revenue = (
        forecast_units *
        selling_price
    )

    profit = (
        revenue *
        profit_margin
    )

    return {
        "revenue": revenue,
        "profit": profit
    }


if __name__ == "__main__":

    print(
        forecast_revenue(
            forecast_units=500,
            selling_price=100,
            profit_margin=0.30
        )
    )