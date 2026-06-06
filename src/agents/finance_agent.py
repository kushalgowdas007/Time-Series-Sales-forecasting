from src.finance.revenue_forecast import forecast_revenue

def run_finance_agent(
        forecast,
        price,
        margin):

    return forecast_revenue(
        forecast,
        price,
        margin
    )