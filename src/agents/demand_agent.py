from src.forecasting.forecast import get_forecast

def run_demand_agent():

    forecast = get_forecast()

    forecast_value = int(forecast.iloc[0])

    return {
        "forecast": forecast,
        "forecast_value": forecast_value
    }