import os
import pickle


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


# ---------------------------------------------------------------------------
# ARIMA model path
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "forecasting",
    "arima_model.pkl"
)


# ---------------------------------------------------------------------------
# Forecast function
# ---------------------------------------------------------------------------

def get_forecast():
    """
    Load the trained ARIMA model and generate a 7-step forecast.
    """

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    forecast = model.forecast(steps=7)

    return forecast


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    forecast = get_forecast()

    forecast_value = int(forecast.iloc[0])

    print("\n===== FORECAST =====")
    print(forecast)

    print("\nFirst Forecast Value:")
    print(forecast_value)