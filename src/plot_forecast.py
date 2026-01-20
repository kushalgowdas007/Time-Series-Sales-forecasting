import pandas as pd
import pickle
import matplotlib.pyplot as plt

print("Starting plot script...")

# Load processed data
df = pd.read_csv("data/processed/cleaned_sales.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

print("Loaded historical data")

# Load trained model
with open("models/arima_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Loaded ARIMA model")

# Forecast next 7 periods
forecast = model.forecast(steps=7)

print("Forecast generated")
print(forecast)

# Create plot
plt.figure()
plt.plot(df.index, df['sales'], label="Actual Sales")
plt.plot(forecast.index, forecast.values, label="Forecasted Demand")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Retail Demand Forecast")
plt.legend()

print("Displaying plot window...")
plt.show(block=True)
