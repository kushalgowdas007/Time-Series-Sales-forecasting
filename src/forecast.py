import pickle

with open("models/arima_model.pkl", "rb") as f:
    model = pickle.load(f)

forecast = model.forecast(steps=7)

print("Next 7 days demand forecast:")
print(forecast)
