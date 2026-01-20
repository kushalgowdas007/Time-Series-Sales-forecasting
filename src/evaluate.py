import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("data/processed/cleaned_sales.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Train-test split
train = df['sales'][:-3]
test = df['sales'][-3:]

# Load trained model
with open("models/arima_model.pkl", "rb") as f:
    model = pickle.load(f)

# Forecast
predictions = model.forecast(steps=len(test))

# Evaluation
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print("Model Evaluation Results:")
print("MAE:", mae)
print("RMSE:", rmse)
