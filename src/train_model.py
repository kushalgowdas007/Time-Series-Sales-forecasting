import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA

df = pd.read_csv("data/processed/cleaned_sales.csv")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

model = ARIMA(df['sales'], order=(1,1,1))
model_fit = model.fit()

with open("models/arima_model.pkl", "wb") as f:
    pickle.dump(model_fit, f)

print("Model trained and saved successfully!")
