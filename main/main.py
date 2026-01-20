from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features
from src.train_model import train_model

raw_path = "data/raw/sales.csv"
processed_path = "data/processed/cleaned_sales.csv"
model_path = "models/demand_forecast_model.pkl"

df = preprocess_data(raw_path, processed_path)
df = create_features(df)
df.to_csv(processed_path, index=False)

train_model(processed_path, model_path)
print("Model trained successfully!")
