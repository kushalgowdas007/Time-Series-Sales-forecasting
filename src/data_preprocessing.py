import pandas as pd
import os

RAW_PATH = "data/raw/sales_data.csv"
PROCESSED_PATH = "data/processed/cleaned_sales.csv"

print("=== PREPROCESSING STARTED ===")

# Check file exists
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError("sales_data.csv not found in data/raw/")

# Try reading with comma
df = pd.read_csv(RAW_PATH)

# If still empty, try semicolon (Excel issue)
if df.empty:
    print("⚠️ Comma read failed, trying semicolon separator...")
    df = pd.read_csv(RAW_PATH, sep=';')

if df.empty:
    raise ValueError("❌ sales_data.csv is still empty. File is not a valid CSV.")

print("✅ Raw data loaded")
print("Shape:", df.shape)
print(df.head())

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.dropna()

os.makedirs("data/processed", exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print("✅ Cleaned data saved successfully")
print("=== PREPROCESSING COMPLETED ===")
