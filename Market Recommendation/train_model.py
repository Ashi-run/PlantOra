import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# 🚀 Load Market Data
market_data = pd.read_csv("data/market_data.csv")
market_data.columns = market_data.columns.str.strip()  # Clean column names

# ✅ Check if required columns exist
required_columns = {'Market', 'Demand', 'Price', 'Logistics', 'Region'}
if not required_columns.issubset(market_data.columns):
    raise ValueError(f"❌ Missing columns in market_data.csv! Required: {required_columns}")

# ✅ Select Features
X = market_data[['Demand', 'Price', 'Logistics']]
y = market_data.index  # Market Index

# ✅ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train Market Recommendation Model
market_model = RandomForestRegressor(n_estimators=200, random_state=42)
market_model.fit(X_train, y_train)

# ✅ Save Market Model
dump(market_model, "models/market_model.pkl")
print("✅ Market Recommendation Model Trained and Saved!")


# 🚀 Load Crop Price Data
crop_data = pd.read_csv("data/crop_prices.csv", encoding='ISO-8859-1')
crop_data.columns = crop_data.columns.str.strip()

# ✅ Strip Extra Spaces from Crop Names
crop_data["Crop"] = crop_data["Crop"].str.strip()

# ✅ Convert Wide Format to Long Format
crop_data_long = crop_data.melt(id_vars=["Crop"], var_name="Year", value_name="Price")

# ✅ Clean Year Column (convert '1990-91' to 1990)
crop_data_long["Year"] = crop_data_long["Year"].apply(lambda x: int(x.split("-")[0]) if "-" in x else int(x))

# ✅ Train ARIMA Model for Each Crop
for crop_name in crop_data_long["Crop"].unique():
    print(f"🚀 Training ARIMA Model for {crop_name}...")

    # Extract time series
    crop_series = crop_data_long[crop_data_long["Crop"] == crop_name][["Year", "Price"]].set_index("Year")

    # ✅ Convert to numeric & drop NaN
    crop_series["Price"] = pd.to_numeric(crop_series["Price"], errors="coerce").dropna()

    # ✅ Skip if missing data
    if crop_series.empty:
        print(f"⚠️ Skipping {crop_name}: No valid price data available.")
        continue

    # ✅ Skip if less than 10 years of data
    if len(crop_series) < 10:
        print(f"⚠️ Skipping {crop_name}: Not enough data for ARIMA (Need ≥10 years, Found {len(crop_series)}).")
        continue

    # ✅ Skip if no variation in price
    if crop_series["Price"].nunique() == 1:
        print(f"⚠️ Skipping {crop_name}: No variation in price data (All values are {crop_series['Price'].iloc[0]}).")
        continue

    try:
        # Train ARIMA Model
        arima_model = ARIMA(crop_series, order=(2, 1, 2))
        arima_model_fit = arima_model.fit()

        # ✅ Save Model with Cleaned Name
        model_filename = f"models/{crop_name}_price_model.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(arima_model_fit, file)

        print(f"✅ Model Trained and Saved: {model_filename}!")
    except Exception as e:
        print(f"❌ ERROR Training {crop_name}: {str(e)}")

print("✅ All Crop Models Processed Successfully!")

