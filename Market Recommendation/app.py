import streamlit as st
import pandas as pd
import numpy as np
import pickle
from joblib import load
import market_recomm  # ✅ Import market recommendation logic

# Load Market Model
market_model = load("models/market_model.pkl")

#  Streamlit UI
st.title("PlantOra🌿 Market & Price Prediction System🌾")

# ======================== 📍 Market Selling Area Recommendation =========================
st.header("📍 Find the Best Market for Your Crops")

# ✅ Add Location Dropdown
location = st.selectbox(
    "Select Location (City)",
    ["Delhi", "Mumbai", "Chennai", "Bengaluru", "Kolkata", "Hyderabad", "Ahmedabad",
     "Pune", "Lucknow", "Jaipur", "Bhopal", "Raipur", "Chandigarh", "Varanasi",
     "Dehradun", "Vishakapatnam", "Indore", "Nagpur", "Guwahati"]
)

# Market Recommendation Inputs
demand = st.number_input("Enter Demand", min_value=1000, max_value=10000, step=500)
price = st.number_input("Enter Price (per unit)", min_value=50, max_value=200, step=10)
logistics = st.number_input("Enter Logistics Cost", min_value=5, max_value=50, step=5)

if st.button("🔍 Recommend Market"):
    recommended_market = market_recomm.recommend_market(demand, price, logistics, location)  # ✅ Use function

    # ✅ **Check if the function returned a dictionary**
    if isinstance(recommended_market, dict):
        # ✅ **Display results in a stylish table**
        st.markdown(
            f"""
            <div style="border-radius: 10px; padding: 15px; background-color: #0000f;">
                <h3>📍 <b>{recommended_market["Market"]}</b> - {recommended_market["City"]}</h3>
                <ul>
                    <li><b>📈 Demand:</b> {recommended_market["Demand"]}</li>
                    <li><b>💰 Price per Unit:</b> {recommended_market["Price"]}</li>
                    <li><b>🚛 Logistics Cost:</b> {recommended_market["Logistics Cost"]}</li>
                    <li><b>🌍 Region:</b> {recommended_market["Region"]}</li>
                    <li><b>⭐ Score:</b> {recommended_market["Score"]}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(recommended_market)  # Display error message if no market found
# ======================== 📈 **Crop Price Prediction** =========================
st.subheader("📈 Crop Price Prediction")

# 🌾 **Crop Selection**
crop = st.selectbox(
    "Select Crop", 
    ["Paddy", "Wheat", "Maize", "Barley", "Gram", "Arhar", "Moong", "Urad", "Cotton", "Jute"]
)

# 📅 **Prediction Range Selection**
prediction_range = st.radio(
    "Select Prediction Duration",
    ["Next 7 Days", "Next 1 Month", "Next 1 Year"]
)

# 🔍 **Predict Prices**
if st.button("📊 Predict Prices"):
    try:
        # Load crop-specific model
        with open(f"models/price_model_{crop}.pkl", "rb") as file:
            crop_price_model = pickle.load(file)
        
        # Determine prediction steps based on user choice
        if prediction_range == "Next 7 Days":
            steps = 7
        elif prediction_range == "Next 1 Month":
            steps = 30
        else:  # Next 1 Year
            steps = 365
        
        # Predict prices
        forecast = crop_price_model.forecast(steps=steps)  # ✅ Crop-specific prediction
        
        # 📉 **Show Predictions**
        st.write(f"📅 **Predicted Prices for {prediction_range} ({crop}):**")
        st.line_chart(forecast)

    except FileNotFoundError:
        st.error(f"❌ No trained model found for {crop}. Please train it first.")
