import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open("crop_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title and description
st.title("🌾 Crop Yield Prediction System")
st.write("Enter the required parameters to predict the expected crop yield.")

# User inputs
rainfall = st.number_input("🌧️ Rainfall (mm)", min_value=0.0, max_value=2000.0, step=1.0)
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
soil_ph = st.number_input("🧪 Soil pH", min_value=3.0, max_value=10.0, step=0.1)
fertilizer_usage = st.number_input("🌱 Fertilizer Usage (kg/ha)", min_value=0.0, max_value=500.0, step=1.0)

# Predict button
if st.button("Predict Yield"):
    input_data = np.array([[rainfall, temperature, soil_ph, fertilizer_usage]])
    predicted_yield = model.predict(input_data)[0]
    
    st.success(f"🌾 Predicted Crop Yield: **{predicted_yield:.2f} kg/ha**")

    # Recommendations
    if predicted_yield < 2000:
        st.warning("⚠️ Low yield detected! Consider improving irrigation and soil fertility.")
    elif predicted_yield > 5000:
        st.success("✅ Excellent yield! Maintain current farming practices.")
