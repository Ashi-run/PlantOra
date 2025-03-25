import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Define model path
import os
from joblib import dump, load
model = load("crop_rotation_model.pkl")
dump(model, "crop_rotation_model.pkl")


# Get the current directory of the script
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct a relative path to the model file
# model_path = os.path.join(BASE_DIR, "crop_rotation_model.pkl")

# Load model, encoder, and scaler
#with open(model_path, "rb") as f:
#    model, encoder, scaler, feature_names = pickle.load(f)

# Streamlit UI
st.title("🌱 Crop Rotation Planner")

# Input fields
previous_crop = st.selectbox("Previous Crop", ["Wheat", "Rice", "Maize", "Soybean", "Barley", "Millet", "Sugarcane", "Cotton", "Potato", "Tomato"])
soil_nitrogen = st.slider("Soil Nitrogen Level", 10, 50, 30)
soil_ph = st.slider("Soil pH Level", 5.5, 7.5, 6.5)
rainfall = st.slider("Rainfall (mm)", 300, 1500, 800)
temperature = st.slider("Temperature (°C)", 15, 35, 25)

# Encode previous crop
encoded_crop = encoder.transform([[previous_crop]])

# Convert numerical features to NumPy array and reshape
numerical_features = np.array([soil_nitrogen, soil_ph, rainfall, temperature]).reshape(1, -1)

# Standardize numerical features
scaled_numerical = scaler.transform(numerical_features)

# Combine scaled numerical and encoded categorical features
input_data = np.hstack([scaled_numerical, encoded_crop])

# Predict next crop
recommended_crop = model.predict(input_data)[0]

# Display output
st.success(f"✅ Recommended Next Crop: **{recommended_crop}**")
