import streamlit as st
import pandas as pd
import time
from sensor_simulation import generate_data

# Page Configuration
st.set_page_config(page_title="Livestock Health Monitoring", layout="wide")

# Title & Description
st.title("🐄 Livestock Health Monitoring System")
st.write("Monitor real-time health status of livestock using AI-powered anomaly detection.")

# Display Live Data
placeholder = st.empty()

while True:
    # Generate new data
    sensor_data, anomalies = generate_data()

    # Load updated dataset
    try:
        df = pd.read_csv("data/livestock_health_data.csv")
    except FileNotFoundError:
        st.error("No data found! Run `sensor_simulation.py` first.")
        break

    # Show latest data
    with placeholder.container():
        st.subheader("📊 Latest Sensor Readings")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("🌡 Temperature", f"{sensor_data['temperature']} °C", 
                    "⚠️" if "Temperature" in sensor_data['status'] else "✅")
        col2.metric("❤️ Heart Rate", f"{sensor_data['heart_rate']} bpm", 
                    "⚠️" if "Heart Rate" in sensor_data['status'] else "✅")
        col3.metric("🏃 Activity Level", sensor_data["activity"], 
                    "⚠️" if "Activity Level" in sensor_data['status'] else "✅")
        col4.metric("📅 Last Update", sensor_data["timestamp"])

        # Show Data Table
        st.subheader("📜 Historical Data")
        st.dataframe(df.tail(10))  # Show last 10 entries

        # Show Alerts
        if anomalies:
            st.warning("🚨 ALERTS DETECTED!")
            for alert in anomalies:
                st.error(alert)

    time.sleep(5)  # Update every 5 seconds
