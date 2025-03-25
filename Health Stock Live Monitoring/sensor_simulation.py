import random
import pandas as pd
from datetime import datetime

# Define thresholds for anomaly detection
TEMP_THRESHOLD = (38.0, 40.5)  # Normal temperature range in °C
HEART_RATE_THRESHOLD = (60, 120)  # Normal heart rate range (bpm)
ACTIVITY_THRESHOLD = (20, 100)  # Normal activity level range

# Function to simulate sensor data
def get_sensor_data():
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "temperature": round(random.uniform(35.0, 42.0), 2),
        "heart_rate": random.randint(50, 130),
        "activity": random.randint(10, 120),
    }

# Function to check anomalies
def detect_anomalies(data):
    alerts = []
    if not (TEMP_THRESHOLD[0] <= data["temperature"] <= TEMP_THRESHOLD[1]):
        alerts.append(f"Abnormal Temperature: {data['temperature']}°C")

    if not (HEART_RATE_THRESHOLD[0] <= data["heart_rate"] <= HEART_RATE_THRESHOLD[1]):
        alerts.append(f"Abnormal Heart Rate: {data['heart_rate']} bpm")

    if not (ACTIVITY_THRESHOLD[0] <= data["activity"] <= ACTIVITY_THRESHOLD[1]):
        alerts.append(f"Abnormal Activity Level: {data['activity']}")

    return alerts

# Function to save data
def save_data(data):
    df = pd.DataFrame([data])
    try:
        df_existing = pd.read_csv("data/livestock_health_data.csv")
        df = pd.concat([df_existing, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv("data/livestock_health_data.csv", index=False)

# Function to generate and store data
def generate_data():
    sensor_data = get_sensor_data()
    anomalies = detect_anomalies(sensor_data)
    sensor_data["status"] = "Normal" if not anomalies else "; ".join(anomalies)

    save_data(sensor_data)
    return sensor_data, anomalies

# Run this function every time to generate new data
if __name__ == "__main__":
    print("Generating Sensor Data...")
    print(generate_data())
