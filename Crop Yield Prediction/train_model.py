import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("data/crop_yield_data.csv")

# Selecting features and target variable
X = df[['rainfall', 'temperature', 'soil_ph', 'fertilizer_usage']]
y = df['yield_kg_per_hectare']

# Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Mean Absolute Error: {mae:.2f} kg/ha")

# Save the model
with open("crop_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as 'crop_model.pkl'")
