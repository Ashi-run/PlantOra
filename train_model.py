import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Define dataset path
dataset_path = r"C:\Users\LENOVO\Downloads\AgriSens-master\AgriSens-master\AgriSens-web-app\crop_rotation_planner\data\large_crop_rotation_data.csv"

# Check if dataset exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"❌ Dataset not found at: {dataset_path}")

# Load dataset
df = pd.read_csv(dataset_path)

# Check if all required columns exist
required_columns = ["previous_crop", "soil_nitrogen", "soil_ph", "rainfall", "temperature", "recommended_next_crop"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"❌ Missing columns in dataset: {missing_columns}")

# One-hot encode "previous_crop"
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded_crops = encoder.fit_transform(df[['previous_crop']])
encoded_crop_df = pd.DataFrame(encoded_crops, columns=encoder.get_feature_names_out(['previous_crop']))

# Merge encoded data
df = pd.concat([df, encoded_crop_df], axis=1)
df.drop(columns=['previous_crop'], inplace=True)

# Extract Features (X) and Target (y)
X = df.drop(columns=['recommended_next_crop'])
y = df['recommended_next_crop']

# **Normalize numerical features for better accuracy**
scaler = StandardScaler()
X[["soil_nitrogen", "soil_ph", "rainfall", "temperature"]] = scaler.fit_transform(X[["soil_nitrogen", "soil_ph", "rainfall", "temperature"]])

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Define model save path
model_path = r"C:\Users\LENOVO\Downloads\AgriSens-master\AgriSens-master\AgriSens-web-app\crop_rotation_planner\crop_rotation_model.pkl"

# Save model, encoder, and scaler
with open(model_path, "wb") as f:
    pickle.dump((model, encoder, scaler, X_train.columns.tolist()), f)

print(f"✅ Model trained and saved successfully at: {model_path}")
