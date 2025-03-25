import pandas as pd
import random

# Define crops, nitrogen levels, pH, rainfall, and temperature ranges
crops = ["Wheat", "Rice", "Maize", "Soybean", "Barley", "Millet", "Sugarcane", "Cotton", "Potato", "Tomato"]
crop_rotation_map = {
    "Wheat": "Rice", "Rice": "Maize", "Maize": "Soybean", "Soybean": "Barley",
    "Barley": "Wheat", "Millet": "Cotton", "Sugarcane": "Potato", "Cotton": "Tomato", 
    "Potato": "Wheat", "Tomato": "Rice"
}

data = []

for _ in range(10000):  # 10,000 rows
    previous_crop = random.choice(crops)
    soil_nitrogen = random.randint(10, 50)
    soil_ph = round(random.uniform(5.5, 7.5), 1)
    rainfall = random.randint(300, 1500)
    temperature = random.randint(15, 35)
    recommended_next_crop = crop_rotation_map.get(previous_crop, random.choice(crops))
    
    data.append([previous_crop, soil_nitrogen, soil_ph, rainfall, temperature, recommended_next_crop])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["previous_crop", "soil_nitrogen", "soil_ph", "rainfall", "temperature", "recommended_next_crop"])

# Save to CSV
df.to_csv("large_crop_rotation_data.csv", index=False)
print("✅ Large dataset created: 'large_crop_rotation_data.csv'")
