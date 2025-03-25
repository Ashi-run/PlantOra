import numpy as np
import pandas as pd
# Load Data
df = pd.read_csv("data/crop_prices.csv", encoding="ISO-8859-1")

# Simulate new data by adding random noise
for col in df.columns[1:]:  # Skip 'Crop' column
    df[col] = df[col] * (1 + np.random.uniform(-0.1, 0.1, len(df)))  # ±10% variation

# Save augmented dataset
df.to_csv("data/crop_prices_augmented.csv", index=False)
print("✅ Synthetic data generated with random variations!")
