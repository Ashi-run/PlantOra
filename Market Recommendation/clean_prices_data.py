import pandas as pd

# Load the CSV file
file_path = r"C:\Users\LENOVO\Downloads\AgriSens-master\AgriSens-master\market_recommendation\clean_prices_data.csv"

try:
    df = pd.read_csv(file_path)
    print("✅ CSV file loaded successfully!")
    print(df.head())  # Display first few rows
except FileNotFoundError:
    print("❌ ERROR: File not found! Check the file path.")


df.dropna(inplace=True)  # Remove empty rows
df.to_csv("cleaned_prices_data.csv", index=False)  # Save cleaned file
print("✅ Data cleaned and saved successfully!")


# Print Data Info
print(df.head())
print(df.columns)
