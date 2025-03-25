import pandas as pd

# Define the file path
file_path = r"C:\Users\LENOVO\Downloads\AgriSens-master\AgriSens-master\market_recommendation\clean_prices_data.csv"

try:
    # Load CSV file
    df = pd.read_csv(file_path)
    print("✅ CSV file loaded successfully!")
    
    # Display first few rows
    print(df.head())

    # Clean data (remove empty rows)
    df.dropna(inplace=True)

    # Save cleaned data
    cleaned_file = r"C:\Users\LENOVO\Downloads\AgriSens-master\AgriSens-master\market_recommendation\cleaned_prices_data.csv"
    df.to_csv(cleaned_file, index=False)

    print(f"✅ Cleaned data saved as {cleaned_file}")

except FileNotFoundError:
    print("❌ ERROR: File not found! Check the file path.")
except Exception as e:
    print(f"❌ ERROR: {e}")
