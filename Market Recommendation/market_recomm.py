import pandas as pd

# Load market data
market_data = pd.read_csv("data/market_data.csv")

def recommend_market(demand, price, logistics, location):
    """
    Recommend the best market based on demand, price, logistics cost, and location.
    """

    # ✅ **Filter markets based on selected location**
    filtered_data = market_data[market_data["City"] == location].copy()  # Prevent SettingWithCopyWarning

    # Check if there are markets in the selected location
    if filtered_data.empty:
        return {"Error": f"❌ No market data available for {location}."}  # ✅ Return dictionary instead of string

    # ✅ **Normalize values for fair comparison**
    filtered_data["Norm_Demand"] = filtered_data["Demand"] / filtered_data["Demand"].max()
    filtered_data["Norm_Price"] = filtered_data["Price"] / filtered_data["Price"].max()
    filtered_data["Norm_Logistics"] = 1 - (filtered_data["Logistics"] / filtered_data["Logistics"].max())  # Lower cost = better

    # ✅ **Compute final score**
    filtered_data["Score"] = (
        (filtered_data["Norm_Demand"] * 0.4) +
        (filtered_data["Norm_Price"] * 0.4) +
        (filtered_data["Norm_Logistics"] * 0.2)
    )

    # ✅ **Get the best market in selected location**
    best_market = filtered_data.loc[filtered_data["Score"].idxmax()]

    return {
        "Market": best_market["Market"],
        "City": best_market["City"],
        "Demand": best_market["Demand"],
        "Price": best_market["Price"],
        "Logistics Cost": best_market["Logistics"],
        "Region": best_market["Region"],
        "Score": round(best_market["Score"], 4)  # ✅ Ensures Score is included
    }


# Run recommendation (for testing)
if __name__ == "__main__":
    # ✅ **Fixed test call: Now includes `location` parameter**
    recommended_market = recommend_market(7200, 75, 50, "Mumbai")

    print("✅ Recommended Market:")
    print(recommended_market)
