import pandas as pd
import joblib

# -------------------------------------------
# Load trained model
# -------------------------------------------
model = joblib.load("sales_forecast_model.pkl")

# -------------------------------------------
# Load original dataset
# -------------------------------------------
df = pd.read_csv("sales_sample.csv", parse_dates=["Date"])

# Sort data
df = df.sort_values(["Store", "Date"])

# Ask user for which store to forecast
store_name = input("Enter Store Name (e.g., Store_A): ")

# Filter store data
store_df = df[df["Store"] == store_name]

# Make sure enough data exists
store_df = store_df.sort_values("Date").tail(3)

if len(store_df) < 3:
    raise ValueError("Not enough historical sales for prediction.")

# -------------------------------------------
# Prepare inputs for prediction
# -------------------------------------------
last_month = store_df.iloc[-1]
prev_month = store_df.iloc[-2]
prev_2_month = store_df.iloc[-3]

future_month = last_month["Date"].month + 1
if future_month == 13:
    future_month = 1

input_data = pd.DataFrame({
    "Month": [future_month],
    "Prev_Month_Sales": [last_month["Sales"]],
    "Prev_2_Month_Sales": [prev_month["Sales"]],
    "Sales_MA_3": [store_df["Sales"].mean()]
})

# -------------------------------------------
# Predict next month sales
# -------------------------------------------
predicted_sales = model.predict(input_data)[0]

print("\n----------------------------------")
print(f"Predicted Sales for {store_name} next month:")
print(f"➡ {predicted_sales:.2f}")
print("----------------------------------\n")
