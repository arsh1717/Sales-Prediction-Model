# -------------------------------------------
# STEP 1: Import libraries
# -------------------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import joblib

# -------------------------------------------
# STEP 2: Load Data
# -------------------------------------------
csv_path = "sales_sample.csv"   # put CSV in same folder as this script

df = pd.read_csv(csv_path, parse_dates=["Date"])

print("Data Loaded Successfully!")
print(df.head())

# -------------------------------------------
# STEP 3: Prepare Data
# -------------------------------------------
# Extract features from date
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Group by store to calculate lag features
df = df.sort_values(["Store", "Date"])

df["Prev_Month_Sales"] = df.groupby("Store")["Sales"].shift(1)
df["Prev_2_Month_Sales"] = df.groupby("Store")["Sales"].shift(2)
df["Sales_MA_3"] = df.groupby("Store")["Sales"].transform(lambda x: x.rolling(3).mean())

# Drop first rows that become NaN due to lag
df = df.dropna()

print("Feature Engineering Completed!")
print(df.head())

# -------------------------------------------
# STEP 4: Select Features
# -------------------------------------------
X = df[["Month", "Prev_Month_Sales", "Prev_2_Month_Sales", "Sales_MA_3"]]
y = df["Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------
# STEP 5: Train Model
# -------------------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Model training completed!")

# -------------------------------------------
# STEP 6: Test & Performance
# -------------------------------------------
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)

print("Model Performance:")
print("MAE:", mae)
print("MSE:", mse)

# -------------------------------------------
# STEP 7: Save Model
# -------------------------------------------
joblib.dump(model, "sales_forecast_model.pkl")

print("Model saved successfully as sales_forecast_model.pkl!")
