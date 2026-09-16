# Sales Prediction Model

A small end-to-end machine learning project that predicts store sales using historical sales data, lag features, and a Random Forest regressor.

## Overview

This project walks through a full ML workflow on daily sales data from 10 stores over a 2-year period:

1. **Exploratory Data Analysis** — understanding the dataset structure and trends
2. **Feature Engineering** — building lag and rolling-average features from past sales
3. **Model Training** — training a Random Forest regressor to predict sales
4. **Prediction** — using the trained model to forecast future sales for a given store

## Dataset

`sales_sample.csv` contains daily records with the following columns:

| Column | Description |
|---|---|
| `Date` | Date of the record |
| `Store` | Store identifier (S01–S10) |
| `StoreType` | Store category (A/B/C) |
| `Sales` | Sales value for that day (target variable) |
| `Customers` | Number of customers that day |
| `Promo` | Whether a promotion was running (0/1) |
| `Holiday` | Whether it was a holiday (0/1) |
| `DayOfWeek` | Day of the week |
| `CompetitionDistance` | Distance to nearest competitor |

## Project Structure

```
├── sales_sample.csv           # Raw dataset
├── Sales_pred.ipynb           # Initial exploratory data analysis
├── Sales_Pred_Test2.ipynb     # Extended feature engineering & time-based validation
├── train.py                   # Trains the model and saves it to disk
├── predict.py                 # Loads the saved model and predicts sales for a store
└── requirements.txt           # Python dependencies
```

## How It Works

**`train.py`**
- Loads and sorts the sales data by store and date
- Builds lag features (previous period's sales, previous-to-previous period's sales) and a 3-period rolling average
- Trains a `RandomForestRegressor` (200 trees) on these features
- Evaluates the model using MAE and MSE
- Saves the trained model as `sales_forecast_model.pkl`

**`predict.py`**
- Loads the saved model
- Takes a store ID as input
- Pulls that store's most recent records
- Predicts the next period's sales for that store

**`Sales_Pred_Test2.ipynb`**
- A deeper exploration that adds week-of-year and weekend flags, multiple lag windows (1-day and 7-day), 7-day and 30-day rolling averages, one-hot encoded store types, and a **time-based train/test split** (training on all data up to a cutoff date, testing on the final 90 days) — the more rigorous way to validate a forecasting model, since it avoids testing on data that overlaps in time with training.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python train.py
```

This will generate `sales_forecast_model.pkl` in the project folder.

Run a prediction:

```bash
python predict.py
```

You'll be prompted to enter a store ID (e.g., `S01`), and the script will output a predicted sales value based on that store's most recent history.

## Notes & Limitations

- The dataset is at **daily granularity** — the lag features represent the previous day(s) of sales rather than previous months, despite similar-sounding variable names in earlier versions of this project.
- The trained model (`.pkl` file) is not included in this repository due to its size. Run `train.py` to generate it locally.
- This is a learning/portfolio project using synthetic sample data, not a production forecasting system.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (RandomForestRegressor)
- joblib (model persistence)
- matplotlib, seaborn (visualization)
