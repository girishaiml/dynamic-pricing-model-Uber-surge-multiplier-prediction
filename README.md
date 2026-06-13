# Dynamic Pricing Model — Uber Surge Multiplier Prediction

Predicts real-time surge pricing multipliers using demand, weather, events, competition, and time-of-day signals — the same signals that power pricing engines at Uber, Airbnb, Airlines, and Hotels.

---

## Problem

Every second, ride-sharing platforms answer one question:

> **What price should we charge right now?**

Inputs that drive that answer:

| Signal | Example |
|---|---|
| Demand | 200 ride requests in 5 min |
| Weather | Heavy rain, storm |
| Events | Concert ending nearby |
| Competition | Lyft raised prices 20% |
| Time | Friday 6 PM rush hour |

This model learns the relationship between those signals and the optimal surge multiplier.

---

## Project Structure

```
DYNAMIC_PRICING/
├── generate_data.py      # Synthetic dataset generation (50,000 rides)
├── preprocess.py         # Data cleaning + feature engineering
├── train.py              # Model training + evaluation + plots
├── predict.py            # Inference on new ride requests
├── data/
│   ├── raw_rides.csv
│   └── processed_rides.csv
├── models/
│   ├── surge_model.pkl
│   └── scaler.pkl
├── outputs/
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
└── requirements.txt
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Generate dataset
python generate_data.py

# 2. Clean + engineer features
python preprocess.py

# 3. Train model
python train.py

# 4. Run predictions
python predict.py
```

---

## Features Used

| Feature | Description |
|---|---|
| `demand_index` | Real-time ride request volume (0–100) |
| `driver_availability` | Supply ratio — lower = scarcer drivers |
| `weather` | 0=clear, 1=rain, 2=storm |
| `event_nearby` | Concert, game, or large event nearby |
| `competitor_price_index` | Normalized competitor pricing |
| `trip_distance_km` | Requested trip distance |
| `is_holiday` | Public holiday flag |
| `hour_sin / hour_cos` | Cyclical hour encoding |
| `dow_sin / dow_cos` | Cyclical day-of-week encoding |
| `demand_supply_ratio` | Core pricing signal: demand / supply |
| `is_rush_hour` | 7–9 AM or 5–7 PM |
| `is_night` | 10 PM – 5 AM |
| `is_weekend` | Saturday / Sunday |
| `bad_weather` | Rain or storm flag |

---

## Model

**Algorithm:** `GradientBoostingRegressor` (scikit-learn)

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `learning_rate` | 0.05 |
| `max_depth` | 5 |
| `subsample` | 0.8 |

### Results

| Metric | Score |
|---|---|
| R² (test) | **0.962** |
| 5-Fold CV R² | **0.961 ± 0.001** |
| MAE | 0.076 |
| RMSE | 0.097 |
| MAPE | **4.67%** |

---

## Sample Predictions

| Scenario | Predicted Surge |
|---|---|
| Friday rush hour + rain + event nearby | **3.22x** |
| Tuesday afternoon, calm, plenty of drivers | **1.44x** |
| Saturday night + storm + holiday | **4.01x** |

---

## Tech Stack

- Python 3.13
- scikit-learn
- pandas / numpy
- matplotlib / seaborn
- joblib
