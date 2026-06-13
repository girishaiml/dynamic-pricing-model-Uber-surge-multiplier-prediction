"""
Inference script — given ride features, return the predicted surge multiplier.

Usage:
  python predict.py
"""

import joblib
import numpy as np
import pandas as pd

FEATURES = [
    "demand_index",
    "weather",
    "event_nearby",
    "driver_availability",
    "competitor_price_index",
    "log_distance",
    "is_holiday",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "demand_supply_ratio",
    "is_rush_hour",
    "is_night",
    "is_weekend",
    "bad_weather",
]


def build_features(
    hour: int,
    day_of_week: int,
    demand_index: float,
    weather: int,
    event_nearby: int,
    driver_availability: float,
    competitor_price_index: float,
    trip_distance_km: float,
    is_holiday: int,
) -> pd.DataFrame:
    row = {
        "demand_index":           demand_index,
        "weather":                weather,
        "event_nearby":           event_nearby,
        "driver_availability":    driver_availability,
        "competitor_price_index": competitor_price_index,
        "log_distance":           np.log1p(trip_distance_km),
        "is_holiday":             is_holiday,
        "hour_sin":               np.sin(2 * np.pi * hour / 24),
        "hour_cos":               np.cos(2 * np.pi * hour / 24),
        "dow_sin":                np.sin(2 * np.pi * day_of_week / 7),
        "dow_cos":                np.cos(2 * np.pi * day_of_week / 7),
        "demand_supply_ratio":    (demand_index / 100) / driver_availability,
        "is_rush_hour":           int(hour in range(7, 10) or hour in range(17, 20)),
        "is_night":               int(hour >= 22 or hour <= 5),
        "is_weekend":             int(day_of_week >= 5),
        "bad_weather":            int(weather > 0),
    }
    return pd.DataFrame([row])[FEATURES]


def predict(features: pd.DataFrame) -> float:
    model  = joblib.load("models/surge_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    X = scaler.transform(features)
    return float(model.predict(X)[0])


if __name__ == "__main__":
    # Example: Friday evening rush hour, rainy, concert nearby
    examples = [
        dict(hour=18, day_of_week=4, demand_index=85, weather=1,
             event_nearby=1, driver_availability=0.6,
             competitor_price_index=1.2, trip_distance_km=5, is_holiday=0),
        dict(hour=14, day_of_week=2, demand_index=40, weather=0,
             event_nearby=0, driver_availability=1.3,
             competitor_price_index=1.0, trip_distance_km=12, is_holiday=0),
        dict(hour=23, day_of_week=5, demand_index=90, weather=2,
             event_nearby=1, driver_availability=0.4,
             competitor_price_index=1.4, trip_distance_km=3, is_holiday=1),
    ]

    labels = [
        "Friday rush + rain + event",
        "Tuesday afternoon, calm",
        "Saturday night + storm + holiday",
    ]

    print("─" * 50)
    for label, ex in zip(labels, examples):
        feats = build_features(**ex)
        surge = predict(feats)
        print(f"{label:<35} → surge {surge:.2f}x")
    print("─" * 50)
