"""
Generate synthetic Uber-style ride pricing dataset.
Features mirror real-world dynamic pricing signals:
  demand, weather, local events, competition, time-of-day, etc.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 50_000


def generate():
    hour = np.random.randint(0, 24, N)
    day_of_week = np.random.randint(0, 7, N)   # 0=Mon, 6=Sun
    month = np.random.randint(1, 13, N)

    # Demand index 0-100
    base_demand = (
        20
        + 30 * np.sin(np.pi * hour / 12)                 # peak at noon/midnight
        + 15 * (day_of_week >= 5).astype(float)           # weekend bump
        + np.random.normal(0, 8, N)
    ).clip(0, 100)

    # Weather: 0=clear, 1=rain, 2=storm
    weather = np.random.choice([0, 1, 2], N, p=[0.65, 0.25, 0.10])

    # Local event nearby (concert, game, etc.)
    event_nearby = np.random.choice([0, 1], N, p=[0.80, 0.20])

    # Available drivers ratio (supply / expected_supply), lower = scarcer
    driver_availability = np.clip(
        np.random.normal(1.0, 0.25, N) - 0.15 * (weather > 0) - 0.2 * event_nearby,
        0.1, 2.0
    )

    # Competitor price index (normalized around 1.0)
    competitor_price_index = np.clip(np.random.normal(1.0, 0.15, N), 0.6, 1.5)

    # Trip distance (km)
    trip_distance_km = np.clip(np.random.exponential(8, N), 0.5, 60)

    # Is holiday
    is_holiday = np.random.choice([0, 1], N, p=[0.94, 0.06])

    # True surge multiplier (what we want to predict)
    surge = (
        1.0
        + 0.015 * base_demand
        + 0.3 * (weather == 1)
        + 0.7 * (weather == 2)
        + 0.4 * event_nearby
        + 0.5 * is_holiday
        - 0.6 * (driver_availability - 1.0)          # scarcity pushes price up
        + 0.2 * (competitor_price_index - 1.0)        # follow competition loosely
        + np.random.normal(0, 0.1, N)                 # noise
    ).clip(1.0, 4.0)

    df = pd.DataFrame({
        "hour":                  hour,
        "day_of_week":           day_of_week,
        "month":                 month,
        "demand_index":          base_demand.round(2),
        "weather":               weather,
        "event_nearby":          event_nearby,
        "driver_availability":   driver_availability.round(3),
        "competitor_price_index": competitor_price_index.round(3),
        "trip_distance_km":      trip_distance_km.round(2),
        "is_holiday":            is_holiday,
        "surge_multiplier":      surge.round(4),       # TARGET
    })

    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/raw_rides.csv", index=False)
    print(f"Saved {len(df):,} rows → data/raw_rides.csv")
    print(df.describe().T.to_string())
