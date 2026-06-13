"""
Clean and feature-engineer the raw rides dataset.
Outputs: data/processed_rides.csv
"""

import pandas as pd
import numpy as np


def load_and_clean(path="data/raw_rides.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Sanity checks ────────────────────────────────────────────────
    print(f"Raw shape: {df.shape}")
    print(f"Nulls:\n{df.isnull().sum()}")

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows")

    # Remove physically impossible values
    df = df[df["driver_availability"] > 0]
    df = df[df["surge_multiplier"].between(1.0, 4.0)]
    df = df[df["trip_distance_km"] > 0]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Cyclical encoding for hour and day_of_week (preserves circularity)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Demand-to-supply ratio — core dynamic pricing signal
    df["demand_supply_ratio"] = (df["demand_index"] / 100) / df["driver_availability"]

    # Rush hour flag
    df["is_rush_hour"] = df["hour"].isin(range(7, 10)) | df["hour"].isin(range(17, 20))
    df["is_rush_hour"] = df["is_rush_hour"].astype(int)

    # Night hours (higher risk premium)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)

    # Weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Bad weather flag
    df["bad_weather"] = (df["weather"] > 0).astype(int)

    # Log-transform distance (right-skewed)
    df["log_distance"] = np.log1p(df["trip_distance_km"])

    # Drop raw cyclical columns (encoded above) and keep originals for inspection
    return df


def main():
    df = load_and_clean()
    df = engineer_features(df)
    df.to_csv("data/processed_rides.csv", index=False)
    print(f"\nProcessed shape: {df.shape}")
    print(f"Saved → data/processed_rides.csv")
    print(df.head(3).T.to_string())


if __name__ == "__main__":
    main()
