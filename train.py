"""
Train a GradientBoostingRegressor to predict surge_multiplier.
Outputs:
  models/surge_model.pkl   — trained model
  models/scaler.pkl        — feature scaler
  outputs/feature_importance.png
  outputs/actual_vs_predicted.png
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

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

TARGET = "surge_multiplier"


def load_data(path="data/processed_rides.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def evaluate(model, X_test, y_test, scaler):
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print("\n── Test-set Metrics ─────────────────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")

    return y_pred


def plot_feature_importance(model, feature_names):
    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Feature Importance — Surge Multiplier Model", fontsize=13)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fig.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("Saved → outputs/feature_importance.png")


def plot_actual_vs_predicted(y_test, y_pred):
    sample = np.random.choice(len(y_test), size=min(3000, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test.iloc[sample], y_pred[sample], alpha=0.25, s=10, color="royalblue")
    lims = [1.0, 4.0]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Surge Multiplier")
    ax.set_ylabel("Predicted Surge Multiplier")
    ax.set_title("Actual vs Predicted — Surge Multiplier", fontsize=13)
    ax.legend()
    plt.tight_layout()
    fig.savefig("outputs/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Saved → outputs/actual_vs_predicted.png")


def main():
    print("Loading data …")
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
        verbose=1,
    )

    print("\nTraining …")
    model.fit(X_train_s, y_train)

    # 5-fold CV on training set
    cv_scores = cross_val_score(
        model, X_train_s, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    print(f"\n5-Fold CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred = evaluate(model, X_test, y_test, scaler)

    plot_feature_importance(model, FEATURES)
    plot_actual_vs_predicted(y_test, y_pred)

    joblib.dump(model,  "models/surge_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("\nSaved → models/surge_model.pkl")
    print("Saved → models/scaler.pkl")


if __name__ == "__main__":
    main()
