"""
ML Model 2 — Gate Closure Duration Prediction

Predicts how long a gate stays closed when a train passes.
A passenger train closes a gate for ~2 minutes. A slow goods
train may close it for 8-10 minutes.

Input features:
  - train_type (express, passenger, goods, EMU local)
  - number_of_coaches (longer train = longer closure)
  - train_speed_at_gate (km/h, estimated from schedule)
  - time_of_day (hour)

Output: predicted_closure_minutes

Training data:
  Option A: Crowdsourced from users (see collect_closure_data.py)
  Option B: Calculated from train length + estimated speed
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'closure_model.pkl')

# Typical train specifications
TRAIN_SPECS = {
    'express':   {'coaches': 24, 'length_m': 460, 'speed_range': (60, 110)},
    'superfast': {'coaches': 24, 'length_m': 460, 'speed_range': (80, 130)},
    'passenger': {'coaches': 18, 'length_m': 360, 'speed_range': (30, 60)},
    'goods':     {'coaches': 40, 'length_m': 600, 'speed_range': (20, 50)},
    'emu':       {'coaches': 12, 'length_m': 240, 'speed_range': (40, 80)},
    'memu':      {'coaches': 12, 'length_m': 240, 'speed_range': (40, 70)},
}


def estimate_closure_duration(train_length_m, speed_kmh):
    """
    Estimate how long a gate stays closed.
    Gate opens when the last coach clears the crossing.
    """
    if speed_kmh <= 0:
        return 5.0  # Default estimate
    speed_ms = speed_kmh * (1000 / 3600)
    # Add approach + clearance buffer (gate closes before train arrives)
    approach_distance_m = 200
    total_distance = train_length_m + approach_distance_m
    duration_seconds = total_distance / speed_ms
    return duration_seconds / 60


def classify_train_type(train_number):
    """Classify train type from train number."""
    tn = str(train_number)
    if tn.startswith('0') or tn.startswith('6'):
        return 'goods'
    first_two = int(tn[:2]) if tn[:2].isdigit() else 0
    if first_two in range(10, 15):
        return 'superfast'
    if first_two in range(15, 20) or first_two in range(20, 25):
        return 'express'
    if first_two in range(55, 58):
        return 'passenger'
    if first_two in range(66, 68):
        return 'memu'
    return 'express'


def generate_training_data():
    """
    Generate training data using physics-based estimates.
    (Option B from the plan — used until real crowdsourced data is available)
    """
    np.random.seed(42)
    n_samples = 5000

    data = []
    for _ in range(n_samples):
        train_type = np.random.choice(list(TRAIN_SPECS.keys()),
                                       p=[0.3, 0.15, 0.25, 0.1, 0.1, 0.1])
        specs = TRAIN_SPECS[train_type]

        # Add variation to coaches and length
        coaches = specs['coaches'] + np.random.randint(-3, 4)
        coaches = max(4, coaches)
        length_m = coaches * (specs['length_m'] / specs['coaches'])

        # Speed varies based on location, time, conditions
        min_speed, max_speed = specs['speed_range']
        speed = np.random.uniform(min_speed * 0.7, max_speed * 1.1)
        speed = max(5, speed)

        hour = np.random.randint(0, 24)

        # Trains slow down at night and dawn
        if 0 <= hour <= 5:
            speed *= 0.8

        actual_duration = estimate_closure_duration(length_m, speed)
        # Add real-world noise (human delays in gate operation)
        actual_duration += np.random.normal(0.5, 0.3)
        actual_duration = max(0.5, actual_duration)

        # Encode train type
        type_map = {'express': 0, 'superfast': 1, 'passenger': 2,
                    'goods': 3, 'emu': 4, 'memu': 5}

        data.append({
            'train_type': type_map[train_type],
            'number_of_coaches': coaches,
            'train_speed_kmh': round(speed, 1),
            'time_of_day': hour,
            'closure_minutes': round(actual_duration, 2),
        })

    return pd.DataFrame(data)


def load_crowdsourced_data():
    """Load real crowdsourced closure data if available."""
    filepath = os.path.join(DATA_DIR, 'gate_closure_events.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        if len(df) >= 100:
            print(f"  Using {len(df)} crowdsourced closure records")
            return df
        print(f"  Only {len(df)} crowdsourced records — need 100+ for training")
    return None


def train_model():
    """Train the closure duration prediction model."""
    print("=== Gate Closure Duration Model ===\n")

    # Try crowdsourced data first, fall back to synthetic
    df = load_crowdsourced_data()
    if df is None:
        print("  No crowdsourced data — using physics-based estimates")
        df = generate_training_data()

    print(f"  Training on {len(df)} records...")

    feature_cols = ['train_type', 'number_of_coaches', 'train_speed_kmh', 'time_of_day']
    X = df[feature_cols]
    y = df['closure_minutes']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"  Test MAE:  {mae:.2f} minutes")
    print(f"  Test R2:   {r2:.3f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    # Quick prediction demo
    print("\nSample predictions:")
    samples = [
        {'train_type': 0, 'number_of_coaches': 24, 'train_speed_kmh': 80, 'time_of_day': 10},
        {'train_type': 2, 'number_of_coaches': 16, 'train_speed_kmh': 40, 'time_of_day': 8},
        {'train_type': 3, 'number_of_coaches': 40, 'train_speed_kmh': 30, 'time_of_day': 14},
    ]
    labels = ['Express (24 coaches, 80km/h)', 'Passenger (16 coaches, 40km/h)', 'Goods (40 wagons, 30km/h)']
    for s, label in zip(samples, labels):
        pred = model.predict(pd.DataFrame([s]))[0]
        print(f"  {label}: ~{pred:.1f} min")

    return model


if __name__ == '__main__':
    train_model()
    print("\nDone!")
