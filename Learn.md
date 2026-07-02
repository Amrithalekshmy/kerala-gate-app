# Kerala Gate Watch — Complete Project Guide

This document explains every part of this project in detail — the architecture, the data pipeline, the algorithms, the machine learning models, and how everything connects. By the end, you'll understand not just *what* the code does, but *why* each decision was made.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [The Real-World Problem](#2-the-real-world-problem)
3. [Project Structure](#3-project-structure)
4. [Data Layer — Where Everything Starts](#4-data-layer--where-everything-starts)
5. [Backend — The Flask API Server](#5-backend--the-flask-api-server)
6. [Gate Status Algorithm — The Core Logic](#6-gate-status-algorithm--the-core-logic)
7. [The Data Pipeline — From Raw Data to Predictions](#7-the-data-pipeline--from-raw-data-to-predictions)
8. [Machine Learning — Model 1: Train Delay Prediction](#8-machine-learning--model-1-train-delay-prediction)
9. [Machine Learning — Model 2: Gate Closure Duration](#9-machine-learning--model-2-gate-closure-duration)
10. [Crowdsourced Data Collection](#10-crowdsourced-data-collection)
11. [Frontend — The User Interface](#11-frontend--the-user-interface)
12. [The Geographic Matching Algorithm](#12-the-geographic-matching-algorithm)
13. [Key Computer Science Concepts Used](#13-key-computer-science-concepts-used)
14. [How to Run Everything](#14-how-to-run-everything)
15. [Potential Improvements](#15-potential-improvements)

---

## 1. What This Project Does

Kerala Gate Watch is a web application that shows the real-time status of **1,222 railway level crossings (gates)** across Kerala. For each gate, it tells you:

- Is the gate **OPEN** (you can pass), **WARNING** (a train is approaching, gate will close soon), or **CLOSED** (a train is crossing right now)?
- Which train is approaching and when will it cross?
- How long will the gate stay closed?

It uses train schedules to calculate status, and machine learning to predict delays and closure durations.

---

## 2. The Real-World Problem

Kerala has one of the densest railway networks in India. Level crossings (where a road crosses a railway track) cause massive traffic delays. If you're driving and a gate closes, you might wait 2-15 minutes. There's no way to know in advance whether a gate on your route is closed or about to close.

This app solves that by combining:
- **Geographic data** (where each gate is located)
- **Train schedules** (when each train departs each station)
- **Spatial algorithms** (which trains pass through which gates)
- **ML predictions** (how late is a train likely to be, and how long will the gate stay closed)

---

## 3. Project Structure

```
kerala-gate-app/
├── backend/                  # Python Flask server
│   ├── app.py                # Main Flask application, API routes
│   ├── database.py           # SQLite setup and CSV data loading
│   ├── gate_status.py        # Core algorithm: gate OPEN/WARNING/CLOSED
│   └── ml_predict.py         # ML model loading and prediction functions
│
├── frontend/                 # Static HTML/CSS/JS (no framework)
│   ├── index.html            # Map page (Leaflet.js map with gate markers)
│   ├── gate.html             # Individual gate detail page
│   ├── search.html           # Search/filter page
│   ├── app.js                # All frontend JavaScript
│   └── style.css             # All styling
│
├── ml/                       # Machine learning models
│   ├── train_delay_model.py  # Model 1: predict train delays
│   ├── train_closure_model.py# Model 2: predict gate closure duration
│   ├── collect_closure_data.py # Crowdsource real closure data from users
│   ├── delay_model.pkl       # Trained delay model (serialized)
│   ├── closure_model.pkl     # Trained closure model (serialized)
│   └── delay_train_encoder.pkl # LabelEncoder for train numbers
│
├── scripts/                  # Data generation and scraping
│   ├── generate_schedules.py # Generate realistic train schedule data
│   ├── scrape_schedules.py   # (Original) Scrape from erail.in
│   └── match_gates_to_trains.py # Geographic matching: gates <-> trains
│
├── data/                     # All data files
│   ├── kerala_gates.csv      # 1,222 gate locations from OpenStreetMap
│   ├── kerala_stations.csv   # 172 railway stations with coordinates
│   ├── train_schedules.csv   # Generated train timetable data
│   ├── gate_train_mapping.csv# Which trains cross which gates
│   ├── historical_delays.csv # Training data for delay model
│   └── railway.db            # SQLite database (created at runtime)
│
└── venv/                     # Python virtual environment
```

---

## 4. Data Layer — Where Everything Starts

### 4.1 The Database (`backend/database.py`)

The app uses **SQLite** — a file-based database that requires zero setup. No server, no passwords, just a single file (`data/railway.db`).

**Why SQLite?** For a project of this scale (thousands of rows, not millions), SQLite is the perfect choice. It's embedded directly in Python's standard library (`sqlite3`), so there's nothing to install.

#### Tables

```sql
-- Where the gates are
gates (gate_id, display_name, lat, lon, road_name, nearest_place, district)

-- Where the stations are
stations (station_code, station_name, station_name_ml, lat, lon)

-- When trains arrive at each station
train_schedules (train_number, train_name, station_code, station_name,
                 arrival_time, departure_time, distance_km,
                 day_of_journey, runs_on_days)

-- THE KEY TABLE: which trains cross which gates, and when
gate_train_mapping (gate_id, train_number, train_name,
                    estimated_crossing_time, prev_station_code,
                    next_station_code, runs_on_days)

-- For ML Model 1 training
historical_delays (train_number, station_code, scheduled_arrival,
                   actual_arrival, delay_minutes, ...)

-- For ML Model 2 training (crowdsourced)
gate_closure_events (gate_id, train_number, closed_at, opened_at,
                     duration_minutes, source)
```

#### Data Loading Flow

When the app starts, `load_all_data()` reads four CSV files and inserts them into SQLite:

```
kerala_gates.csv      → gates table        (1,222 rows)
kerala_stations.csv   → stations table     (172 rows)
train_schedules.csv   → train_schedules    (5,360 rows)
gate_train_mapping.csv→ gate_train_mapping (32,652 rows)
```

The CSV loading uses `csv.DictReader` to read each row as a dictionary, then inserts with parameterized queries (`?` placeholders) to prevent SQL injection.

### 4.2 The Gate Data (`data/kerala_gates.csv`)

Contains 1,222 level crossings extracted from OpenStreetMap. Each gate has:
- A unique ID (OSM node ID like `node/269201602`)
- A human-readable name like `LC near Vellayil (P. T. Usha Road)`
- GPS coordinates (latitude, longitude)
- The road it's on, nearest place, and district

### 4.3 The Station Data (`data/kerala_stations.csv`)

Contains 172 railway stations in Kerala with:
- Station code (e.g., `TVC` for Thiruvananthapuram Central, `ERS` for Ernakulam Junction)
- Station name in English and Malayalam
- GPS coordinates

Kerala's main railway lines:
- **Coastal Main Line**: Mangalore → Kasargod → Kannur → Kozhikode → Shoranur → Thrissur → Ernakulam → Alappuzha → Kollam → Trivandrum (runs along the coast, north to south)
- **Inland Route via Kottayam**: Ernakulam → Thrippunithura → Kottayam → Changanasseri → Kayamkulam (alternative to the coastal route)
- **Shoranur–Palakkad**: Branch line heading east toward Tamil Nadu
- **Shoranur–Nilambur**: Short branch line
- **Thrissur–Guruvayur**: Short branch to the temple town
- **Kayamkulam–Punalur–Shencottah**: Line heading east through the hills

---

## 5. Backend — The Flask API Server

### 5.1 What is Flask?

Flask is a Python web framework — a library that lets you define URL patterns (routes) and the Python functions that respond to them. It's called a "micro-framework" because it gives you the bare minimum and lets you add what you need.

### 5.2 The App (`backend/app.py`)

The Flask app serves two purposes:
1. **API server** — Returns JSON data at `/api/*` endpoints
2. **Static file server** — Serves the HTML/CSS/JS frontend files

```python
app = Flask(__name__,
    static_folder='../frontend',  # Serve files from the frontend directory
    static_url_path=''            # Serve them at the root URL (not /static/)
)
```

### 5.3 API Endpoints

| Endpoint | What it returns |
|---|---|
| `GET /api/gates` | All 1,222 gates with current status |
| `GET /api/gates/<gate_id>` | Detailed info for one gate, including upcoming trains |
| `GET /api/gates/nearby?lat=X&lon=Y&radius=5` | Gates within 5 km of a location |
| `GET /api/trains/<number>/position` | Current position and upcoming gates for a train |
| `GET /api/stats` | Dashboard stats (counts by status) |
| `GET /api/predict/delay?train=16301` | ML-predicted delay for a train |
| `POST /api/report/closed` | User reports a gate just closed |
| `POST /api/report/opened` | User reports a gate just opened |

### 5.4 The Startup Sequence

When `create_app()` runs:

```python
def create_app():
    init_db()              # 1. Create SQLite tables
    load_all_data()        # 2. Load CSVs into the database
    load_models()          # 3. Load ML models from .pkl files
    refresh_status_cache() # 4. Calculate status for all 1,222 gates
    # 5. Start a background scheduler to recalculate every 60 seconds
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_status_cache, 'interval', seconds=60)
    scheduler.start()
```

### 5.5 The Background Scheduler

**APScheduler** runs `refresh_status_cache()` every 60 seconds. This recalculates the OPEN/WARNING/CLOSED status for all gates. Without this, the statuses would be frozen at whatever they were when the app started.

The results are stored in an **in-memory cache** (a Python dictionary), so API requests don't need to recalculate — they just read from the cache. This is a common pattern: do the expensive work in the background, serve cheap results to users.

---

## 6. Gate Status Algorithm — The Core Logic

This is the heart of the application. The file `backend/gate_status.py` determines whether each gate is OPEN, WARNING, or CLOSED.

### 6.1 The Rules

```
Train 0-5 minutes away  → CLOSED  (gate barriers are down)
Train 5-15 minutes away → WARNING (gate will close soon)
Train >15 minutes away  → OPEN    (safe to pass)
```

These thresholds are defined as constants:
```python
CLOSED_THRESHOLD = 5    # minutes
WARNING_THRESHOLD = 15  # minutes
```

### 6.2 How Status is Calculated

For each gate:

1. **Query the database** for all trains mapped to this gate (from `gate_train_mapping` table)
2. **Filter by today's day** — some trains only run Mon/Wed/Fri, etc.
3. **Calculate time difference** — for each train, how many minutes until it crosses this gate?
4. **Apply the rules** — pick the closest approaching train and set the status

```python
def calculate_gate_status(gate_id, now=None):
    # Get current time in minutes since midnight
    current_minutes = now.hour * 60 + now.minute  # e.g., 14:30 → 870

    for row in rows:  # Each train mapped to this gate
        # Parse crossing time: "14:35" → 875 minutes
        crossing_minutes = parse_time_to_minutes(row['estimated_crossing_time'])

        # How far away is this train?
        time_diff = crossing_minutes - current_minutes  # e.g., 875 - 870 = 5

        # Handle midnight wrap (e.g., current=23:50, train=00:10)
        if time_diff < -720:
            time_diff += 1440   # Add 24 hours
        elif time_diff > 720:
            time_diff -= 1440   # Subtract 24 hours

        # Apply thresholds
        if 0 <= time_diff <= 5:
            status = 'CLOSED'
        elif 5 < time_diff <= 15:
            if status != 'CLOSED':  # Don't downgrade from CLOSED
                status = 'WARNING'
```

### 6.3 Midnight Wrap Logic

Time wraps around at midnight. If it's 23:50 and a train crosses at 00:05, the naive calculation gives `5 - 1430 = -1425`. The fix:

```python
if time_diff < -720:    # More than 12 hours in the past?
    time_diff += 1440   # It's actually in the future (next day)
```

The 720-minute (12-hour) threshold is the halfway point of a day. If the difference is more than 12 hours negative, we assume the train is actually in the future, not the distant past.

### 6.4 The `runs_on_days` Format

Trains don't always run daily. The `runs_on_days` string encodes which days a train operates:

```
"MTWTFSS" → Runs every day
"MTWTF--" → Runs Monday through Friday only
"M-W-F--" → Runs Monday, Wednesday, Friday
"--W--S-" → Runs Wednesday and Saturday
```

Each character position maps to a day (0=Monday through 6=Sunday). A dash `-` means the train doesn't run that day.

```python
def train_runs_today(runs_on_days, now):
    day_idx = now.weekday()  # 0=Monday
    return runs_on_days[day_idx] != '-'
```

### 6.5 The Caching Strategy

Calculating status for 1,222 gates means querying the database 1,222 times. This is too slow to do on every API request. The solution:

1. A **global dictionary** `_status_cache` holds the latest status for every gate
2. A **background scheduler** updates this cache every 60 seconds
3. API requests just read from the cache — O(1) lookup, no database hit

```python
_status_cache = {}  # gate_id → {status, next_train, minutes_until_closing, ...}

def refresh_status_cache():
    global _status_cache
    _status_cache = calculate_all_gate_statuses()

def get_cached_status(gate_id):
    return _status_cache.get(gate_id, {'status': 'UNKNOWN', ...})
```

---

## 7. The Data Pipeline — From Raw Data to Predictions

The data flows through three stages:

```
Stage 1: Generate Train Schedules
  generate_schedules.py → train_schedules.csv
  (defines when each train departs each station)

Stage 2: Match Gates to Trains
  match_gates_to_trains.py → gate_train_mapping.csv
  (determines which trains cross which gates, and at what time)

Stage 3: Runtime Status Calculation
  gate_status.py reads gate_train_mapping + current time
  → OPEN / WARNING / CLOSED for each gate
```

### 7.1 Stage 1: Schedule Generation (`scripts/generate_schedules.py`)

This script generates realistic train schedules for 158 trains across Kerala's railway network.

**How it works:**

1. **Define routes** — Each railway line is defined as an ordered list of station codes:
   ```python
   MAIN_LINE_NORTH = ['MJS', 'UAA', 'KMQ', 'KGQ', ..., 'CLT']
   ```

2. **Define trains** — Each train has a route, departure time, average speed, and running days:
   ```python
   ('16605', 'Mangala Lakshadweep Exp', FULL_COASTAL_SOUTH, 6, 0, 55, 2, 'MTWTFSS')
   # Train 16605 departs at 06:00, averages 55 km/h, 2-min halt per station, runs daily
   ```

3. **Calculate timings** — For each station on the route:
   - Compute the distance from the starting station (using the Haversine formula)
   - Divide distance by speed to get travel time
   - Add cumulative halt time
   - Convert to HH:MM format

```python
arrival_total = start_time + (distance / speed * 60) + (station_index * halt_minutes)
```

### 7.2 Stage 2: Geographic Matching (`scripts/match_gates_to_trains.py`)

This is the most algorithmically interesting part. It answers: **which trains pass through which gates?**

A gate sits on a road that crosses a railway track. That track connects two stations. If we know which trains run between those two stations, and where exactly the gate is between them, we can estimate when each train will pass through.

The algorithm is explained in detail in [Section 12](#12-the-geographic-matching-algorithm).

---

## 8. Machine Learning — Model 1: Train Delay Prediction

**File:** `ml/train_delay_model.py`

### 8.1 The Problem

Indian trains are frequently late. If a train is scheduled to cross a gate at 14:30 but it's actually running 20 minutes late, the gate won't close at 14:25 — it'll close at 14:45. Without delay prediction, the app would give wrong CLOSED/WARNING timings.

### 8.2 The Algorithm: Random Forest Regressor

A **Random Forest** is an ensemble of decision trees. Here's how it works conceptually:

1. **Decision Tree**: A tree that splits data based on features. At each node, it asks a question like "Is the scheduled hour > 17?" and branches left or right. The leaf nodes contain predictions.

2. **Random Forest**: Train 100 decision trees, each on a slightly different random subset of the data. To make a prediction, ask all 100 trees and average their answers. This reduces overfitting dramatically.

```python
model = RandomForestRegressor(
    n_estimators=100,    # 100 trees in the forest
    max_depth=15,        # Each tree can be at most 15 levels deep
    min_samples_leaf=5,  # Leaf nodes must have at least 5 samples
    n_jobs=-1,           # Use all CPU cores for training
)
```

### 8.3 Input Features (What the Model Sees)

| Feature | Type | Why it matters |
|---|---|---|
| `train_number_encoded` | Integer | Some trains are chronically late, others are punctual. This captures per-train behavior. |
| `day_of_week` | 0-6 | Weekend trains may have different delay patterns (less commuter traffic). |
| `month` | 1-12 | **Monsoon months (June-September)** cause significantly more delays in Kerala — flooding, track submersion, landslides. |
| `scheduled_hour` | 0-23 | Rush hour trains (7-10am, 5-8pm) face more congestion. |
| `prev_station_delay` | Float | **This is the most important feature** (91.68% importance). If a train was 20 min late at the previous station, it's probably still late. Delays compound. |
| `distance_from_origin` | Float | Delays accumulate over longer distances. A train starting 5 minutes late might be 30 minutes late by the time it reaches a station 800 km away. |
| `is_express` | 0/1 | Express/superfast trains have fewer stops and dedicated scheduling, so they're generally more punctual than passenger trains. |

### 8.4 Feature Importance

The model report reveals:

```
prev_station_delay        0.9168   ← Dominates! 91.7% of prediction power
distance_from_origin      0.0273
scheduled_hour            0.0162
train_number_encoded      0.0161
month                     0.0133
day_of_week               0.0094
is_express                0.0009   ← Almost irrelevant
```

**Key insight**: The previous station's delay is overwhelmingly the best predictor. This makes intuitive sense — if a train was 30 minutes late at Ernakulam, it's going to be roughly 30 minutes late at Aluva (the next station). The other features add small corrections.

### 8.5 Training Data

The training data (`data/historical_delays.csv`) is synthetic — generated with realistic delay patterns:

```python
# Base delay follows an exponential distribution
base_delay = np.random.exponential(8 if is_express else 15)

# Monsoon effect: 50% more delays
if 6 <= month <= 9:
    base_delay *= 1.5

# Rush hour effect: 30% more delays
if 7 <= scheduled_hour <= 10 or 17 <= scheduled_hour <= 20:
    base_delay *= 1.3
```

**Why exponential distribution?** Most trains are close to on time (small delays), but a few are extremely late (long tail). This matches real-world delay data. The exponential distribution captures this "most are small, a few are huge" pattern.

### 8.6 Model Evaluation

```
Mean Absolute Error (test): 5.0 minutes
RMSE (test):                6.4 minutes
R-squared (test):           0.799
```

- **MAE of 5 minutes**: On average, predictions are off by 5 minutes. For gate status (which uses 5-minute and 15-minute thresholds), this is acceptable.
- **R-squared of 0.799**: The model explains ~80% of the variance in delays. Not perfect, but useful.
- **Target was MAE < 10 minutes**: PASS.

### 8.7 How the Model is Used at Runtime

```python
# In ml_predict.py
def predict_delay(train_number, scheduled_hour, prev_delay, distance_km):
    features = pd.DataFrame([{
        'train_number_encoded': encoder.transform([train_number])[0],
        'day_of_week': now.weekday(),
        'month': now.month,
        'scheduled_hour': scheduled_hour,
        'prev_station_delay': prev_delay,
        'distance_from_origin': distance_km,
        'is_express': 1 if train_number.startswith('1') else 0,
    }])
    return model.predict(features)[0]
```

### 8.8 Serialization with joblib

Trained models are saved to disk with `joblib`:

```python
joblib.dump(model, 'delay_model.pkl')        # Save
_delay_model = joblib.load('delay_model.pkl') # Load
```

**What's a .pkl file?** It's a serialized Python object — the entire trained model (all 100 trees with their split points and leaf values) is converted to bytes and saved. Loading it gives you back the exact same model without retraining.

### 8.9 LabelEncoder — Handling Categorical Data

Train numbers are strings like "16301", "12625". Random Forest needs numbers. `LabelEncoder` maps each unique train number to an integer:

```python
encoder = LabelEncoder()
encoder.fit_transform(["16301", "12625", "56361"])
# → [1, 0, 2]  (alphabetical order)
```

The encoder is saved separately (`delay_train_encoder.pkl`) so the same mapping is used at prediction time.

---

## 9. Machine Learning — Model 2: Gate Closure Duration

**File:** `ml/train_closure_model.py`

### 9.1 The Problem

When a gate closes, how long does it stay closed? This varies enormously:
- A **superfast express** at 100+ km/h: gate closed for ~1 minute
- A **passenger train** at 40 km/h: gate closed for ~2-3 minutes
- A **goods train** at 25 km/h with 40 wagons: gate closed for 8-10 minutes

### 9.2 Physics-Based Estimation

Before any ML, the code uses basic physics to estimate closure duration:

```python
def estimate_closure_duration(train_length_m, speed_kmh):
    speed_ms = speed_kmh * (1000 / 3600)     # Convert km/h to m/s
    approach_distance_m = 200                  # Gate closes before train arrives
    total_distance = train_length_m + approach_distance_m
    duration_seconds = total_distance / speed_ms
    return duration_seconds / 60               # Convert to minutes
```

**Example**: An express train (460m long) at 80 km/h:
```
speed = 80 * (1000/3600) = 22.2 m/s
total_distance = 460 + 200 = 660 m
duration = 660 / 22.2 = 29.7 seconds = 0.5 minutes
```

But real-world gate closures are longer — gatekeepers close gates before the train is visible (safety margin), and open them only after confirming the track is clear. So the ML model adds a learned offset.

### 9.3 Train Type Classification

Train numbers encode the train type. Indian Railways follows a numbering system:

```python
TRAIN_SPECS = {
    'express':   {'coaches': 24, 'length_m': 460, 'speed_range': (60, 110)},
    'superfast': {'coaches': 24, 'length_m': 460, 'speed_range': (80, 130)},
    'passenger': {'coaches': 18, 'length_m': 360, 'speed_range': (30, 60)},
    'goods':     {'coaches': 40, 'length_m': 600, 'speed_range': (20, 50)},
    'emu':       {'coaches': 12, 'length_m': 240, 'speed_range': (40, 80)},
    'memu':      {'coaches': 12, 'length_m': 240, 'speed_range': (40, 70)},
}
```

```
Train numbers 10000-14999 → Superfast
Train numbers 15000-24999 → Express
Train numbers 55000-57999 → Passenger
Train numbers 66000-67999 → MEMU (Mainline EMU)
```

### 9.4 The Model

Same algorithm (Random Forest Regressor), but with fewer, simpler features:

| Feature | What it is |
|---|---|
| `train_type` | Encoded integer: 0=express, 1=superfast, 2=passenger, 3=goods, 4=emu, 5=memu |
| `number_of_coaches` | How many coaches/wagons (longer train = longer closure) |
| `train_speed_kmh` | Estimated speed at the gate |
| `time_of_day` | Hour (0-23) — trains slow down at night |

### 9.5 Fallback: Physics When ML Isn't Available

If the model file doesn't exist, the code falls back to pure physics:

```python
if _closure_model is not None:
    return _closure_model.predict(features)[0]

# Fallback: physics
length_m = default_coaches * 19  # ~19m per coach
speed_ms = speed_kmh * (1000 / 3600)
duration_min = (length_m + 200) / speed_ms / 60
```

This is a good software pattern: **graceful degradation**. The app still works without ML models, just with less accurate estimates.

---

## 10. Crowdsourced Data Collection

**File:** `ml/collect_closure_data.py`

### 10.1 The Feedback Loop

The synthetic training data is good for a v1, but real-world data is better. This module lets users report gate closures:

1. User sees a gate close → taps "Gate Closed" → `POST /api/report/closed`
2. User sees the gate open → taps "Gate Opened" → `POST /api/report/opened`
3. The system calculates the duration and stores it in `gate_closure_events`

```python
def record_gate_closed(gate_id, train_number):
    # Insert a row with closed_at = now, opened_at = NULL
    cursor.execute('''
        INSERT INTO gate_closure_events (gate_id, train_number, closed_at, ...)
        VALUES (?, ?, ?, 'user_report')
    ''', (gate_id, train_number, now))

def record_gate_opened(gate_id):
    # Find the most recent unfinished closure, calculate duration
    # Sanity check: reject if duration > 30 minutes (data quality)
    duration = (opened_at - closed_at).total_seconds() / 60
    if duration > 30:
        return None  # Probably user forgot to tap "opened"
```

### 10.2 The Data Quality Problem

User-reported data has issues:
- Users might forget to tap "opened"
- They might tap "closed" late or early
- Bad network might delay the request

The 30-minute sanity check catches the most obvious bad data. The target is 500+ clean events before retraining Model 2 with real data.

### 10.3 Export for Retraining

Once enough data is collected:
```python
def export_training_data():
    # Write completed events to gate_closure_events.csv
    # Then retrain: python ml/train_closure_model.py
```

The closure model checks for crowdsourced data first and falls back to synthetic:
```python
def train_model():
    df = load_crowdsourced_data()       # Try real data first
    if df is None:
        df = generate_training_data()   # Fall back to synthetic
```

---

## 11. Frontend — The User Interface

The frontend is vanilla HTML/CSS/JavaScript — no React, no Vue, no build step. This keeps things simple and fast.

### 11.1 The Map Page (`index.html` + `app.js`)

Uses **Leaflet.js**, an open-source JavaScript library for interactive maps.

```javascript
// Initialize map centered on Kerala
map = L.map('map').setView([10.5, 76.3], 8);

// Add OpenStreetMap tiles
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
```

Each gate is a **circle marker** colored by status:
```javascript
const STATUS_COLORS = {
    OPEN: '#4caf50',     // Green
    WARNING: '#ff9800',  // Orange
    CLOSED: '#f44336',   // Red
    UNKNOWN: '#9e9e9e',  // Gray
};
```

Clicking a marker shows a popup with gate name, status, and the next train.

### 11.2 Auto-Refresh

The map refreshes every 60 seconds to stay current:
```javascript
refreshTimer = setInterval(loadGates, 60000);
```

This matches the backend's 60-second cache refresh cycle.

### 11.3 The "Near Me" Button

Uses the browser's **Geolocation API** to find the user's position:
```javascript
navigator.geolocation.getCurrentPosition(pos => {
    map.setView([pos.coords.latitude, pos.coords.longitude], 14);
    // Also calls /api/gates/nearby to filter
});
```

### 11.4 The Search Page (`search.html`)

- Text search across gate name, district, and gate ID
- Filter buttons to show only CLOSED/WARNING/OPEN gates
- Each result is a clickable card that links to the gate detail page

The filtering is **client-side** — all 1,222 gates are loaded once, then filtered in JavaScript. This is fast because the dataset is small enough to hold in memory.

### 11.5 The Gate Detail Page (`gate.html`)

Shows:
- A mini map zoomed into the gate location
- The gate's current status with a countdown
- A list of upcoming trains with estimated crossing times

### 11.6 XSS Prevention

All user-visible text is escaped before rendering:
```javascript
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;    // textContent escapes HTML entities
    return div.innerHTML;
}
```

This prevents **Cross-Site Scripting (XSS)** — if a gate name somehow contained `<script>alert('hack')</script>`, it would be displayed as text, not executed.

---

## 12. The Geographic Matching Algorithm

**File:** `scripts/match_gates_to_trains.py`

This is the most complex algorithm in the project. It solves: **"Given a gate's GPS coordinates, which trains pass through it?"**

### 12.1 The Approach

A railway gate sits on a road that crosses a railway track. That track runs between two stations. If we know which two stations a gate is between, we know which trains pass through it (any train that stops at both stations consecutively).

```
Station A ───────── Gate ────────── Station B
  (TVC)        (LC on NH 66)         (NEM)

All trains that go TVC → NEM (or NEM → TVC) pass through this gate.
```

### 12.2 Step 1: Build Station-Pair Segments

From the train schedule, extract every pair of consecutive stations:

```
Train 16301: TVC → NEM → BRAM → NYY → ...
  Pairs: (TVC, NEM), (NEM, BRAM), (BRAM, NYY), ...
```

Each pair defines a line segment on the map.

### 12.3 Step 2: Point-to-Segment Distance

For each gate, calculate its distance to every station-pair segment. This uses the **point-to-line-segment distance** formula.

#### The Math

Given a point P (the gate) and a line segment from A to B (two stations), find the closest point on the segment to P.

```python
def point_to_segment_distance(px, py, ax, ay, bx, by):
    # Convert lat/lon to flat km coordinates (flat-earth approximation)
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * cos(mid_latitude)

    # Vector math: project P onto the line AB
    # t = dot(AP, AB) / dot(AB, AB)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx*dx + dy*dy)
    t = clamp(t, 0, 1)  # Keep within segment bounds

    # Closest point on segment
    cx = ax + t * dx
    cy = ay + t * dy

    # Distance from P to closest point
    return sqrt((px - cx)^2 + (py - cy)^2), t
```

The return value `t` is the **projection fraction**:
- `t = 0.0` → the gate is at station A
- `t = 0.5` → the gate is exactly halfway between A and B
- `t = 1.0` → the gate is at station B

#### Flat-Earth Approximation

GPS coordinates are in degrees, but distance needs to be in km. At Kerala's latitude (~10 degrees N):
- 1 degree of latitude = 111.32 km (always)
- 1 degree of longitude = 111.32 x cos(10 degrees) = 109.6 km

This approximation is accurate within Kerala (distances < 50 km between stations).

### 12.4 Step 3: Filter and Estimate Crossing Time

A gate matches a segment if:
1. Its distance to the segment is <= 0.5 km (500 meters)
2. Its projection fraction `t` is between 0.01 and 0.99 (not at the very ends)

Then, for each train on that segment, estimate when it crosses the gate:

```python
# Train departs Station A at dep, arrives Station B at arr
travel_time = arr - dep                        # e.g., 30 minutes
estimated_crossing = dep + t * travel_time     # e.g., dep + 0.4 * 30 = dep + 12 min
```

This assumes the train moves at constant speed between stations — not perfectly accurate, but good enough.

### 12.5 Why 500 Meters?

The `MAX_GATE_DISTANCE_KM = 0.5` threshold is a tradeoff:
- **Too small** (100m): would miss gates that are slightly off the geometric line between stations (railways curve)
- **Too large** (2km): would match gates on nearby parallel roads or other railway lines
- **500m**: catches gates on curves while avoiding false matches

### 12.6 The Edge Exclusion

```python
if t <= 0.01 or t >= 0.99:
    continue  # Skip gates at the very edge
```

If a gate appears to be right at a station (t = 0 or t = 1), it's probably not actually between those stations — it might be on a different line that happens to pass near the station. This avoids false matches at junction stations.

---

## 13. Key Computer Science Concepts Used

### 13.1 The Haversine Formula

Calculates the great-circle distance between two points on a sphere (Earth).

```python
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c
```

**Why not just Euclidean distance?** Because the Earth is round. At the equator, 1 degree longitude = 111 km. At 60 degrees N latitude, 1 degree longitude = 55 km. The Haversine formula accounts for this curvature.

### 13.2 In-Memory Caching

Instead of recalculating gate statuses on every request:

```
Request → Check cache → Return cached result
                           ↑
                           |
Background timer ──────── Update cache every 60s
```

This is a **space-time tradeoff**: use memory (store all 1,222 gate statuses) to save time (avoid 1,222 database queries per request).

### 13.3 REST API Design

The API follows REST conventions:
- **Resources** are nouns: `/api/gates`, `/api/trains`
- **GET** for reading, **POST** for creating
- **Nested resources**: `/api/gates/<gate_id>` for a specific gate
- **Query parameters** for filtering: `/api/gates/nearby?lat=X&lon=Y`

### 13.4 Feature Engineering for ML

Raw data isn't useful for ML. Feature engineering transforms it:

| Raw data | Engineered feature |
|---|---|
| Train number "16301" | `is_express = 1` (starts with 1) |
| Train number "16301" | `train_number_encoded = 7` (LabelEncoder) |
| Timestamp "2024-07-15 14:30" | `month = 7`, `scheduled_hour = 14`, `day_of_week = 0` |

### 13.5 Ensemble Learning (Random Forest)

A single decision tree overfits — it memorizes the training data. A Random Forest fixes this:

1. **Bagging**: Each tree trains on a random subset of the data (with replacement)
2. **Feature randomness**: At each split, each tree only considers a random subset of features
3. **Averaging**: Final prediction = average of all trees

This is the **bias-variance tradeoff**: individual trees have high variance (different subsets lead to different trees), but averaging reduces variance without increasing bias.

### 13.6 Row Factory Pattern (SQLite)

```python
conn.row_factory = sqlite3.Row
```

This makes query results behave like dictionaries instead of tuples:
```python
# Without row factory:
row[0], row[1], row[2]    # Fragile, position-dependent

# With row factory:
row['gate_id'], row['lat'], row['lon']  # Self-documenting
```

### 13.7 XSS Prevention

User-controlled data that gets rendered in HTML must be escaped:
```javascript
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;    // The browser escapes it for you
    return div.innerHTML;     // Get the escaped version
}
```

`textContent` is safe because it treats the string as text, not HTML. `<script>` becomes `&lt;script&gt;`.

### 13.8 Graceful Degradation

The ML layer is completely optional. If model files don't exist:
- Delay prediction returns 0 (assume on time)
- Closure duration uses physics-based estimation

The app works at every level of data availability:
- No train data → all gates show OPEN (degraded but not broken)
- Train data but no ML → accurate status, no delay/duration predictions
- Full data + ML → accurate status with delay-adjusted timings

---

## 14. How to Run Everything

### 14.1 First-Time Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 14.2 Generate Data (if missing)

```bash
# Step 1: Generate train schedules
python scripts/generate_schedules.py

# Step 2: Match gates to trains (takes ~10 seconds)
python scripts/match_gates_to_trains.py

# Optional: Train/retrain ML models
python ml/train_delay_model.py
python ml/train_closure_model.py
```

### 14.3 Run the App

```bash
python -m backend.app
# Server starts at http://127.0.0.1:5000
```

### 14.4 Dependencies

| Package | What it does |
|---|---|
| `flask` | Web framework for API and serving files |
| `apscheduler` | Background job scheduling (refresh every 60s) |
| `requests` | HTTP client (used by the scraper) |
| `beautifulsoup4` | HTML parsing (used by the scraper) |
| `scikit-learn` | ML framework: Random Forest, train/test split, metrics |
| `pandas` | Data manipulation, feature DataFrames for ML |
| `joblib` | Serialize/deserialize ML models to .pkl files |

---

## 15. Potential Improvements

### Data Quality
- **Real train schedules**: Connect to Indian Railways' API or IRCTC for live schedule data instead of generated schedules
- **Live train tracking**: Use real-time train position data to calculate exact gate crossing times
- **More gate data**: Add gate type (manned/unmanned), number of tracks, road traffic volume

### ML Improvements
- **Real delay data**: Replace synthetic data with actual Indian Railways delay records from data.gov.in
- **More features**: Add weather data (API), festival/holiday flags, special train flags
- **Model upgrade**: Try gradient boosting (XGBoost/LightGBM) which often outperforms Random Forest on tabular data
- **Online learning**: Update the model with each new delay observation instead of batch retraining

### Architecture
- **WebSocket for live updates**: Instead of polling every 60 seconds, push status changes to connected clients in real-time
- **PostgreSQL**: If the app scales beyond a single server, move from SQLite to PostgreSQL
- **Redis cache**: Replace the in-memory dictionary cache with Redis for multi-process setups
- **Progressive Web App (PWA)**: Add a service worker for offline access and push notifications ("Gate X on your route is closing!")

### Frontend
- **Route planning**: "Show me all gates between Kochi and Trivandrum" with a route overlay
- **Notifications**: Alert when a gate on a saved route is about to close
- **Historical view**: "How often is this gate closed at 8 AM on weekdays?" — useful for commuters planning their route
