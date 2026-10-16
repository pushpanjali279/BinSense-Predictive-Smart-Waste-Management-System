"""
BinSense — Central Configuration
All tunable parameters live here. Nothing is hardcoded in the pipeline modules.
Adjust these for live "what-if" demos without touching code.
"""

# ─────────────────────────────────────────────
# 1. Demo Area / Bounding Box (OSM download)
# ─────────────────────────────────────────────
# Using a compact area in central Bangalore (Cubbon Park / MG Road area)
# Small enough for fast download, dense enough road network for realistic routing.
DEMO_BBOX = {
    "north": 12.9850,
    "south": 12.9650,
    "east":  77.6100,
    "west":  77.5850,
}
DEMO_PLACE_NAME = "Cubbon Park Area, Bengaluru"

# Road graph cache path (relative to project root)
ROAD_GRAPH_CACHE = "data/road_graph.graphml"

# OSMnx network type for road graph download
NETWORK_TYPE = "drive"  # "drive" = drivable roads only

# ─────────────────────────────────────────────
# 2. Data Simulation
# ─────────────────────────────────────────────
RANDOM_SEED = 42

NUM_BINS = 35  # Number of waste bins to simulate (20–50 range per PRD)

# Bin capacity range (liters)
BIN_CAPACITY_MIN = 120
BIN_CAPACITY_MAX = 360

# Accumulation rate profiles (%/hour) — bins are assigned to zones
ACCUMULATION_PROFILES = {
    "slow":   {"mean": 0.8,  "std": 0.2},   # residential, low-traffic
    "medium": {"mean": 2.0,  "std": 0.5},   # mixed-use areas
    "fast":   {"mean": 4.0,  "std": 1.0},   # commercial, markets, transit hubs
}

# Zone distribution (proportions summing to 1.0)
ZONE_DISTRIBUTION = {
    "slow":   0.30,
    "medium": 0.45,
    "fast":   0.25,
}

# Fill history simulation
HISTORY_HOURS = 72         # Hours of historical data to generate per bin
HISTORY_INTERVAL_HOURS = 1 # One reading per hour
COLLECTION_THRESHOLD = 90  # Bins are "collected" (reset) when they hit this %

# ─────────────────────────────────────────────
# 3. Fleet / Vehicles
# ─────────────────────────────────────────────
NUM_VEHICLES = 3
VEHICLE_CAPACITY_LITERS = 1200  # Per-vehicle max carrying capacity (liters)

# Depot location (lat, lon) — central point in the demo area
DEPOT_LOCATION = {
    "latitude":  12.9750,
    "longitude": 77.5950,
}

# ─────────────────────────────────────────────
# 4. Fill-Level Prediction
# ─────────────────────────────────────────────
PREDICTION_HORIZON_HOURS = 12   # How far ahead to forecast (configurable 6–24)
PREDICTION_LOOKBACK_HOURS = 24  # How many recent hours of history to use for regression

# ─────────────────────────────────────────────
# 5. Dynamic Priority Engine — Weights
# ─────────────────────────────────────────────
# Composite score = w1*current_fill + w2*predicted_fill + w3*accumulation_rate_norm
#                 + w4*overflow_risk_norm + w5*time_since_collection_norm
PRIORITY_WEIGHTS = {
    "w1_current_fill":             0.20,
    "w2_predicted_fill":           0.30,
    "w3_accumulation_rate":        0.15,
    "w4_overflow_risk":            0.25,
    "w5_time_since_collection":    0.10,
}

# Urgency tier thresholds (score out of 100)
TIER_THRESHOLDS = {
    "critical": 85,   # score >= 85
    "high":     65,   # 65 <= score < 85
    "medium":   40,   # 40 <= score < 65
    # Low: score < 40
}

# ─────────────────────────────────────────────
# 6. Bin Selection
# ─────────────────────────────────────────────
# Minimum tier to auto-include in a collection cycle
SELECTION_MIN_TIER = "medium"   # "critical", "high", "medium", or "low"
FLEET_CAPACITY_BUFFER = 0.85    # Budget 85% of theoretical capacity for realistic dispatching

# ─────────────────────────────────────────────
# 7. CVRP Solver (OR-Tools)
# ─────────────────────────────────────────────
CVRP_TIME_LIMIT_SECONDS = 10         # Max solver time (seconds)
CVRP_FIRST_SOLUTION_STRATEGY = "PARALLEL_CHEAPEST_INSERTION"
CVRP_LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"

# Distance scaling: OR-Tools uses integers, so we scale meters to avoid rounding
DISTANCE_SCALE_FACTOR = 1   # 1 = meters as-is (suitable for demo scale)

# ─────────────────────────────────────────────
# 8. Baseline Comparator
# ─────────────────────────────────────────────
BASELINE_FILL_THRESHOLD = 80  # Naive baseline collects bins above this % only
# Baseline uses straight-line (Euclidean) distance, not road network

# ─────────────────────────────────────────────
# 9. UI / Dashboard
# ─────────────────────────────────────────────
# Design system tokens (from UIUX-Design-Document.md)
COLORS = {
    "bg":         "#0E1912",
    "surface":    "#182620",
    "surface_2":  "#1E2F27",
    "line":       "rgba(241,239,230,0.10)",
    "text":       "#F1EFE6",
    "text_dim":   "#A9B3AC",
    "text_faint": "#6E7A72",
    "leaf":       "#6FAE7C",   # Low urgency / positive / efficiency
    "amber":      "#E8A13D",   # High urgency / warning
    "danger":     "#D9714E",   # Critical urgency / overflow
}

TIER_COLORS = {
    "critical": "#D9714E",
    "high":     "#E8A13D",
    "medium":   "#E8D44D",  # Yellow-ish for medium
    "low":      "#6FAE7C",
}

TIER_ORDER = ["critical", "high", "medium", "low"]

FONTS = {
    "display": "Space Grotesk",
    "body":    "Inter",
}

# Map tile settings
MAP_TILES = "OpenStreetMap"  # Switched from CartoDB to avoid API key warnings (see MEMORY.md)
MAP_DEFAULT_ZOOM = 15

# Vehicle route colors (distinct per vehicle)
VEHICLE_COLORS = [
    "#6FAE7C",   # leaf green
    "#5BA8D9",   # calm blue
    "#E8A13D",   # amber
    "#C07ADB",   # purple
    "#D9714E",   # coral
]
