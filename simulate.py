"""
BinSense — Data Simulation Module

Generates synthetic bin and vehicle datasets with realistic fill-history
time series. All parameters are drawn from config.py.

Usage:
    from src.simulate import generate_bins, generate_vehicles, generate_fill_history
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    RANDOM_SEED,
    NUM_BINS,
    BIN_CAPACITY_MIN,
    BIN_CAPACITY_MAX,
    ACCUMULATION_PROFILES,
    ZONE_DISTRIBUTION,
    HISTORY_HOURS,
    HISTORY_INTERVAL_HOURS,
    COLLECTION_THRESHOLD,
    NUM_VEHICLES,
    VEHICLE_CAPACITY_LITERS,
    DEPOT_LOCATION,
    DEMO_BBOX,
    ROAD_GRAPH_CACHE,
)


def _load_road_graph():
    """Load the cached OSM road graph. Returns the graph object."""
    import osmnx as ox

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_path = os.path.join(project_root, ROAD_GRAPH_CACHE)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Road graph not found at {graph_path}. "
            "Run 'python scripts/cache_road_graph.py' first."
        )

    return ox.load_graphml(graph_path)


def _get_road_network_nodes(G):
    """Extract node coordinates from the road graph as a DataFrame."""
    import osmnx as ox

    nodes = ox.graph_to_gdfs(G, edges=False)[["y", "x"]]
    nodes.columns = ["latitude", "longitude"]
    return nodes


def generate_bins(G=None, seed=RANDOM_SEED, num_bins=NUM_BINS):
    """
    Generate bin locations snapped to real road-network nodes.

    Parameters
    ----------
    G : networkx.MultiDiGraph, optional
        Cached OSM road graph. If None, loads from cache.
    seed : int
        Random seed for reproducibility.
    num_bins : int
        Number of bins to generate.

    Returns
    -------
    pd.DataFrame
        Bin dataset with columns: bin_id, latitude, longitude,
        capacity_liters, zone, avg_accumulation_rate, node_id
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    if G is None:
        G = _load_road_graph()

    # Get all road network nodes
    nodes_df = _get_road_network_nodes(G)

    # Filter nodes within the demo bounding box
    mask = (
        (nodes_df["latitude"] >= DEMO_BBOX["south"])
        & (nodes_df["latitude"] <= DEMO_BBOX["north"])
        & (nodes_df["longitude"] >= DEMO_BBOX["west"])
        & (nodes_df["longitude"] <= DEMO_BBOX["east"])
    )
    valid_nodes = nodes_df[mask]

    if len(valid_nodes) < num_bins:
        raise ValueError(
            f"Only {len(valid_nodes)} road nodes in bounding box, "
            f"but {num_bins} bins requested. Expand the bounding box."
        )

    # Randomly sample node positions for bins
    sampled_indices = rng.choice(len(valid_nodes), size=num_bins, replace=False)
    sampled_nodes = valid_nodes.iloc[sampled_indices]

    # Assign zones based on configured distribution
    zones = []
    zone_names = list(ZONE_DISTRIBUTION.keys())
    zone_probs = list(ZONE_DISTRIBUTION.values())
    for _ in range(num_bins):
        zones.append(rng.choice(zone_names, p=zone_probs))

    # Generate accumulation rates per zone
    accumulation_rates = []
    for zone in zones:
        profile = ACCUMULATION_PROFILES[zone]
        rate = max(0.1, rng.normal(profile["mean"], profile["std"]))
        accumulation_rates.append(round(rate, 2))

    # Build the bins DataFrame
    bins_df = pd.DataFrame(
        {
            "bin_id": [f"BIN-{i+1:03d}" for i in range(num_bins)],
            "latitude": sampled_nodes["latitude"].values,
            "longitude": sampled_nodes["longitude"].values,
            "capacity_liters": rng.integers(
                BIN_CAPACITY_MIN, BIN_CAPACITY_MAX + 1, size=num_bins
            ),
            "zone": zones,
            "avg_accumulation_rate": accumulation_rates,
            "node_id": sampled_nodes.index.values,
        }
    )

    return bins_df


def generate_fill_history(
    bins_df,
    seed=RANDOM_SEED,
    history_hours=HISTORY_HOURS,
    interval_hours=HISTORY_INTERVAL_HOURS,
    collection_threshold=COLLECTION_THRESHOLD,
):
    """
    Generate synthetic fill-level time series for each bin.

    Simulates fill accumulation with periodic collections (resets) when
    the fill level exceeds the collection threshold, mimicking real-world
    waste collection patterns.

    Parameters
    ----------
    bins_df : pd.DataFrame
        Bin dataset from generate_bins().
    seed : int
        Random seed.
    history_hours : int
        Number of hours of history to simulate.
    interval_hours : int
        Time between readings (hours).
    collection_threshold : float
        Fill % at which the bin is "collected" (reset to near-zero).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: bin_id, timestamp, fill_pct
    """
    rng = np.random.default_rng(seed + 1)  # offset seed for variation

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    num_readings = history_hours // interval_hours

    records = []

    for _, bin_row in bins_df.iterrows():
        bin_id = bin_row["bin_id"]
        base_rate = bin_row["avg_accumulation_rate"]  # % per hour
        fill = rng.uniform(5, 30)  # start at a random low level

        last_collected_at = now - timedelta(hours=history_hours)

        for step in range(num_readings):
            timestamp = now - timedelta(hours=(history_hours - step * interval_hours))

            # Diurnal modulation: higher accumulation during peak hours
            hour_of_day = timestamp.hour
            if 7 <= hour_of_day <= 9:       # Morning peak
                diurnal_factor = 1.6
            elif 17 <= hour_of_day <= 19:    # Evening peak
                diurnal_factor = 1.8
            elif 11 <= hour_of_day <= 14:    # Midday moderate
                diurnal_factor = 1.2
            elif 22 <= hour_of_day or hour_of_day <= 5:  # Overnight low
                diurnal_factor = 0.4
            else:
                diurnal_factor = 1.0

            # Add noise to the accumulation rate (±30%)
            noise = rng.normal(1.0, 0.15)
            hourly_increment = base_rate * interval_hours * max(0.3, noise) * diurnal_factor

            fill += hourly_increment
            fill = min(fill, 100.0)  # cap at 100%

            records.append(
                {
                    "bin_id": bin_id,
                    "timestamp": timestamp,
                    "fill_pct": round(fill, 1),
                }
            )

            # Simulate collection: reset if above threshold
            if fill >= collection_threshold:
                fill = rng.uniform(2, 10)  # small residual after emptying
                last_collected_at = timestamp

    history_df = pd.DataFrame(records)
    return history_df


def get_current_state(bins_df, history_df):
    """
    Extract the latest state for each bin from the fill history.

    Returns
    -------
    pd.DataFrame
        bins_df enriched with: current_fill_pct, last_collected_at,
        hours_since_collection
    """
    # Get the latest reading per bin
    latest = (
        history_df.sort_values("timestamp")
        .groupby("bin_id")
        .last()
        .reset_index()[["bin_id", "timestamp", "fill_pct"]]
    )
    latest.columns = ["bin_id", "last_reading_at", "current_fill_pct"]

    # Estimate last collection time: find the last time fill dropped significantly
    last_collections = []
    for bin_id in bins_df["bin_id"]:
        bin_hist = history_df[history_df["bin_id"] == bin_id].sort_values("timestamp")
        fills = bin_hist["fill_pct"].values
        timestamps = bin_hist["timestamp"].values

        # Find the last significant drop (collection event)
        last_coll = timestamps[0]  # default: beginning of history
        for i in range(1, len(fills)):
            if fills[i] < fills[i - 1] - 20:  # drop of >20% = collection
                last_coll = timestamps[i]

        last_collections.append(
            {"bin_id": bin_id, "last_collected_at": pd.Timestamp(last_coll)}
        )

    collections_df = pd.DataFrame(last_collections)

    # Merge everything
    state_df = bins_df.merge(latest, on="bin_id").merge(collections_df, on="bin_id")

    # Calculate hours since last collection
    now = pd.Timestamp(datetime.now().replace(minute=0, second=0, microsecond=0))
    state_df["hours_since_collection"] = (
        (now - state_df["last_collected_at"]).dt.total_seconds() / 3600
    ).round(1)

    return state_df


def generate_vehicles(
    seed=RANDOM_SEED,
    num_vehicles=NUM_VEHICLES,
    capacity=VEHICLE_CAPACITY_LITERS,
):
    """
    Generate vehicle fleet dataset.

    Returns
    -------
    pd.DataFrame
        Vehicle dataset with columns: vehicle_id, capacity_liters,
        depot_latitude, depot_longitude, available
    """
    vehicles = []
    for i in range(num_vehicles):
        vehicles.append(
            {
                "vehicle_id": f"VEH-{i+1:02d}",
                "capacity_liters": capacity,
                "depot_latitude": DEPOT_LOCATION["latitude"],
                "depot_longitude": DEPOT_LOCATION["longitude"],
                "available": True,
            }
        )

    return pd.DataFrame(vehicles)


def snap_depot_to_graph(G=None):
    """
    Snap the depot location to the nearest road-network node.

    Returns
    -------
    int
        OSM node ID for the depot.
    """
    import osmnx as ox

    if G is None:
        G = _load_road_graph()

    depot_node = ox.distance.nearest_nodes(
        G, DEPOT_LOCATION["longitude"], DEPOT_LOCATION["latitude"]
    )
    return depot_node


def save_datasets(bins_df, vehicles_df, history_df, data_dir=None):
    """Save generated datasets to CSV files in the data directory."""
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
        )
    os.makedirs(data_dir, exist_ok=True)

    bins_df.to_csv(os.path.join(data_dir, "bins.csv"), index=False)
    vehicles_df.to_csv(os.path.join(data_dir, "vehicles.csv"), index=False)
    history_df.to_csv(os.path.join(data_dir, "fill_history.csv"), index=False)

    print(f"  Saved bins.csv ({len(bins_df)} bins)")
    print(f"  Saved vehicles.csv ({len(vehicles_df)} vehicles)")
    print(f"  Saved fill_history.csv ({len(history_df)} readings)")


if __name__ == "__main__":
    print("BinSense - Data Generation")
    print("-" * 40)

    print("Loading road graph...")
    G = _load_road_graph()

    print(f"Generating {NUM_BINS} bins...")
    bins_df = generate_bins(G)

    print(f"Generating {NUM_VEHICLES} vehicles...")
    vehicles_df = generate_vehicles()

    print(f"Generating {HISTORY_HOURS}h fill history...")
    history_df = generate_fill_history(bins_df)

    print("Extracting current state...")
    state_df = get_current_state(bins_df, history_df)

    print("\nSaving datasets...")
    save_datasets(bins_df, vehicles_df, history_df)

    print("\n--- Sample Current State ---")
    print(
        state_df[
            ["bin_id", "zone", "current_fill_pct", "avg_accumulation_rate",
             "hours_since_collection"]
        ].to_string(index=False)
    )
    print("\nZone distribution:")
    print(bins_df["zone"].value_counts().to_string())
    print("\nFill level distribution:")
    print(state_df["current_fill_pct"].describe().to_string())
