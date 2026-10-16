"""
BinSense — Baseline Comparator Module

Implements the naive baseline approach for before/after comparison:
- Threshold-based bin selection (fixed 80% threshold, no prediction)
- Straight-line (Euclidean) distance routing (nearest-neighbor heuristic)

Usage:
    from src.baseline import baseline_routes, compare_results
"""

import math
import numpy as np
import pandas as pd

from src.config import BASELINE_FILL_THRESHOLD, NUM_VEHICLES, VEHICLE_CAPACITY_LITERS

# Euclidean-to-road correction factor: urban roads are typically 30-40%
# longer than straight-line distance. Applied to baseline for fair comparison.
ROAD_DISTANCE_MULTIPLIER = 1.35


def _haversine(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in meters between two lat/lon points."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _build_euclidean_matrix(depot_lat, depot_lon, bins_df):
    """Build a straight-line distance matrix (meters)."""
    lats = [depot_lat] + bins_df["latitude"].tolist()
    lons = [depot_lon] + bins_df["longitude"].tolist()
    n = len(lats)

    matrix = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            d = int(_haversine(lats[i], lons[i], lats[j], lons[j]))
            matrix[i][j] = d
            matrix[j][i] = d

    return matrix


def _nearest_neighbor_route(distance_matrix, stops, depot=0, capacity=None, demand=None):
    """
    Simple nearest-neighbor heuristic for a single vehicle.

    Returns an ordered list of stop indices and total distance.
    """
    if not stops:
        return [], 0

    unvisited = set(stops)
    route = []
    current = depot
    total_distance = 0
    total_load = 0

    while unvisited:
        nearest = None
        nearest_dist = float("inf")

        for s in unvisited:
            d = distance_matrix[current][s]
            if d < nearest_dist:
                # Check capacity constraint if provided
                if capacity is not None and demand is not None:
                    if total_load + demand[s] > capacity:
                        continue
                nearest = s
                nearest_dist = d

        if nearest is None:
            break  # Can't fit any more stops

        route.append(nearest)
        total_distance += nearest_dist
        if demand is not None:
            total_load += demand[nearest]
        unvisited.discard(nearest)
        current = nearest

    # Return to depot
    if route:
        total_distance += distance_matrix[current][depot]

    return route, total_distance, total_load


def baseline_routes(
    bins_df,
    depot_lat,
    depot_lon,
    threshold=BASELINE_FILL_THRESHOLD,
    num_vehicles=NUM_VEHICLES,
    vehicle_capacity=VEHICLE_CAPACITY_LITERS,
):
    """
    Compute baseline routes using naive threshold + straight-line distance.

    Parameters
    ----------
    bins_df : pd.DataFrame
        Full bin dataset with current_fill_pct, latitude, longitude,
        capacity_liters columns.
    depot_lat, depot_lon : float
        Depot coordinates.
    threshold : float
        Fill % threshold for naive selection (default 80%).
    num_vehicles : int
        Number of available vehicles.
    vehicle_capacity : int
        Per-vehicle capacity in liters.

    Returns
    -------
    dict
        {
            "selected_bins": pd.DataFrame,
            "routes": list[dict],
            "total_distance": int,   # meters (straight-line)
            "total_load": int,
            "num_bins_selected": int,
            "bins_at_overflow_risk": int,
            "method": "baseline",
        }
    """
    # Step 1: Naive threshold selection (no prediction)
    selected = bins_df[bins_df["current_fill_pct"] >= threshold].copy()
    selected = selected.sort_values("current_fill_pct", ascending=False)
    selected = selected.reset_index(drop=True)

    if selected.empty:
        return {
            "selected_bins": selected,
            "routes": [],
            "total_distance": 0,
            "total_load": 0,
            "num_bins_selected": 0,
            "bins_at_overflow_risk": 0,
            "method": "baseline",
        }

    # Step 2: Build straight-line distance matrix
    matrix = _build_euclidean_matrix(depot_lat, depot_lon, selected)

    # Estimate demands
    demands = [0]  # depot
    for _, row in selected.iterrows():
        demands.append(int(row["current_fill_pct"] / 100 * row["capacity_liters"]))

    # Step 3: Simple greedy assignment — split bins across vehicles
    all_stops = list(range(1, len(selected) + 1))
    routes = []
    total_distance = 0
    total_load = 0
    remaining_stops = set(all_stops)

    for v in range(num_vehicles):
        if not remaining_stops:
            routes.append({
                "vehicle_id": v,
                "stops": [],
                "distance": 0,
                "load": 0,
                "num_stops": 0,
            })
            continue

        route_stops, route_dist, route_load = _nearest_neighbor_route(
            matrix,
            list(remaining_stops),
            depot=0,
            capacity=vehicle_capacity,
            demand=demands,
        )

        remaining_stops -= set(route_stops)

        stop_bin_ids = [selected.iloc[s - 1]["bin_id"] for s in route_stops]

        routes.append({
            "vehicle_id": v,
            "vehicle_label": f"VEH-{v + 1:02d}",
            "stops": route_stops,
            "stop_bin_ids": stop_bin_ids,
            "distance": route_dist,
            "distance_km": round(route_dist / 1000, 2),
            "load": route_load,
            "num_stops": len(route_stops),
            "utilization_pct": round(route_load / vehicle_capacity * 100, 1),
        })

        total_distance += route_dist
        total_load += route_load

    # Count bins that are at overflow risk (above 90%)
    bins_at_risk = int((bins_df["current_fill_pct"] >= 90).sum())

    return {
        "selected_bins": selected,
        "routes": routes,
        "total_distance": total_distance,
        "total_distance_km": round(total_distance / 1000, 2),
        "total_load": total_load,
        "num_bins": len(selected),
        "num_bins_selected": len(selected),
        "bins_at_overflow_risk": bins_at_risk,
        "method": "baseline",
    }


def compare_results(binsense_result, baseline_result):
    """
    Compare BinSense (optimized) vs. baseline results.

    Parameters
    ----------
    binsense_result : dict
        Results from the BinSense pipeline (CVRP).
    baseline_result : dict
        Results from the baseline method.

    Returns
    -------
    dict
        Comparison metrics.
    """
    bs_dist = binsense_result.get("total_distance", 0)
    bl_dist = baseline_result.get("total_distance", 0)

    # Apply road-distance correction to baseline's Euclidean distance
    # so the comparison is fair (both represent real-world road distances)
    bl_dist_corrected = int(bl_dist * ROAD_DISTANCE_MULTIPLIER)

    if bl_dist_corrected > 0:
        distance_saved_pct = round((bl_dist_corrected - bs_dist) / bl_dist_corrected * 100, 1)
        distance_saved_m = bl_dist_corrected - bs_dist
    else:
        distance_saved_pct = 0
        distance_saved_m = 0

    bs_bins = binsense_result.get("num_bins_selected",
                                   len(binsense_result.get("routes", [])))
    bl_bins = baseline_result.get("num_bins_selected", 0)

    # Count active vehicles (with at least 1 stop)
    bs_active = sum(1 for r in binsense_result.get("routes", [])
                    if r.get("num_stops", 0) > 0)
    bl_active = sum(1 for r in baseline_result.get("routes", [])
                    if r.get("num_stops", 0) > 0)

    bs_load = binsense_result.get("total_load", 0)
    bl_load = baseline_result.get("total_load", 0)

    return {
        "binsense_distance_m": bs_dist,
        "binsense_distance_km": round(bs_dist / 1000, 2),
        "baseline_distance_m": bl_dist,
        "baseline_distance_km": round(bl_dist / 1000, 2),
        "baseline_distance_corrected_m": bl_dist_corrected,
        "baseline_distance_corrected_km": round(bl_dist_corrected / 1000, 2),
        "distance_saved_m": distance_saved_m,
        "distance_saved_km": round(distance_saved_m / 1000, 2),
        "distance_saved_pct": distance_saved_pct,
        "binsense_bins": bs_bins,
        "baseline_bins": bl_bins,
        "bins_serviced_diff": bs_bins - bl_bins,
        "binsense_active_vehicles": bs_active,
        "baseline_active_vehicles": bl_active,
        "binsense_total_load": bs_load,
        "binsense_load_liters": bs_load,
        "baseline_total_load": bl_load,
        "baseline_load_liters": bl_load,
    }
