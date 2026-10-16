"""
BinSense — End-to-End Pipeline Test

Runs the full pipeline: simulate -> predict -> prioritize -> select ->
distance matrix -> CVRP solve -> baseline compare.

Validates that all modules work together and produces the complete
result set needed by the dashboard.
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEPOT_LOCATION, NUM_VEHICLES, VEHICLE_CAPACITY_LITERS


def run_pipeline():
    print("=" * 60)
    print("BinSense - End-to-End Pipeline Test")
    print("=" * 60)

    # Phase 1: Data Layer
    print("\n[Phase 1] Data Layer")
    t0 = time.time()
    from src.simulate import (
        generate_bins, generate_vehicles, generate_fill_history,
        get_current_state, snap_depot_to_graph, save_datasets,
    )
    from src.network import load_road_graph

    G = load_road_graph()
    bins_df = generate_bins(G)
    vehicles_df = generate_vehicles()
    history_df = generate_fill_history(bins_df)
    state_df = get_current_state(bins_df, history_df)
    depot_node = snap_depot_to_graph(G)

    save_datasets(bins_df, vehicles_df, history_df)

    print(f"  Bins: {len(bins_df)}, Vehicles: {len(vehicles_df)}")
    print(f"  History: {len(history_df)} readings")
    print(f"  Depot node: {depot_node}")
    print(f"  Zone distribution: {bins_df['zone'].value_counts().to_dict()}")
    print(f"  [OK] Phase 1 done in {time.time()-t0:.1f}s")

    # Phase 2: Prediction
    print("\n[Phase 2] Prediction")
    t0 = time.time()
    from src.predict import predict_all_bins

    pred_df = predict_all_bins(state_df, history_df)

    high_risk = (pred_df["predicted_fill_pct"] >= 90).sum()
    print(f"  Predicted high-risk bins (>=90%): {high_risk}")
    print(f"  Avg predicted fill: {pred_df['predicted_fill_pct'].mean():.1f}%")
    overflow_finite = pred_df[pred_df["time_to_overflow_hours"] < float("inf")]
    if not overflow_finite.empty:
        print(f"  Avg time-to-overflow: {overflow_finite['time_to_overflow_hours'].mean():.1f}h")
    print(f"  [OK] Phase 2 done in {time.time()-t0:.1f}s")

    # Phase 3: Priority Engine
    print("\n[Phase 3] Priority Engine")
    t0 = time.time()
    from src.priority import score_all_bins, get_tier_summary

    scored_df = score_all_bins(pred_df)
    summary = get_tier_summary(scored_df)

    print(f"  Tier summary: {summary}")
    print(f"  Score range: {scored_df['priority_score'].min():.1f} - {scored_df['priority_score'].max():.1f}")
    print(f"  [OK] Phase 3 done in {time.time()-t0:.1f}s")

    # Phase 4: Bin Selection
    print("\n[Phase 4] Bin Selection")
    t0 = time.time()
    from src.select import select_bins

    selection = select_bins(scored_df)
    selected = selection["selected"]

    print(f"  Selected: {len(selected)} bins")
    print(f"  Total demand: {selection['total_demand_liters']:.0f}L / {selection['fleet_capacity_liters']:.0f}L")
    print(f"  Capacity exceeded: {selection['capacity_exceeded']}")
    if selection["deferred_count"] > 0:
        print(f"  Deferred: {selection['deferred_count']} bins")
    print(f"  [OK] Phase 4 done in {time.time()-t0:.1f}s")

    if selected.empty:
        print("\n  No bins selected for collection - pipeline stops here.")
        print("  (This is a valid state per the spec)")
        return

    # Phase 5: Distance Matrix
    print("\n[Phase 5] Road Network Distance Matrix")
    t0 = time.time()
    from src.network import build_distance_matrix

    matrix_result = build_distance_matrix(G, selected_bins_df=selected)

    n = len(matrix_result["distance_matrix"])
    print(f"  Matrix size: {n}x{n} (1 depot + {n-1} bins)")
    nonzero = matrix_result["distance_matrix"][matrix_result["distance_matrix"] > 0]
    print(f"  Distance range: {nonzero.min()}m - {nonzero.max()}m")
    print(f"  [OK] Phase 5 done in {time.time()-t0:.1f}s")

    # Phase 6: CVRP Solver
    print("\n[Phase 6] CVRP Solver")
    t0 = time.time()
    from src.routing import solve_routes, format_route_results

    # Build demands list: depot=0, then bin demands mapped by bin_ids in matrix
    bin_id_to_demand = {
        row["bin_id"]: int(row["current_fill_pct"] / 100 * row["capacity_liters"])
        for _, row in selected.iterrows()
    }
    demands = [0] + [bin_id_to_demand[bid] for bid in matrix_result["bin_ids"]]

    solution = solve_routes(
        matrix_result["distance_matrix"],
        demands,
        num_vehicles=NUM_VEHICLES,
        vehicle_capacities=[VEHICLE_CAPACITY_LITERS] * NUM_VEHICLES,
    )

    print(f"  Status: {solution['status']}")
    print(f"  Total distance: {solution['total_distance']}m ({solution['total_distance']/1000:.2f}km)")
    print(f"  Total load: {solution['total_load']}L")

    formatted = format_route_results(solution, matrix_result["bin_ids"])
    for r in formatted:
        print(f"    {r['vehicle_label']}: {r['num_stops']} stops, "
              f"{r['distance_km']}km, {r['load_liters']}L "
              f"({r['utilization_pct']}% cap)")
    print(f"  [OK] Phase 6 done in {time.time()-t0:.1f}s")

    # Phase 7: Baseline Comparison
    print("\n[Phase 7] Baseline Comparison")
    t0 = time.time()
    from src.baseline import baseline_routes, compare_results

    baseline = baseline_routes(
        scored_df,
        DEPOT_LOCATION["latitude"],
        DEPOT_LOCATION["longitude"],
    )

    binsense_result = {
        "total_distance": solution["total_distance"],
        "total_load": solution["total_load"],
        "routes": formatted,
        "num_bins_selected": len(selected),
    }

    comparison = compare_results(binsense_result, baseline)

    print(f"  Baseline: {baseline['total_distance_km']}km | "
          f"BinSense: {comparison['binsense_distance_km']}km")
    print(f"  Distance saved: {comparison['distance_saved_km']}km "
          f"({comparison['distance_saved_pct']}%)")
    print(f"  Baseline bins: {comparison['baseline_bins']} | "
          f"BinSense bins: {comparison['binsense_bins']}")
    print(f"  [OK] Phase 7 done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 60)
    print("ALL PHASES PASSED - Pipeline runs end-to-end!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
