import sys
sys.path.insert(0, ".")
from src.network import load_road_graph, build_distance_matrix
from src.simulate import generate_bins, generate_fill_history, get_current_state
from src.predict import predict_all_bins
from src.priority import score_all_bins
from src.select import select_bins
from src.routing import solve_routes
from src.config import NUM_VEHICLES, VEHICLE_CAPACITY_LITERS

G = load_road_graph()
bins = generate_bins(G)
history = generate_fill_history(bins)
state = get_current_state(bins, history)
preds = predict_all_bins(state, history)
scored = score_all_bins(preds)
selection = select_bins(scored)
selected = selection["selected"]

print(f"Selected {len(selected)} bins, total demand: {selection['total_demand_liters']}L")
mat = build_distance_matrix(G, selected_bins_df=selected)

# Map demands matching mat["bin_ids"]
bin_id_to_demand = {
    row["bin_id"]: int(row["current_fill_pct"] / 100 * row["capacity_liters"])
    for _, row in selected.iterrows()
}
demands = [0] + [bin_id_to_demand[bid] for bid in mat["bin_ids"]]
capacities = [VEHICLE_CAPACITY_LITERS] * NUM_VEHICLES

print(f"Demands (depot=0): {demands}")
print(f"Total demand: {sum(demands)} across {len(demands)-1} bins")
print(f"Fleet capacities: {capacities} (Total = {sum(capacities)})")

sol = solve_routes(
    mat["distance_matrix"],
    demands,
    num_vehicles=NUM_VEHICLES,
    vehicle_capacities=capacities,
)
print(f"CVRP result: status={sol['status']}, total_dist={sol['total_distance']}, total_load={sol['total_load']}")
