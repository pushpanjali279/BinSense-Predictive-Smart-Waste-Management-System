"""
BinSense — CVRP Route Optimization Module (Google OR-Tools)

Solves the Capacitated Vehicle Routing Problem using OR-Tools,
producing per-vehicle optimized routes on real road-network distances.

Usage:
    from src.routing import solve_routes
"""

import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

from src.config import (
    NUM_VEHICLES,
    VEHICLE_CAPACITY_LITERS,
    CVRP_TIME_LIMIT_SECONDS,
    CVRP_FIRST_SOLUTION_STRATEGY,
    CVRP_LOCAL_SEARCH_METAHEURISTIC,
)


def _get_first_solution_strategy(name):
    """Map config string to OR-Tools enum."""
    strategies = {
        "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
        "PATH_MOST_CONSTRAINED_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
        "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
        "SWEEP": routing_enums_pb2.FirstSolutionStrategy.SWEEP,
        "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
        "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    }
    return strategies.get(name, routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)


def _get_metaheuristic(name):
    """Map config string to OR-Tools enum."""
    metaheuristics = {
        "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
        "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        "GREEDY_DESCENT": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    }
    return metaheuristics.get(name, routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)


def solve_routes(
    distance_matrix,
    demands,
    num_vehicles=NUM_VEHICLES,
    vehicle_capacities=None,
    depot_index=0,
    time_limit_seconds=CVRP_TIME_LIMIT_SECONDS,
):
    """
    Solve the Capacitated Vehicle Routing Problem using Google OR-Tools.

    Parameters
    ----------
    distance_matrix : np.ndarray or list[list[int]]
        (N+1) x (N+1) distance matrix. Index 0 = depot.
        Values should be integers (meters).
    demands : list[int]
        Demand (waste volume in liters) per node. Index 0 = depot (demand=0).
    num_vehicles : int
        Number of available vehicles.
    vehicle_capacities : list[int], optional
        Per-vehicle capacity in liters. Defaults to uniform VEHICLE_CAPACITY_LITERS.
    depot_index : int
        Index of the depot node in the matrix (default 0).
    time_limit_seconds : int
        Maximum solver time.

    Returns
    -------
    dict
        {
            "status": str ("OPTIMAL"/"FEASIBLE"/"NO_SOLUTION"),
            "routes": list[dict],  # per-vehicle route info
            "total_distance": int,
            "total_load": int,
        }
        Each route dict:
        {
            "vehicle_id": int,
            "stops": list[int],        # matrix indices (excl. depot at start/end)
            "full_path": list[int],     # matrix indices (incl. depot at start/end)
            "distance": int,           # total route distance (meters)
            "load": int,               # total load carried (liters)
            "num_stops": int,
        }
    """
    if vehicle_capacities is None:
        vehicle_capacities = [VEHICLE_CAPACITY_LITERS] * num_vehicles

    n = len(distance_matrix)

    # Validate inputs
    assert len(demands) == n, f"Demands length {len(demands)} != matrix size {n}"
    assert demands[depot_index] == 0, "Depot demand must be 0"

    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot_index)

    # Create the routing model
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(demands[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        vehicle_capacities,
        True,  # start cumul to zero
        "Capacity",
    )

    # Add disjunctions with a large penalty so if capacity is tight,
    # the solver drops the least critical nodes rather than failing completely
    PENALTY = 1_000_000
    for node in range(n):
        if node != depot_index:
            routing.AddDisjunction([manager.NodeToIndex(node)], PENALTY)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = _get_first_solution_strategy(
        CVRP_FIRST_SOLUTION_STRATEGY
    )
    search_parameters.local_search_metaheuristic = _get_metaheuristic(
        CVRP_LOCAL_SEARCH_METAHEURISTIC
    )
    search_parameters.time_limit.seconds = time_limit_seconds

    # Solve
    solution = routing.SolveWithParameters(search_parameters)

    # If first choice strategy failed, fallback to PARALLEL_CHEAPEST_INSERTION or AUTOMATIC
    if not solution:
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        return {
            "status": "NO_SOLUTION",
            "routes": [],
            "total_distance": 0,
            "total_load": 0,
            "dropped_nodes": list(range(1, n)),
        }

    # Extract solution
    status = "OPTIMAL" if routing.status() == 1 else "FEASIBLE"
    routes = []
    total_distance = 0
    total_load = 0
    visited_nodes = set()

    for vehicle_id in range(num_vehicles):
        route_distance = 0
        route_load = 0
        stops = []
        full_path = []

        index = routing.Start(vehicle_id)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            full_path.append(node)
            if node != depot_index:
                stops.append(node)
                visited_nodes.add(node)
                route_load += demands[node]

            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )

        # Add depot at end
        full_path.append(depot_index)

        routes.append(
            {
                "vehicle_id": vehicle_id,
                "stops": stops,
                "full_path": full_path,
                "distance": route_distance,
                "load": route_load,
                "num_stops": len(stops),
            }
        )

        total_distance += route_distance
        total_load += route_load

    dropped_nodes = [node for node in range(n) if node != depot_index and node not in visited_nodes]

    return {
        "status": status,
        "routes": routes,
        "total_distance": total_distance,
        "total_load": total_load,
        "dropped_nodes": dropped_nodes,
    }


def format_route_results(solution, bin_ids, vehicle_capacities=None):
    """
    Format solver output with bin IDs and capacity utilization.

    Parameters
    ----------
    solution : dict
        Output from solve_routes().
    bin_ids : list[str]
        Bin IDs corresponding to matrix indices 1..N.
    vehicle_capacities : list[int], optional
        Per-vehicle capacity.

    Returns
    -------
    list[dict]
        Enriched per-vehicle route summaries.
    """
    if vehicle_capacities is None:
        vehicle_capacities = [VEHICLE_CAPACITY_LITERS] * len(solution["routes"])

    formatted = []
    for route in solution["routes"]:
        vid = route["vehicle_id"]
        cap = vehicle_capacities[vid]

        # Map matrix indices to bin IDs (index 0 = depot, 1+ = bins)
        stop_bin_ids = [bin_ids[s - 1] for s in route["stops"]]

        formatted.append(
            {
                "vehicle_id": vid,
                "vehicle_label": f"VEH-{vid + 1:02d}",
                "stops": route["stops"],
                "stop_bin_ids": stop_bin_ids,
                "stops_bin_ids": stop_bin_ids,
                "num_stops": route["num_stops"],
                "distance_m": route["distance"],
                "distance_km": round(route["distance"] / 1000, 2),
                "load_liters": route["load"],
                "capacity_liters": cap,
                "utilization_pct": round(route["load"] / cap * 100, 1) if cap > 0 else 0,
                "full_path_indices": route["full_path"],
            }
        )

    return formatted


if __name__ == "__main__":
    print("BinSense - CVRP Solver Test")
    print("-" * 40)

    # Simple test with a small distance matrix
    # 0 = depot, 1-4 = bins
    test_matrix = [
        [0, 500, 800, 1200, 700],
        [500, 0, 400, 900, 1100],
        [800, 400, 0, 600, 800],
        [1200, 900, 600, 0, 500],
        [700, 1100, 800, 500, 0],
    ]
    test_demands = [0, 200, 300, 250, 350]  # liters

    print(f"Solving CVRP: {len(test_matrix)-1} bins, 2 vehicles...")
    result = solve_routes(
        test_matrix,
        test_demands,
        num_vehicles=2,
        vehicle_capacities=[600, 600],
    )

    print(f"Status: {result['status']}")
    print(f"Total distance: {result['total_distance']}m")
    print(f"Total load: {result['total_load']}L")

    for route in result["routes"]:
        print(f"\n  Vehicle {route['vehicle_id']}:")
        print(f"    Path: {route['full_path']}")
        print(f"    Stops: {route['stops']}")
        print(f"    Distance: {route['distance']}m")
        print(f"    Load: {route['load']}L")
