"""
BinSense — Road Network & Distance/Time Matrix Module

Uses the cached OSM road graph to compute shortest-path distance and
travel-time matrices between the depot and selected bins.

Usage:
    from src.network import build_distance_matrix, get_route_geometry
"""

import os
import numpy as np
import networkx as nx
import pandas as pd

from src.config import ROAD_GRAPH_CACHE, DEPOT_LOCATION, DISTANCE_SCALE_FACTOR


def load_road_graph():
    """Load the cached OSM road graph."""
    import osmnx as ox

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_path = os.path.join(project_root, ROAD_GRAPH_CACHE)

    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Road graph not found at {graph_path}. "
            "Run 'python scripts/cache_road_graph.py' first."
        )

    return ox.load_graphml(graph_path)


def snap_to_graph(G, latitude, longitude):
    """Snap a lat/lon point to the nearest road-network node."""
    import osmnx as ox
    return int(ox.distance.nearest_nodes(G, float(longitude), float(latitude)))


def build_distance_matrix(
    G,
    depot_lat=None,
    depot_lon=None,
    selected_bins_df=None,
    weight="length",
):
    """
    Build a road-network distance matrix between depot and selected bins.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Cached OSM road graph.
    depot_lat, depot_lon : float
        Depot coordinates. Defaults to config.
    selected_bins_df : pd.DataFrame
        Selected bins with latitude, longitude, and node_id columns.
    weight : str
        Edge weight to use for shortest paths.
        "length" = meters, "travel_time" = seconds.

    Returns
    -------
    dict
        {
            "distance_matrix": np.ndarray,   # (N+1) x (N+1), index 0 = depot
            "time_matrix": np.ndarray,        # (N+1) x (N+1) in seconds
            "node_ids": list[int],            # [depot_node, bin1_node, ...]
            "bin_ids": list[str],             # [bin_id_1, bin_id_2, ...]
            "node_to_bin": dict,              # {node_id: bin_id}
        }
    """
    import osmnx as ox

    if depot_lat is None:
        depot_lat = DEPOT_LOCATION["latitude"]
    if depot_lon is None:
        depot_lon = DEPOT_LOCATION["longitude"]

    # Snap depot to nearest graph node
    depot_node = snap_to_graph(G, depot_lat, depot_lon)

    # Get bin node IDs (snap if node_id not available)
    bin_nodes = []
    bin_ids = []
    node_to_bin = {}

    for _, row in selected_bins_df.iterrows():
        if "node_id" in row and pd.notna(row["node_id"]):
            node = int(row["node_id"])
        else:
            node = snap_to_graph(G, row["latitude"], row["longitude"])
        bin_nodes.append(node)
        bin_ids.append(row["bin_id"])
        node_to_bin[node] = row["bin_id"]

    # All nodes: depot (index 0) + bins
    all_nodes = [depot_node] + bin_nodes
    n = len(all_nodes)

    # Convert to undirected for shortest-path computation
    # (one-way streets shouldn't block VRP — vehicles can traverse both ways
    #  for routing purposes; the real constraint is distance, not direction)
    G_undirected = G.to_undirected()

    # First, filter out any bins not reachable from depot
    reachable_nodes = set()
    for node in all_nodes:
        try:
            nx.shortest_path_length(G_undirected, depot_node, node, weight="length")
            reachable_nodes.add(node)
        except nx.NetworkXNoPath:
            pass

    # Rebuild node list with only reachable bins
    filtered_bin_nodes = [nd for nd in bin_nodes if nd in reachable_nodes]
    filtered_bin_ids = [bin_ids[i] for i, nd in enumerate(bin_nodes) if nd in reachable_nodes]
    filtered_node_to_bin = {nd: node_to_bin[nd] for nd in filtered_bin_nodes if nd in node_to_bin}

    if len(filtered_bin_nodes) < len(bin_nodes):
        dropped = len(bin_nodes) - len(filtered_bin_nodes)
        print(f"  Warning: {dropped} bin(s) unreachable from depot, excluded from routing.")

    all_nodes = [depot_node] + filtered_bin_nodes
    bin_ids = filtered_bin_ids
    node_to_bin = filtered_node_to_bin
    n = len(all_nodes)

    # Compute distance matrix (meters) and time matrix (seconds)
    distance_matrix = np.zeros((n, n), dtype=np.int64)
    time_matrix = np.zeros((n, n), dtype=np.int64)

    # Use shortest path length on the undirected graph for all pairs
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            try:
                dist = nx.shortest_path_length(
                    G_undirected, all_nodes[i], all_nodes[j], weight="length"
                )
                distance_matrix[i][j] = int(dist * DISTANCE_SCALE_FACTOR)
            except nx.NetworkXNoPath:
                # Should not happen after filtering, but safeguard
                distance_matrix[i][j] = 999999

            try:
                time_val = nx.shortest_path_length(
                    G_undirected, all_nodes[i], all_nodes[j], weight="travel_time"
                )
                time_matrix[i][j] = int(time_val)
            except (nx.NetworkXNoPath, nx.NetworkXError, KeyError, TypeError):
                time_matrix[i][j] = 99999

    return {
        "distance_matrix": distance_matrix,
        "time_matrix": time_matrix,
        "node_ids": all_nodes,
        "bin_ids": bin_ids,
        "node_to_bin": node_to_bin,
        "depot_node": depot_node,
    }


def get_route_geometry(G, node_sequence):
    """
    Get the actual road geometry (lat/lon points) for a route.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Road graph.
    node_sequence : list[int]
        Ordered list of OSM node IDs defining the route.

    Returns
    -------
    list[tuple[float, float]]
        List of (latitude, longitude) points along the route.
    """
    G_undirected = G.to_undirected()
    route_coords = []

    for i in range(len(node_sequence) - 1):
        try:
            path = nx.shortest_path(
                G_undirected, node_sequence[i], node_sequence[i + 1], weight="length"
            )
            for node in path:
                node_data = G.nodes[node]
                route_coords.append((node_data["y"], node_data["x"]))
        except nx.NetworkXNoPath:
            # If no path, add direct line between nodes
            for node in [node_sequence[i], node_sequence[i + 1]]:
                node_data = G.nodes[node]
                route_coords.append((node_data["y"], node_data["x"]))

    return route_coords


def get_node_coords(G, node_id):
    """Get (latitude, longitude) for a graph node."""
    node_data = G.nodes[node_id]
    return (node_data["y"], node_data["x"])


if __name__ == "__main__":
    from src.simulate import generate_bins, _load_road_graph

    print("BinSense - Network Module Test")
    print("-" * 40)

    G = load_road_graph()
    bins_df = generate_bins(G)

    # Test with first 5 bins
    test_bins = bins_df.head(5)
    print(f"Building distance matrix for {len(test_bins)} bins + depot...")

    result = build_distance_matrix(G, selected_bins_df=test_bins)

    print(f"\nDistance matrix ({result['distance_matrix'].shape}):")
    print(f"  Labels: [DEPOT] + {result['bin_ids']}")
    print(f"  Matrix (meters):")
    print(result["distance_matrix"])

    print(f"\nTime matrix (seconds):")
    print(result["time_matrix"])
