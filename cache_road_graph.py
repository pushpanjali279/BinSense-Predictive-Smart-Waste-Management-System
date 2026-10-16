"""
BinSense — Road-Network Graph Download & Cache

Downloads the OSM road graph for the configured demo area and caches it
locally as a GraphML file so the app can run fully offline afterward.

Usage:
    python scripts/cache_road_graph.py
"""

import os
import sys

# Add parent directory to path so we can import src.config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DEMO_BBOX, NETWORK_TYPE, ROAD_GRAPH_CACHE


def download_and_cache_graph():
    """Download OSM road graph for the demo area and save as GraphML."""
    import osmnx as ox

    print(f"Downloading OSM road graph for demo area...")
    print(f"  Bounding box: {DEMO_BBOX}")
    print(f"  Network type: {NETWORK_TYPE}")

    # osmnx 2.x: graph_from_bbox takes bbox=(west, south, east, north)
    G = ox.graph_from_bbox(
        bbox=(DEMO_BBOX["west"], DEMO_BBOX["south"],
              DEMO_BBOX["east"], DEMO_BBOX["north"]),
        network_type=NETWORK_TYPE,
    )

    # Add travel time (minutes) as edge weight based on road length and speed
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)

    # Ensure data directory exists
    cache_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ROAD_GRAPH_CACHE,
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Save to GraphML
    ox.save_graphml(G, cache_path)

    node_count = len(G.nodes)
    edge_count = len(G.edges)
    print(f"  Graph saved to: {cache_path}")
    print(f"  Nodes: {node_count}, Edges: {edge_count}")
    print("  Road graph cached — app can now run fully offline.")

    return G


if __name__ == "__main__":
    download_and_cache_graph()
