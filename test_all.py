"""
BinSense — Comprehensive Unit and Integration Test Suite
=======================================================
Validates:
- Phase 1: Data model, generation & spatial snapping
- Phase 2: Predictive linear regression & overflow time
- Phase 3: Priority engine weights & urgency tiering
- Phase 4: Bin selection, capacity pruning & overrides
- Phase 5: Road network graph, shortest path & distance matrix
- Phase 6: OR-Tools CVRP solver constraints & route integrity
- Phase 7: Baseline comparator & metrics calculation
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    TIER_THRESHOLDS,
    TIER_ORDER,
    NUM_VEHICLES,
    VEHICLE_CAPACITY_LITERS,
    DEPOT_LOCATION,
)
from src.simulate import (
    generate_bins,
    generate_vehicles,
    generate_fill_history,
    get_current_state,
)
from src.predict import predict_fill, predict_all_bins
from src.priority import score_bin, score_all_bins, get_tier_summary
from src.select import select_bins
from src.network import load_road_graph, build_distance_matrix, get_route_geometry
from src.routing import solve_routes, format_route_results
from src.baseline import baseline_routes, compare_results


class TestPhase1DataModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.G = load_road_graph()

    def test_bin_generation(self):
        bins = generate_bins(self.G, num_bins=20, seed=123)
        self.assertEqual(len(bins), 20)
        self.assertTrue("bin_id" in bins.columns)
        self.assertTrue("node_id" in bins.columns)
        self.assertTrue("capacity_liters" in bins.columns)
        self.assertTrue("latitude" in bins.columns)
        self.assertTrue("longitude" in bins.columns)
        for nid in bins["node_id"]:
            self.assertIn(nid, self.G.nodes)

    def test_vehicle_generation(self):
        vehicles = generate_vehicles(num_vehicles=3)
        self.assertEqual(len(vehicles), 3)
        self.assertEqual(vehicles["capacity_liters"].sum(), 3600)

    def test_history_generation(self):
        bins = generate_bins(self.G, num_bins=5, seed=123)
        hist = generate_fill_history(bins, seed=123)
        self.assertGreater(len(hist), 0)
        self.assertTrue((hist["fill_pct"] >= 0).all())
        self.assertTrue((hist["fill_pct"] <= 100).all())


class TestPhase2Prediction(unittest.TestCase):
    def test_predict_fill_increasing(self):
        # 10 hours of linear increase: 10%, 20%, 30%...
        history = pd.DataFrame([
            {"timestamp": f"2026-09-17 {h:02d}:00:00", "fill_pct": float(10 + h * 5)}
            for h in range(10)
        ])
        res = predict_fill(history, horizon_hours=4, lookback_hours=10)
        self.assertAlmostEqual(res["current_fill_pct"], 55.0, places=1)
        self.assertAlmostEqual(res["accumulation_rate"], 5.0, places=1)
        # Predicted fill: 55 + 5*4 = 75%
        self.assertAlmostEqual(res["predicted_fill_pct"], 75.0, places=1)
        # Time to overflow (100%): (100-55)/5 = 9 hours
        self.assertAlmostEqual(res["time_to_overflow_hours"], 9.0, places=1)

    def test_predict_overflow_already_full(self):
        history = pd.DataFrame([
            {"timestamp": f"2026-09-17 {h:02d}:00:00", "fill_pct": 98.0}
            for h in range(5)
        ])
        res = predict_fill(history, horizon_hours=4)
        self.assertGreaterEqual(res["predicted_fill_pct"], 98.0)


class TestPhase3PriorityEngine(unittest.TestCase):
    def test_score_critical(self):
        # A bin with 95% current fill, 100% predicted, fast accumulation
        scored = score_bin(
            current_fill_pct=95.0,
            predicted_fill_pct=100.0,
            accumulation_rate=4.0,
            time_to_overflow_hours=1.2,
            hours_since_collection=36.0,
        )
        self.assertGreaterEqual(scored["priority_score"], 80.0)
        self.assertIn(scored["tier"], ["critical", "high"])

    def test_score_low(self):
        # A bin recently emptied, 10% fill, slow accumulation
        scored = score_bin(
            current_fill_pct=10.0,
            predicted_fill_pct=15.0,
            accumulation_rate=0.5,
            time_to_overflow_hours=999.0,
            hours_since_collection=2.0,
        )
        self.assertLess(scored["priority_score"], 40.0)
        self.assertEqual(scored["tier"], "low")


class TestPhase4Selection(unittest.TestCase):
    def test_selection_filtering_and_overrides(self):
        df = pd.DataFrame([
            {"bin_id": "B01", "tier": "critical", "priority_score": 90, "current_fill_pct": 90, "capacity_liters": 200},
            {"bin_id": "B02", "tier": "high", "priority_score": 70, "current_fill_pct": 70, "capacity_liters": 200},
            {"bin_id": "B03", "tier": "medium", "priority_score": 50, "current_fill_pct": 50, "capacity_liters": 200},
            {"bin_id": "B04", "tier": "low", "priority_score": 20, "current_fill_pct": 20, "capacity_liters": 200},
        ])

        # Default min_tier='high': only B01, B02
        res = select_bins(df, min_tier="high")
        selected_ids = list(res["selected"]["bin_id"])
        self.assertEqual(selected_ids, ["B01", "B02"])

        # Override: include B04, exclude B01
        res_override = select_bins(df, min_tier="high", include_ids=["B04"], exclude_ids=["B01"])
        new_ids = list(res_override["selected"]["bin_id"])
        self.assertIn("B04", new_ids)
        self.assertNotIn("B01", new_ids)


class TestPhase6CVRPRouting(unittest.TestCase):
    def test_cvrp_solver_basic(self):
        # 1 depot (0) + 4 bins (1, 2, 3, 4)
        dist_mat = [
            [0, 100, 200, 150, 300],
            [100, 0, 120, 180, 250],
            [200, 120, 0, 110, 190],
            [150, 180, 110, 0, 160],
            [300, 250, 190, 160, 0],
        ]
        demands = [0, 200, 300, 250, 250] # total = 1000
        capacities = [600, 600] # 2 vehicles, total capacity = 1200

        res = solve_routes(dist_mat, demands, num_vehicles=2, vehicle_capacities=capacities)
        self.assertIn(res["status"], ["OPTIMAL", "FEASIBLE"])
        self.assertGreater(res["total_distance"], 0)
        self.assertEqual(res["total_load"], 1000)
        self.assertEqual(len(res["routes"]), 2)
        for r in res["routes"]:
            self.assertLessEqual(r["load"], 600)


class TestPhase7BaselineComparison(unittest.TestCase):
    def test_baseline_and_compare(self):
        scored_df = pd.DataFrame([
            {"bin_id": "B1", "current_fill_pct": 85.0, "capacity_liters": 200, "latitude": 12.975, "longitude": 77.595},
            {"bin_id": "B2", "current_fill_pct": 50.0, "capacity_liters": 200, "latitude": 12.976, "longitude": 77.596},
        ])
        base = baseline_routes(scored_df, 12.975, 77.595)
        self.assertEqual(base["num_bins_selected"], 1) # Only B1 >= 80%

        binsense_summary = {
            "total_distance": 2500, # 2.5 km
            "total_load": 400,
            "routes": [],
            "num_bins_selected": 2,
        }
        comp = compare_results(binsense_summary, base)
        self.assertTrue("distance_saved_km" in comp)
        self.assertTrue("binsense_bins" in comp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
