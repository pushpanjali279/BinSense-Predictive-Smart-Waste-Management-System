# 🧠 BinSense Project Memory & Implementation Record

> **Project:** BinSense — Predictive & Dynamic Smart Waste Management System  
> **Status:** All Phases (0 through 10) Completed & Verified  
> **Last Updated:** 2026-09-17  
> **Location:** `c:\Project\SWM\binsense`

---

## 1. Executive Summary

**BinSense** is a smart waste management prototype built for competition demonstration. It transforms municipal solid waste management from **reactive** (fixed truck schedules or reacting after citizen overflow complaints) to **predictive and dynamic** (forecasting fill levels, computing urgency tiers, and optimizing multi-vehicle routes on real road networks using Google OR-Tools).

---

## 2. Completed Work by Phase

### Phase 0: Environment Setup & Road Network Caching
- **Python Environment:** Created Python 3.11 64-bit virtual environment (`binsense/venv`) with 100% free/open-source libraries: `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `osmnx`, `networkx`, `ortools`, `streamlit`, `folium`, `streamlit-folium`, `plotly`.
- **Road Network Cache:** Downloaded OpenStreetMap road network for Bengaluru Central (CBD / Cubbon Park area: `north=12.985, south=12.965, east=77.610, west=77.585`) and cached offline in [`data/road_graph.graphml`](file:///c:/Project/SWM/binsense/data/road_graph.graphml) (664 nodes, 1,368 edges).

### Phase 1: Data Model & Simulator ([`src/simulate.py`](file:///c:/Project/SWM/binsense/src/simulate.py))
- **Smart Bins:** Generated 35 bins snapped to real OSM road-network nodes with capacities (120L–360L) and consumption zones (*commercial* fast, *residential* medium, *park* slow).
- **Fleet:** Generated 3 collection vehicles (1,200L capacity each = 3,600L fleet capacity).
- **Time-Series History:** Generated 72 hours of hourly sensor readings per bin with diurnal peak cycles and automated collection resets at 90% fill.
- **Datasets:** Exported to [`data/bins.csv`](file:///c:/Project/SWM/binsense/data/bins.csv), [`data/vehicles.csv`](file:///c:/Project/SWM/binsense/data/vehicles.csv), and [`data/fill_history.csv`](file:///c:/Project/SWM/binsense/data/fill_history.csv).

### Phase 2: Fill-Level Prediction ([`src/predict.py`](file:///c:/Project/SWM/binsense/src/predict.py))
- **Algorithm:** Linear regression on recent 24-hour fill history with collection-reset boundary detection.
- **Outputs:** Forecasted fill level `predicted_fill_pct` at future horizon $t + T$ (default $T=12$h), waste accumulation rate (%/hour), and `time_to_overflow_hours` until 100% full.
- **Safeguards:** Handled empty data, single readings, negative slopes, and already-overflowing bins.

### Phase 3: Dynamic Priority Engine ([`src/priority.py`](file:///c:/Project/SWM/binsense/src/priority.py))
- **Weighted Formula:** Normalized composite score (0–100) combining 5 operational signals:
  $$\text{Score} = w_1(\text{current}) + w_2(\text{predicted}) + w_3(\text{accumulation rate}) + w_4(\text{overflow risk}) + w_5(\text{time since collection})$$
  *(Weights: $w_1=0.20, w_2=0.30, w_3=0.15, w_4=0.25, w_5=0.10$ from [`src/config.py`](file:///c:/Project/SWM/binsense/src/config.py))*
- **Urgency Classification:**
  - **Critical** ($\ge 85$): Immediate collection required
  - **High** ($65 - 84$): Imminent overflow within forecast horizon
  - **Medium** ($40 - 64$): Routine collection candidate
  - **Low** ($< 40$): Safe to defer to subsequent cycles

### Phase 4: Bin Selection ([`src/select.py`](file:///c:/Project/SWM/binsense/src/select.py))
- **Filtering:** Includes bins at or above minimum urgency tier (default `medium`).
- **Fleet Capacity Budgeting:** Applied `FLEET_CAPACITY_BUFFER = 0.85` (85% of theoretical capacity) to guarantee feasible bin packing across vehicles.
- **Capacity Pruning:** Automatically defers lowest-priority qualifying bins if total demand exceeds the safe fleet budget.
- **Operator Overrides:** Allows manual inclusion/exclusion of any bin without altering priority scoring.

### Phase 5: Road Network & Distance Matrix ([`src/network.py`](file:///c:/Project/SWM/binsense/src/network.py))
- **Distance & Time Matrices:** Computes multi-source Dijkstra shortest paths on undirected road graph between central depot (`12.9750, 77.5950`) and all selected bins.
- **Geometry Extraction:** `get_route_geometry(G, node_sequence)` extracts true road coordinate waypoints along streets for realistic map rendering.
- **Disconnected Node Safeguard:** Automatically detects and filters unreachable nodes.

### Phase 6: CVRP Route Optimization ([`src/routing.py`](file:///c:/Project/SWM/binsense/src/routing.py))
- **Solver:** Google OR-Tools Capacitated Vehicle Routing Problem (CVRP).
- **Constraints & Heuristics:** Vehicle capacity dimensions, `PARALLEL_CHEAPEST_INSERTION` first solution strategy, and Guided Local Search metaheuristic.
- **Disjunctions:** Added node disjunctions with heavy penalties to guarantee feasible solutions even at near-capacity conditions.
- **Output:** Returns balanced per-vehicle routes with stop sequences, distances, loads, and vehicle utilization percentages.

### Phase 7: Baseline Comparator ([`src/baseline.py`](file:///c:/Project/SWM/binsense/src/baseline.py))
- **Naive Baseline:** Models traditional municipal operations (only collects bins currently $\ge 80\%$ fill, straight-line Euclidean nearest-neighbor routing).
- **Comparison Engine:** Produces quantitative Before/After metrics:
  - Bins serviced difference (typically +200% to +300% more high-risk bins proactively cleared)
  - Distance per liter collected (route productivity in L/km)
  - Theoretical vs. road network route comparison

### Phase 8: 5-Screen Operations Dashboard ([`app.py`](file:///c:/Project/SWM/binsense/app.py))
- Built with Streamlit, Folium, and Plotly following the Dark Ops Center design system (`#0E1912`, `#182620`, `#1E2F27`, `#6FAE7C`, `#E8A13D`, `#D9714E`, Space Grotesk and Inter fonts).
- **Screen 1 (Dashboard):** Real-time urgency summary tiles (Critical with pulse glow, High, Medium, Low), fleet capacity utilization bar, city-wide overflow risk status, and analytics charts.
- **Screen 2 (Priority Map):** Interactive Folium map with color-coded markers, pulsing Critical bins, and live data popovers.
- **Screen 3 (Priority Queue):** Ranked candidate table, live fleet demand counter, manual include/exclude controls, and "Build Routes" CTA.
- **Screen 4 (Route Optimizer):** Turn-by-turn road network paths, numbered stops, per-vehicle metric cards, and the **Before / After Baseline Toggle**.
- **Screen 5 (Fleet Dispatch):** Live vehicle status cards, distance saved callout banner, and "Complete Cycle & Reset" button that updates history and closes the feedback loop.

### Phase 9: Test Suite ([`tests/test_all.py`](file:///c:/Project/SWM/binsense/tests/test_all.py))
- **Automated Tests:** 10 unit and integration tests covering data models, linear regression prediction, priority scoring formulas, selection overrides, CVRP solver constraints, and baseline comparisons.
- **Status:** **10/10 passed** in 10.2s.
- **End-to-End Test:** [`scripts/test_pipeline.py`](file:///c:/Project/SWM/binsense/scripts/test_pipeline.py) runs Phases 1 through 7 end-to-end.

### Bug Fixes & Stability Updates (Post-Verification)
- **Priority Queue Overrides:** Fixed `StreamlitDefaultNotInOptionsError` in `st.multiselect` where default included/excluded bins were pruned from dynamic options on re-render. All bin IDs are now provided with safe default filtering.
- **Route Optimizer Stop Rendering:** Fixed `KeyError: 'stops'` by iterating directly over `r.get("stop_bin_ids", [])` and adding alias support to `format_route_results`.
- **Baseline Comparison Metrics:** Added missing keys `num_bins`, `binsense_load_liters`, and `bins_serviced_diff` to `src/baseline.py` to prevent key errors in the Before/After comparison view.
- **Map Basemap Tiles:** Switched default map tiles from CartoDB to OpenStreetMap to prevent third-party API key warnings and ensure uninterrupted offline/online rendering.
- **Collection Cycle Reset:** Fixed `KeyError: 'accumulation_rate_pct_per_hr'` when completing a collection cycle in Screen 5. State update now safely pulls accumulation rates from predictions with fallbacks, updates collection timestamps, and resets emptied bins.

### Phase 10: Documentation
- **[`README.md`](file:///c:/Project/SWM/binsense/README.md):** Complete project overview, architecture diagram, 5-screen flow, technology stack, and quickstart commands.
- **[`walkthrough.md`](file:///C:/Users/devay/.gemini/antigravity-ide/brain/0570fa15-6e93-479e-86b3-931866da51bc/walkthrough.md):** Comprehensive phase-by-phase implementation narrative.

---

## 3. Key Technical Decisions & Patterns

| Aspect | Implementation Choice | Rationale |
|---|---|---|
| **Tech Stack** | 100% Free & Open-Source (Python, OR-Tools, OSMnx, Streamlit, Folium) | Zero recurring API costs, reproducible anywhere, offline capable |
| **Prediction Model** | Linear Regression with reset detection | Explainable, fast, robust for hourly IoT fill curves without requiring GPU |
| **Routing Algorithm** | Road-network CVRP (OSMnx + Google OR-Tools) | Straight-line distance underestimates urban transit by 30-50% |
| **Capacity Buffer** | 85% of theoretical fleet capacity | Guarantees multi-vehicle bin packing feasibility under integer constraints |
| **UI Design System** | Dark ops dashboard (`#0E1912`, `#182620`) | Meets UI/UX design spec, high contrast for live projector/demo screens |
| **Feedback Loop** | Cycle completion updates sensor state | Closes the loop: emptied bins reset to 0-5%, advancing future predictions |

---

## 4. Key Commands Reference

```powershell
# Navigate to prototype
cd c:\Project\SWM\binsense

# Activate virtual environment
.\venv\Scripts\activate

# Run Streamlit Dashboard (Port 8501)
streamlit run app.py

# Run Automated Test Suite (10 tests)
python -m unittest tests/test_all.py

# Run E2E Pipeline Smoke Test
python scripts/test_pipeline.py
```
