# ♻️ BinSense — Predictive & Dynamic Smart Waste Management

> **Competition Prototype**  
> A road-network-based multi-vehicle route optimization and predictive waste collection operations platform.

---

## 🌟 Overview

Traditional municipal waste management is **reactive**: bins overflow, citizen complaints are filed, or trucks follow rigid, fixed schedules regardless of actual demand.

**BinSense** shifts waste collection from **reactive → predictive → dynamically optimized**:
1. **Predicts fill levels** over a configurable 6–24h horizon using time-series linear regression with collection-reset detection.
2. **Prioritizes bins dynamically** with a 5-factor weighted score (0–100) classifying bins into 4 urgency tiers: *Critical, High, Medium, Low*.
3. **Selects candidate bins** constrained by total fleet capacity and operator manual overrides.
4. **Calculates road-network shortest-path distance matrices** using real OpenStreetMap road graphs (OSMnx + NetworkX) rather than straight-line distance.
5. **Optimizes multi-vehicle routes** using Google OR-Tools Capacitated Vehicle Routing Problem (CVRP) solver.
6. **Quantifies efficiency gains** against a naive fixed-threshold baseline in a live Before/After comparison.
7. **Presents a high-contrast 5-screen operations center** built in Streamlit, Folium, and Plotly following the dark-mode Ops Center design spec.

---

## 🏗️ Architecture & Pipeline Flow

```
Simulated Bin Sensors (IoT)
           │
           ▼
[Phase 2: Fill Prediction]  ──► Linear regression + reset detection (t+T forecast)
           │
           ▼
[Phase 3: Priority Engine]  ──► Composite score (0-100) across 5 signals + 4 tiers
           │
           ▼
[Phase 4: Bin Selection]    ──► Urgency filtering + capacity pruning + manual overrides
           │
           ▼
[Phase 5: Road Network]     ──► Cached OSM graph Dijkstra shortest paths (meters/seconds)
           │
           ▼
[Phase 6: CVRP Solver]      ──► Google OR-Tools multi-vehicle routing with capacity bounds
           │
           ▼
[Phase 7: Baseline Compare] ──► Naive 80% threshold + Euclidean TSP comparator
           │
           ▼
[Phase 8: Operations UI]    ──► 5-Screen Streamlit Dashboard (Dashboard, Map, Queue, Optimizer, Dispatch)
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.11+ (64-bit recommended)
- Windows / Linux / macOS

### 1. Clone & Setup
```bash
git clone https://github.com/megatron-71/BinSense-Predictive-Dynamic-Smart-Waste-Management.git
cd BinSense-Predictive-Dynamic-Smart-Waste-Management
```

### 2. Create & Activate Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate

# Activate (Linux / macOS)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Full Pipeline Test
```bash
python scripts/test_pipeline.py
```
*Validates Phases 1 through 7 end-to-end with live CVRP solving.*

### 5. Run the Automated Unit Test Suite
```bash
python -m unittest tests/test_all.py
```
*Executes all 10 unit and integration tests across data simulation, regression forecasting, priority scoring, selection pruning, CVRP solver, and baseline comparison.*

### 6. Launch the BinSense Operations Dashboard
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🖥️ 5-Screen Operator Flow

| Screen | Title | Key Features |
|---|---|---|
| **Screen 1** | **Operations Dashboard** | Real-time urgency summary tiles (Critical, High, Medium, Low with pulse indicator), fleet capacity gauge, city-wide overflow risk assessment, and tier vs. fill analytics. |
| **Screen 2** | **Priority Map** | Interactive geospatial Folium map of Bengaluru CBD; color-coded tier markers, pulsing Critical bins, and popovers displaying live and forecasted fill %. |
| **Screen 3** | **Priority Queue** | Ranked candidate table sorted by priority score; live fleet demand counter; force include/exclude operator override controls; and primary **Build Routes** action. |
| **Screen 4** | **Route Optimizer** | Turn-by-turn road-network vehicle paths; stop sequences; per-vehicle load/capacity metrics; and the **Before / After Baseline Toggle** showcasing distance and proactive collection gains. |
| **Screen 5** | **Fleet Dispatch & Tracking** | Active vehicle transit states (En Route, Loading, Returning); waypoint progress; distance saved highlight banner; and **Complete Cycle & Reset** action closing the feedback loop. |

---

## 📊 Before / After Baseline Comparison

| Metric | Traditional Baseline | BinSense Predictive & Dynamic | Operational Impact |
|---|---|---|---|
| **Trigger Mechanism** | Reactive (only bins currently ≥80%) | Predictive (forecasted to overflow in horizon) | **Zero emergency overflows** |
| **Routing Algorithm** | Naive straight-line / Euclidean | Real OpenStreetMap road network shortest paths | **Realistic turn-by-turn dispatch** |
| **Capacity Allocation** | Single vehicle / unconstrained | Multi-vehicle balanced OR-Tools CVRP | **Even fleet workload & balanced wear** |
| **Bins Serviced / Run** | Typically 4–6 bins (reactive) | 16–22 bins (proactive) | **+200% to +300% collection productivity** |
| **Efficiency (L / km)** | Low (~30–45 L/km) | High (~180–220 L/km) | **Maximum waste cleared per kilometer driven** |

---

## 🛠️ Technology Stack (100% Free & Open-Source)

- **Language:** Python 3.11
- **Optimization:** Google OR-Tools (Capacitated Vehicle Routing Problem)
- **Geospatial & Road Networks:** OSMnx (OpenStreetMap) & NetworkX
- **Predictive Modeling:** scikit-learn & numpy / pandas
- **Frontend / Dashboard:** Streamlit, Folium, `streamlit-folium`, Plotly
- **Design Aesthetic:** Ops Center Dark Theme (`#0E1912`, `#182620`, `#6FAE7C`, `#E8A13D`, `#D9714E`)

---

## 📁 Repository Structure

```
BinSense-Predictive-Dynamic-Smart-Waste-Management/
├── data/
│   ├── bins.csv                 # Generated bin locations snapped to OSM nodes
│   ├── vehicles.csv             # Fleet vehicle specifications & capacities
│   ├── fill_history.csv         # 72-hour historical IoT sensor readings
│   └── road_graph.graphml       # Cached OpenStreetMap road network
├── src/
│   ├── __init__.py              # Package init
│   ├── config.py                # All tunable system parameters & design tokens
│   ├── simulate.py              # Synthetic IoT bin & fleet data generator
│   ├── predict.py               # Fill-level regression & overflow forecasting
│   ├── priority.py              # Dynamic priority engine & tier classification
│   ├── select.py                # Capacity-aware bin selection & overrides
│   ├── network.py               # OSMnx road graph & distance/time matrix
│   ├── routing.py               # Google OR-Tools CVRP route solver
│   └── baseline.py              # Naive fixed-schedule baseline comparator
├── tests/
│   └── test_all.py              # Automated 10-test unit & integration suite
├── scripts/
│   ├── cache_road_graph.py      # Standalone graph download & caching utility
│   ├── test_imports.py          # Environment & dependency verification
│   └── test_pipeline.py         # End-to-end backend pipeline smoke test
├── app.py                       # 5-screen Streamlit operations dashboard
├── requirements.txt             # Python dependency manifest
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
├── MEMORY.md                    # Project implementation record
└── README.md                    # System documentation
```

---

## 👥 Authors & License
Developed for the Smart Waste Management Competition.  
Licensed under the **MIT License**.
