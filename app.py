"""
BinSense — Predictive & Dynamic Smart Waste Management
=====================================================
5-Screen Operator Dashboard for Night-Operations Control Center.
Built with Streamlit, Folium, Plotly, OR-Tools, and OSMnx.
"""

import sys
import os
import time
import math
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import folium
from streamlit_folium import st_folium

# Ensure src modules are importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import (
    COLORS,
    TIER_COLORS,
    TIER_ORDER,
    DEPOT_LOCATION,
    NUM_VEHICLES,
    VEHICLE_CAPACITY_LITERS,
    FLEET_CAPACITY_BUFFER,
    PREDICTION_HORIZON_HOURS,
    SELECTION_MIN_TIER,
    BASELINE_FILL_THRESHOLD,
    CVRP_TIME_LIMIT_SECONDS,
    VEHICLE_COLORS,
    MAP_TILES,
)
from src.simulate import (
    generate_bins,
    generate_vehicles,
    generate_fill_history,
    get_current_state,
    save_datasets,
)
from src.network import (
    load_road_graph,
    build_distance_matrix,
    get_route_geometry,
    snap_to_graph,
)
from src.predict import predict_all_bins
from src.priority import score_all_bins, get_tier_summary
from src.select import select_bins
from src.routing import solve_routes, format_route_results
from src.baseline import baseline_routes, compare_results

# ─────────────────────────────────────────────────────────────
# 1. Page Configuration & Custom CSS Injection
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BinSense — Smart Waste Management",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #0E1912;
    --surface: #182620;
    --surface-2: #1E2F27;
    --line: rgba(241, 239, 230, 0.12);
    --text: #F1EFE6;
    --text-dim: #A9B3AC;
    --text-faint: #6E7A72;
    --leaf: #6FAE7C;
    --amber: #E8A13D;
    --danger: #D9714E;
    --medium: #E8D44D;
}

/* Global App Styling */
.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, .display-font {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    color: var(--text);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #121F17 !important;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--text);
}

/* Card components */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    border-color: rgba(241, 239, 230, 0.25);
}

.metric-title {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    margin-bottom: 6px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.1;
}

.metric-sub {
    font-size: 0.85rem;
    color: var(--text-faint);
    margin-top: 6px;
}

/* Urgency Badges & Pulse */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.badge-critical {
    background: rgba(217, 113, 78, 0.2);
    color: var(--danger);
    border: 1px solid var(--danger);
    box-shadow: 0 0 10px rgba(217, 113, 78, 0.35);
    animation: pulse-border 2s infinite ease-in-out;
}
.badge-high {
    background: rgba(232, 161, 61, 0.2);
    color: var(--amber);
    border: 1px solid var(--amber);
}
.badge-medium {
    background: rgba(232, 212, 77, 0.2);
    color: var(--medium);
    border: 1px solid var(--medium);
}
.badge-low {
    background: rgba(111, 174, 124, 0.2);
    color: var(--leaf);
    border: 1px solid var(--leaf);
}

@keyframes pulse-border {
    0% { box-shadow: 0 0 4px rgba(217, 113, 78, 0.2); }
    50% { box-shadow: 0 0 16px rgba(217, 113, 78, 0.7); }
    100% { box-shadow: 0 0 4px rgba(217, 113, 78, 0.2); }
}

/* Status Indicator Dot */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.dot-live {
    background-color: var(--leaf);
    box-shadow: 0 0 8px var(--leaf);
}
.dot-critical {
    background-color: var(--danger);
    box-shadow: 0 0 8px var(--danger);
}

/* Custom buttons */
.stButton > button {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #253A30;
    border-color: var(--leaf);
    color: #FFFFFF;
}

/* Primary Action CTA */
.primary-cta button {
    background: var(--leaf) !important;
    color: #0E1912 !important;
    font-weight: 700 !important;
    border: none !important;
}
.primary-cta button:hover {
    background: #84C592 !important;
    box-shadow: 0 0 12px rgba(111, 174, 124, 0.5) !important;
}

/* Table styling */
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: var(--leaf);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Landing Page CSS — injected separately for cleaner organization
LANDING_CSS = """
<style>
/* ─── Hide sidebar on landing page ─── */
.landing-active [data-testid="stSidebar"] { display: none !important; }
.landing-active [data-testid="stSidebarCollapsedControl"] { display: none !important; }
.landing-active .stMainBlockContainer { max-width: 100% !important; padding-top: 0 !important; }
.landing-active header[data-testid="stHeader"] { display: none !important; }

/* ─── Animated gradient background ─── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.landing-hero {
    position: relative;
    min-height: 100vh;
    background: linear-gradient(-45deg, #0A1210, #0E1912, #132A1C, #0B1F15, #091510);
    background-size: 400% 400%;
    animation: gradientShift 16s ease infinite;
    padding: 0;
    margin: -1rem -1rem 0 -1rem;
    overflow: hidden;
}

/* ─── Floating ambient orbs ─── */
@keyframes floatOrb1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    25%      { transform: translate(60px, -40px) scale(1.1); }
    50%      { transform: translate(-30px, -80px) scale(0.95); }
    75%      { transform: translate(40px, -20px) scale(1.05); }
}
@keyframes floatOrb2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33%      { transform: translate(-50px, 30px) scale(1.15); }
    66%      { transform: translate(70px, 60px) scale(0.9); }
}
.landing-orb {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
}
.orb-1 {
    width: 500px; height: 500px;
    top: -100px; right: -100px;
    background: radial-gradient(circle, rgba(111,174,124,0.18), transparent 70%);
    animation: floatOrb1 20s ease-in-out infinite;
}
.orb-2 {
    width: 400px; height: 400px;
    bottom: 50px; left: -80px;
    background: radial-gradient(circle, rgba(232,161,61,0.12), transparent 70%);
    animation: floatOrb2 24s ease-in-out infinite;
}
.orb-3 {
    width: 300px; height: 300px;
    top: 40%; left: 50%;
    background: radial-gradient(circle, rgba(91,168,217,0.10), transparent 70%);
    animation: floatOrb1 18s ease-in-out infinite reverse;
}

/* ─── Hero content ─── */
.landing-content {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 40px 20px;
    text-align: center;
}

/* ─── Logo mark ─── */
@keyframes logoPulse {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 20px rgba(111,174,124,0.3)); }
    50%      { transform: scale(1.05); filter: drop-shadow(0 0 40px rgba(111,174,124,0.5)); }
}
.landing-logo {
    font-size: 4.5rem;
    margin-bottom: 12px;
    animation: logoPulse 4s ease-in-out infinite;
}

/* ─── Title animation ─── */
@keyframes titleReveal {
    from { opacity: 0; transform: translateY(30px); }
    to   { opacity: 1; transform: translateY(0); }
}
.landing-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #F1EFE6;
    margin: 0;
    animation: titleReveal 1s ease-out forwards;
}
.landing-title .highlight {
    background: linear-gradient(135deg, #6FAE7C, #5BC99A, #34D399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ─── Subtitle ─── */
.landing-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    color: #A9B3AC;
    max-width: 620px;
    line-height: 1.65;
    margin: 16px auto 40px auto;
    animation: titleReveal 1s ease-out 0.3s both;
}

/* ─── Feature cards ─── */
@keyframes cardFadeIn {
    from { opacity: 0; transform: translateY(40px); }
    to   { opacity: 1; transform: translateY(0); }
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    max-width: 960px;
    width: 100%;
    margin: 0 auto 48px auto;
}
.feature-card {
    background: rgba(24, 38, 32, 0.65);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(241, 239, 230, 0.08);
    border-radius: 16px;
    padding: 28px 22px;
    text-align: left;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    animation: cardFadeIn 0.8s ease-out both;
}
.feature-card:nth-child(1) { animation-delay: 0.5s; }
.feature-card:nth-child(2) { animation-delay: 0.65s; }
.feature-card:nth-child(3) { animation-delay: 0.8s; }
.feature-card:nth-child(4) { animation-delay: 0.95s; }
.feature-card:hover {
    border-color: rgba(111, 174, 124, 0.35);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
}
.feature-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    margin-bottom: 16px;
}
.feature-card h4 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #F1EFE6;
    margin: 0 0 8px 0;
}
.feature-card p {
    font-size: 0.85rem;
    color: #A9B3AC;
    line-height: 1.5;
    margin: 0;
}

/* ─── Stats ticker ─── */
.stats-row {
    display: flex;
    gap: 48px;
    justify-content: center;
    margin: 0 auto 48px auto;
    animation: cardFadeIn 0.8s ease-out 1.1s both;
}
.stat-item {
    text-align: center;
}
.stat-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #F1EFE6;
    line-height: 1;
}
.stat-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6E7A72;
    margin-top: 6px;
}

/* ─── CTA button ─── */
@keyframes ctaGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(111,174,124,0.25), 0 0 60px rgba(111,174,124,0.08); }
    50%      { box-shadow: 0 0 30px rgba(111,174,124,0.45), 0 0 80px rgba(111,174,124,0.15); }
}
.cta-wrapper {
    animation: cardFadeIn 0.8s ease-out 1.3s both;
}
.landing-cta {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, #6FAE7C, #5BA870);
    color: #0E1912;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    padding: 16px 40px;
    border-radius: 14px;
    border: none;
    cursor: pointer;
    transition: all 0.3s ease;
    animation: ctaGlow 3s ease-in-out infinite;
    text-decoration: none;
}
.landing-cta:hover {
    background: linear-gradient(135deg, #84C592, #6FAE7C);
    transform: translateY(-2px) scale(1.02);
}
.landing-cta .arrow {
    transition: transform 0.3s ease;
}
.landing-cta:hover .arrow {
    transform: translateX(4px);
}

/* ─── Bottom info bar ─── */
.landing-footer {
    margin-top: 40px;
    font-size: 0.8rem;
    color: #6E7A72;
    animation: cardFadeIn 0.8s ease-out 1.5s both;
}
.landing-footer a {
    color: #6FAE7C;
    text-decoration: none;
}
.tech-badges {
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    margin-top: 12px;
}
.tech-badge {
    background: rgba(241,239,230,0.06);
    border: 1px solid rgba(241,239,230,0.08);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.75rem;
    color: #A9B3AC;
    font-family: 'Inter', sans-serif;
}

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .landing-title { font-size: 2.4rem; }
    .landing-subtitle { font-size: 1rem; }
    .feature-grid { grid-template-columns: repeat(2, 1fr); gap: 12px; }
    .stats-row { gap: 24px; flex-wrap: wrap; }
}
</style>
"""


# ─────────────────────────────────────────────────────────────
# 2. Session State Initialization
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading cached road network graph...")
def get_cached_graph():
    return load_road_graph()

def init_simulation():
    """Initializes or resets the end-to-end simulation dataset."""
    G = get_cached_graph()
    bins_df = generate_bins(G, seed=42)
    vehicles_df = generate_vehicles(num_vehicles=NUM_VEHICLES)
    history_df = generate_fill_history(bins_df, seed=42)
    state_df = get_current_state(bins_df, history_df)
    
    st.session_state.road_graph = G
    st.session_state.bins_df = bins_df
    st.session_state.vehicles_df = vehicles_df
    st.session_state.history_df = history_df
    st.session_state.current_state_df = state_df
    st.session_state.include_ids = set()
    st.session_state.exclude_ids = set()
    st.session_state.selected_bins_df = None
    st.session_state.matrix_result = None
    st.session_state.cvrp_solution = None
    st.session_state.baseline_result = None
    st.session_state.comparison = None
    st.session_state.cycle_count = 1
    st.session_state.dispatched = False

if "current_state_df" not in st.session_state:
    init_simulation()

if "current_screen" not in st.session_state:
    st.session_state.current_screen = "Landing"

G = st.session_state.road_graph
state_df = st.session_state.current_state_df
history_df = st.session_state.history_df


# ─────────────────────────────────────────────────────────────
# 3. Core Reactive Data Pipeline
# ─────────────────────────────────────────────────────────────

def run_pipeline(horizon=PREDICTION_HORIZON_HOURS, min_tier=SELECTION_MIN_TIER):
    """Executes prediction and priority scoring."""
    pred_df = predict_all_bins(
        st.session_state.current_state_df,
        st.session_state.history_df,
        horizon_hours=horizon,
    )
    scored_df = score_all_bins(pred_df)
    selection_res = select_bins(
        scored_df,
        min_tier=min_tier,
        include_ids=st.session_state.include_ids,
        exclude_ids=st.session_state.exclude_ids,
        capacity_buffer=FLEET_CAPACITY_BUFFER,
    )
    return pred_df, scored_df, selection_res

pred_df, scored_df, selection_res = run_pipeline(
    horizon=st.session_state.get("sel_horizon", PREDICTION_HORIZON_HOURS),
    min_tier=st.session_state.get("sel_min_tier", SELECTION_MIN_TIER),
)
selected_df = selection_res["selected"]


# ─────────────────────────────────────────────────────────────
# 4A. Landing Page (shown before entering the Operations Center)
# ─────────────────────────────────────────────────────────────

if st.session_state.current_screen == "Landing":
    # Hide sidebar on landing page
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    .stMainBlockContainer { max-width: 100% !important; padding-top: 0 !important; }
    </style>
    """, unsafe_allow_html=True)

    # Tier counts for live stats
    tier_counts = scored_df["tier"].value_counts().to_dict()
    crit_count = tier_counts.get("critical", 0)
    high_count = tier_counts.get("high", 0)
    total_bins = len(scored_df)
    urgent_bins = crit_count + high_count

    # Use st.html() for the full landing page (Streamlit 1.33+ renders raw HTML)
    st.html(LANDING_CSS + f"""
    <div class="landing-hero">
        <div class="landing-orb orb-1"></div>
        <div class="landing-orb orb-2"></div>
        <div class="landing-orb orb-3"></div>

        <div class="landing-content">
            <div class="landing-logo">\u267b\ufe0f</div>

            <h1 class="landing-title">
                <span class="highlight">BinSense</span>
            </h1>
            <p class="landing-subtitle">
                Predictive &amp; Dynamic Smart Waste Management System.<br>
                Forecast fill levels, compute urgency tiers, and optimize
                multi-vehicle collection routes on real road networks
                &mdash; powered by AI and Google OR-Tools.
            </p>

            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon" style="background: rgba(111,174,124,0.15);">
                        <span style="color: #6FAE7C;">\U0001F4C8</span>
                    </div>
                    <h4>Predictive Analytics</h4>
                    <p>Linear regression on sensor data forecasts fill levels
                    12 hours ahead with collection-reset detection</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon" style="background: rgba(232,161,61,0.15);">
                        <span style="color: #E8A13D;">\u26a1</span>
                    </div>
                    <h4>Dynamic Priority</h4>
                    <p>Weighted 5-signal scoring engine classifies bins into
                    Critical, High, Medium, and Low urgency tiers</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon" style="background: rgba(91,168,217,0.15);">
                        <span style="color: #5BA8D9;">\U0001F5FA\ufe0f</span>
                    </div>
                    <h4>CVRP Route Optimizer</h4>
                    <p>Real road-network multi-vehicle routing via OSMnx +
                    Google OR-Tools with capacity constraints</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon" style="background: rgba(217,113,78,0.15);">
                        <span style="color: #D9714E;">\U0001F4CA</span>
                    </div>
                    <h4>Ops Dashboard</h4>
                    <p>5-screen dark operations center with real-time maps,
                    queue management, and fleet dispatch tracking</p>
                </div>
            </div>

            <div class="stats-row">
                <div class="stat-item">
                    <div class="stat-value">{total_bins}</div>
                    <div class="stat-label">Smart Bins</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{NUM_VEHICLES}</div>
                    <div class="stat-label">Vehicles</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{len(G.nodes)}</div>
                    <div class="stat-label">Road Nodes</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #D9714E;">{urgent_bins}</div>
                    <div class="stat-label">Urgent Bins</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" style="color: #6FAE7C;">{NUM_VEHICLES * VEHICLE_CAPACITY_LITERS}L</div>
                    <div class="stat-label">Fleet Capacity</div>
                </div>
            </div>

            <div class="landing-footer">
                <div>100% Open Source &middot; Zero API Costs &middot; Offline Capable</div>
                <div class="tech-badges">
                    <span class="tech-badge">Python 3.11</span>
                    <span class="tech-badge">Google OR-Tools</span>
                    <span class="tech-badge">OSMnx</span>
                    <span class="tech-badge">Streamlit</span>
                    <span class="tech-badge">Folium</span>
                    <span class="tech-badge">Plotly</span>
                    <span class="tech-badge">scikit-learn</span>
                </div>
            </div>
        </div>
    </div>
    """)

    # CTA Button (must be a real Streamlit button for interactivity)
    cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
    with cta_col2:
        st.markdown("<div class='primary-cta' style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
        if st.button("\U0001F680  Enter Operations Center  \u2192", key="landing_cta", use_container_width=True):
            st.session_state.current_screen = "Dashboard"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 4B. Sidebar Controls & Navigation (hidden on Landing)
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.8rem;">♻️</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.4rem; letter-spacing: -0.02em;">BinSense</h2>
                    <span style="font-size: 0.75rem; color: #A9B3AC; text-transform: uppercase; letter-spacing: 0.08em;">Smart Waste OS</span>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <span class="status-dot dot-live"></span>
                <span style="font-size: 0.8rem; color: #6FAE7C; font-weight: 600;">OPERATIONS LIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    screens = [
        "Dashboard",
        "Priority Map",
        "Priority Queue",
        "Route Optimizer",
        "Fleet Dispatch",
    ]
    icons = ["📊", "🗺️", "📋", "⚡", "🚛"]

    for screen, icon in zip(screens, icons):
        is_active = st.session_state.current_screen == screen
        btn_label = f"{icon}  {screen}"
        if is_active:
            st.markdown(
                f"<div style='background: #1E2F27; border-left: 3px solid #6FAE7C; padding: 6px 12px; border-radius: 4px; font-weight: 600; margin-bottom: 4px;'>{btn_label}</div>",
                unsafe_allow_html=True,
            )
        else:
            if st.button(btn_label, key=f"nav_{screen}", use_container_width=True):
                st.session_state.current_screen = screen
                st.rerun()

    st.markdown("---")
    st.markdown("### Operational Controls")
    
    with st.expander("⚙️ Pipeline Configuration", expanded=False):
        sel_horizon = st.slider(
            "Prediction Horizon (Hours)",
            min_value=6,
            max_value=24,
            value=st.session_state.get("sel_horizon", PREDICTION_HORIZON_HOURS),
            step=2,
            key="sel_horizon",
            help="Forecast horizon for bin fill rate regression.",
        )
        sel_min_tier = st.selectbox(
            "Auto-Select Urgency Tier",
            options=["critical", "high", "medium", "low"],
            index=["critical", "high", "medium", "low"].index(
                st.session_state.get("sel_min_tier", SELECTION_MIN_TIER)
            ),
            key="sel_min_tier",
            help="Minimum urgency tier to include in automatic route building.",
        )
        fleet_buffer = st.slider(
            "Fleet Capacity Buffer",
            min_value=70,
            max_value=100,
            value=int(FLEET_CAPACITY_BUFFER * 100),
            format="%d%%",
            help="Target percentage of theoretical fleet capacity to budget for routes.",
        )
        
    st.markdown("---")
    st.markdown("### Quick Info")
    st.caption(f"**City Area:** Bengaluru Central (Cubbon / CBD)")
    st.caption(f"**Network Graph:** {len(G.nodes)} nodes, {len(G.edges)} road links")
    st.caption(f"**Fleet Size:** {NUM_VEHICLES} Vehicles ({NUM_VEHICLES * VEHICLE_CAPACITY_LITERS}L total)")
    st.caption(f"**Cycle Index:** Cycle #{st.session_state.cycle_count}")

    if st.button("🔄 Reset Simulation Dataset", use_container_width=True):
        init_simulation()
        st.success("Simulation re-initialized with seeded state!")
        st.rerun()


# ─────────────────────────────────────────────────────────────
# 5. Screen 1 — Dashboard
# ─────────────────────────────────────────────────────────────

if st.session_state.current_screen == "Dashboard":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 24px;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem;">Operations Dashboard</h1>
                <p style="color: #A9B3AC; margin: 4px 0 0 0;">Real-time urban waste monitoring & predictive urgency tracking</p>
            </div>
            <div>
                <span class="badge badge-critical" style="font-size: 0.85rem; padding: 6px 14px;">
                    ● Night Ops Active
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tier_counts = scored_df["tier"].value_counts().to_dict()
    crit_count = tier_counts.get("critical", 0)
    high_count = tier_counts.get("high", 0)
    med_count = tier_counts.get("medium", 0)
    low_count = tier_counts.get("low", 0)

    # 4 Urgency Count Tiles
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid var(--danger);">
                <div class="metric-title">Critical Urgency</div>
                <div class="metric-value" style="color: var(--danger);">{crit_count}</div>
                <div class="metric-sub">≥85 score · Immediate dispatch</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid var(--amber);">
                <div class="metric-title">High Urgency</div>
                <div class="metric-value" style="color: var(--amber);">{high_count}</div>
                <div class="metric-sub">65–84 score · Imminent overflow</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid var(--medium);">
                <div class="metric-title">Medium Urgency</div>
                <div class="metric-value" style="color: var(--medium);">{med_count}</div>
                <div class="metric-sub">40–64 score · Approaching threshold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 4px solid var(--leaf);">
                <div class="metric-title">Low / Normal</div>
                <div class="metric-value" style="color: var(--leaf);">{low_count}</div>
                <div class="metric-sub">&lt;40 score · Sufficient capacity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Operational Fleet & Capacity Row
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(
            """
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div class="metric-title" style="margin:0;">Fleet Capacity & Current Demand</div>
                    <span style="font-size: 0.85rem; color: #6FAE7C; font-weight: 600;">3 of 3 Vehicles Ready</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        
        tot_fleet_cap = NUM_VEHICLES * VEHICLE_CAPACITY_LITERS
        tot_demand = selection_res["total_demand_liters"]
        cap_util = min(100.0, (tot_demand / tot_fleet_cap) * 100.0)

        bar_color = COLORS["leaf"] if cap_util < 75 else (COLORS["amber"] if cap_util < 90 else COLORS["danger"])
        
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 1.5rem; font-weight: 700; font-family: 'Space Grotesk';">{tot_demand:.0f} L</span>
                <span style="font-size: 0.9rem; color: #A9B3AC;">of {tot_fleet_cap} L Fleet Capacity ({cap_util:.1f}%)</span>
            </div>
            <div style="background: rgba(241,239,230,0.08); border-radius: 8px; height: 12px; margin: 10px 0; overflow: hidden;">
                <div style="background: {bar_color}; width: {cap_util}%; height: 100%; border-radius: 8px; transition: width 0.5s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #6E7A72;">
                <span>0 L</span>
                <span>Selected Candidates: {len(selected_df)} Bins</span>
                <span>{tot_fleet_cap} L Max</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Overview Analytics Plotly
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Urgency Tiers vs. Average Fill Levels</div>", unsafe_allow_html=True)
        
        tier_stats = scored_df.groupby("tier").agg(
            avg_current=("current_fill_pct", "mean"),
            avg_pred=("predicted_fill_pct", "mean"),
            count=("bin_id", "count")
        ).reindex(TIER_ORDER).dropna()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tier_stats.index.str.capitalize(),
            y=tier_stats["avg_current"],
            name="Current Fill %",
            marker_color="#6E7A72",
        ))
        fig.add_trace(go.Bar(
            x=tier_stats.index.str.capitalize(),
            y=tier_stats["avg_pred"],
            name=f"Predicted Fill % (+{PREDICTION_HORIZON_HOURS}h)",
            marker_color=[TIER_COLORS[t] for t in tier_stats.index],
        ))
        fig.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            font=dict(color="#F1EFE6", family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(gridcolor="rgba(241,239,230,0.08)", title="Fill Percentage (%)"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        # Overflow Risk Card
        high_risk_bins = len(scored_df[scored_df["time_to_overflow_hours"] <= 12])
        risk_level = "HIGH" if high_risk_bins >= 5 else ("MODERATE" if high_risk_bins >= 2 else "LOW")
        risk_color = COLORS["danger"] if risk_level == "HIGH" else (COLORS["amber"] if risk_level == "MODERATE" else COLORS["leaf"])

        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid {risk_color};">
                <div class="metric-title">City-Wide Overflow Risk</div>
                <div class="metric-value" style="color: {risk_color}; font-size: 1.8rem;">{risk_level}</div>
                <div class="metric-sub" style="margin-top: 8px;">
                    <strong>{high_risk_bins} bins</strong> are projected to reach 100% capacity within 12 hours if unserviced.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Actionable Quick Links</div>
                <p style="font-size: 0.85rem; color: #A9B3AC;">Proceed through the operational pipeline:</p>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗺️ Open Priority Map", use_container_width=True):
            st.session_state.current_screen = "Priority Map"
            st.rerun()
        if st.button("📋 Review Priority Queue", use_container_width=True):
            st.session_state.current_screen = "Priority Queue"
            st.rerun()
        if st.button("⚡ Generate CVRP Routes", use_container_width=True):
            st.session_state.current_screen = "Route Optimizer"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# 6. Screen 2 — Priority Map
# ─────────────────────────────────────────────────────────────

elif st.session_state.current_screen == "Priority Map":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem;">Priority Map</h1>
                <p style="color: #A9B3AC; margin: 4px 0 0 0;">Geospatial urgency distribution across Bengaluru road network</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Filter toggles
    fcol1, fcol2, fcol3 = st.columns([2, 2, 1])
    with fcol1:
        selected_tiers = st.multiselect(
            "Filter Displayed Tiers",
            options=TIER_ORDER,
            default=TIER_ORDER,
            format_func=lambda x: f"{x.capitalize()} ({len(scored_df[scored_df['tier'] == x])})"
        )
    with fcol2:
        zone_filter = st.multiselect(
            "Filter Zones",
            options=sorted(scored_df["zone"].unique()),
            default=sorted(scored_df["zone"].unique()),
        )
    with fcol3:
        st.write("")
        st.write("")
        if st.button("Proceed to Queue →", use_container_width=True):
            st.session_state.current_screen = "Priority Queue"
            st.rerun()

    filtered_map_bins = scored_df[
        (scored_df["tier"].isin(selected_tiers)) &
        (scored_df["zone"].isin(zone_filter))
    ]

    # Folium map construction
    depot_lat, depot_lon = DEPOT_LOCATION["latitude"], DEPOT_LOCATION["longitude"]
    m = folium.Map(
        location=[depot_lat, depot_lon],
        zoom_start=14,
        tiles=MAP_TILES,
        control_scale=True,
    )

    # Depot marker
    folium.Marker(
        location=[depot_lat, depot_lon],
        tooltip="Central Depot / Dispatch Hub",
        icon=folium.Icon(color="white", icon="home", prefix="fa"),
    ).add_to(m)

    # Bin markers
    for _, row in filtered_map_bins.iterrows():
        tier = row["tier"]
        color = TIER_COLORS[tier]
        is_crit = tier == "critical"
        radius = 9 if is_crit else (7 if tier == "high" else 5)
        
        popup_html = f"""
        <div style="font-family: 'Inter', sans-serif; color: #182620; width: 220px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <strong style="font-size: 1.1rem;">{row['bin_id']}</strong>
                <span style="background: {color}; color: #fff; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; font-weight: 700; text-transform: uppercase;">{tier}</span>
            </div>
            <div style="font-size: 0.85rem; line-height: 1.4;">
                <div>Zone: <strong>{row['zone'].capitalize()}</strong></div>
                <div>Current Fill: <strong>{row['current_fill_pct']:.1f}%</strong></div>
                <div>Predicted (+{PREDICTION_HORIZON_HOURS}h): <strong>{row['predicted_fill_pct']:.1f}%</strong></div>
                <div>Time to Overflow: <strong>{row['time_to_overflow_hours']:.1f}h</strong></div>
                <div>Priority Score: <strong>{row['priority_score']:.1f} / 100</strong></div>
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=radius,
            color="#FFFFFF" if is_crit else color,
            weight=2 if is_crit else 1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['bin_id']} ({tier.upper()}) - {row['current_fill_pct']:.0f}% fill",
        ).add_to(m)

    st_folium(m, width="100%", height=550)

    # Map Legend
    st.markdown(
        """
        <div style="display: flex; gap: 20px; background: #182620; padding: 12px 20px; border-radius: 8px; border: 1px solid var(--line); margin-top: 10px; font-size: 0.85rem;">
            <div><span style="color: #D9714E; font-size: 1.2rem;">●</span> <strong>Critical (≥85)</strong> — Needs immediate collection</div>
            <div><span style="color: #E8A13D; font-size: 1.2rem;">●</span> <strong>High (65–84)</strong> — Predicted overflow soon</div>
            <div><span style="color: #E8D44D; font-size: 1.2rem;">●</span> <strong>Medium (40–64)</strong> — Routine cycle candidate</div>
            <div><span style="color: #6FAE7C; font-size: 1.2rem;">●</span> <strong>Low (&lt;40)</strong> — Can safely defer</div>
            <div style="margin-left: auto;"><strong>🏛️ White Icon</strong> = Central Depot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# 7. Screen 3 — Priority Queue
# ─────────────────────────────────────────────────────────────

elif st.session_state.current_screen == "Priority Queue":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem;">Priority Queue</h1>
                <p style="color: #A9B3AC; margin: 4px 0 0 0;">Ranked collection candidates with operator manual override capabilities</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tot_fleet_cap = NUM_VEHICLES * VEHICLE_CAPACITY_LITERS
    eff_cap = tot_fleet_cap * FLEET_CAPACITY_BUFFER
    curr_demand = selection_res["total_demand_liters"]
    cap_pct = (curr_demand / tot_fleet_cap) * 100.0

    # Live summary banner
    st.markdown(
        f"""
        <div style="background: #182620; border: 1px solid var(--line); border-left: 4px solid var(--leaf); padding: 14px 20px; border-radius: 8px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-family: 'Space Grotesk'; font-size: 1.25rem; font-weight: 700; color: #F1EFE6;">{len(selected_df)} Bins Selected</span>
                <span style="color: #A9B3AC; margin-left: 10px;">· Estimated Demand: <strong>{curr_demand:.0f} L</strong> of {tot_fleet_cap} L Fleet Capacity ({cap_pct:.1f}%)</span>
            </div>
            <div>
                <span style="color: {'#D9714E' if selection_res['capacity_exceeded'] else '#6FAE7C'}; font-weight: 600; font-size: 0.9rem;">
                    {'⚠️ Capacity Buffer Adjusted' if selection_res['capacity_exceeded'] else '✅ Within Safe Fleet Budget'}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Queue table with controls
    col_q1, col_q2 = st.columns([3, 1])

    with col_q2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-title'>Manual Overrides</div>", unsafe_allow_html=True)
        
        all_bin_ids = sorted(scored_df["bin_id"].unique())

        force_inc = st.multiselect(
            "Force Include Bins",
            options=all_bin_ids,
            default=[b for b in st.session_state.include_ids if b in all_bin_ids],
            help="Force selected bins into the run regardless of tier."
        )
        force_exc = st.multiselect(
            "Force Exclude Bins",
            options=all_bin_ids,
            default=[b for b in st.session_state.exclude_ids if b in all_bin_ids],
            help="Remove selected bins from this cycle's run."
        )

        if set(force_inc) != st.session_state.include_ids or set(force_exc) != st.session_state.exclude_ids:
            st.session_state.include_ids = set(force_inc)
            st.session_state.exclude_ids = set(force_exc)
            st.rerun()

        st.markdown("---")
        st.markdown("<div class='primary-cta'>", unsafe_allow_html=True)
        if st.button("⚡ Build Routes", use_container_width=True, disabled=len(selected_df) == 0):
            with st.spinner("Computing road network distance matrix & running OR-Tools CVRP..."):
                # Compute distance matrix and CVRP
                mat_res = build_distance_matrix(G, selected_bins_df=selected_df)
                bin_id_to_demand = {
                    row["bin_id"]: int(row["current_fill_pct"] / 100 * row["capacity_liters"])
                    for _, row in selected_df.iterrows()
                }
                demands = [0] + [bin_id_to_demand[bid] for bid in mat_res["bin_ids"]]
                
                sol = solve_routes(
                    mat_res["distance_matrix"],
                    demands,
                    num_vehicles=NUM_VEHICLES,
                    vehicle_capacities=[VEHICLE_CAPACITY_LITERS] * NUM_VEHICLES,
                    time_limit_seconds=CVRP_TIME_LIMIT_SECONDS,
                )
                formatted_routes = format_route_results(sol, mat_res["bin_ids"])
                
                # Baseline comparison
                base_res = baseline_routes(
                    scored_df,
                    DEPOT_LOCATION["latitude"],
                    DEPOT_LOCATION["longitude"],
                )
                binsense_summary = {
                    "total_distance": sol["total_distance"],
                    "total_load": sol["total_load"],
                    "routes": formatted_routes,
                    "num_bins_selected": len(selected_df),
                }
                comparison = compare_results(binsense_summary, base_res)

                # Store in session state
                st.session_state.selected_bins_df = selected_df
                st.session_state.matrix_result = mat_res
                st.session_state.cvrp_solution = sol
                st.session_state.formatted_routes = formatted_routes
                st.session_state.baseline_result = base_res
                st.session_state.comparison = comparison
                st.session_state.current_screen = "Route Optimizer"
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_q1:
        # Display Table
        display_cols = [
            "bin_id", "tier", "priority_score", "current_fill_pct",
            "predicted_fill_pct", "time_to_overflow_hours", "capacity_liters", "zone"
        ]
        q_df = scored_df.copy()
        q_df["in_run"] = q_df["bin_id"].isin(selected_df["bin_id"])
        q_df = q_df.sort_values("priority_score", ascending=False)
        
        # Formatting for pretty display
        styled_df = q_df.copy()
        styled_df["tier"] = styled_df["tier"].str.upper()
        styled_df["priority_score"] = styled_df["priority_score"].round(1)
        styled_df["current_fill_pct"] = styled_df["current_fill_pct"].apply(lambda x: f"{x:.1f}%")
        styled_df["predicted_fill_pct"] = styled_df["predicted_fill_pct"].apply(lambda x: f"{x:.1f}%")
        styled_df["time_to_overflow_hours"] = styled_df["time_to_overflow_hours"].apply(
            lambda x: f"{x:.1f}h" if math.isfinite(x) else "No overflow"
        )
        styled_df["status"] = styled_df["in_run"].apply(lambda x: "✅ Selected" if x else "⏸️ Deferred")

        st.dataframe(
            styled_df[["status", "bin_id", "tier", "priority_score", "current_fill_pct", "predicted_fill_pct", "time_to_overflow_hours", "zone"]],
            use_container_width=True,
            height=500,
            column_config={
                "status": st.column_config.TextColumn("Status", width="small"),
                "bin_id": st.column_config.TextColumn("Bin ID", width="small"),
                "tier": st.column_config.TextColumn("Tier", width="small"),
                "priority_score": st.column_config.NumberColumn("Score (0-100)", format="%.1f"),
                "current_fill_pct": st.column_config.TextColumn("Current %"),
                "predicted_fill_pct": st.column_config.TextColumn(f"Pred % (+{PREDICTION_HORIZON_HOURS}h)"),
                "time_to_overflow_hours": st.column_config.TextColumn("Overflow In"),
                "zone": st.column_config.TextColumn("Zone"),
            },
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────
# 8. Screen 4 — Route Optimizer
# ─────────────────────────────────────────────────────────────

elif st.session_state.current_screen == "Route Optimizer":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem;">Route Optimizer</h1>
                <p style="color: #A9B3AC; margin: 4px 0 0 0;">Road-network multi-vehicle route optimization via Google OR-Tools CVRP</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.cvrp_solution is None:
        st.info("No active route plan. Please select bins and click 'Build Routes' from the Priority Queue.")
        if st.button("← Go to Priority Queue"):
            st.session_state.current_screen = "Priority Queue"
            st.rerun()
    else:
        routes = st.session_state.formatted_routes
        matrix_result = st.session_state.matrix_result
        comparison = st.session_state.comparison
        baseline = st.session_state.baseline_result

        # Top comparison banner with Before/After toggle
        top_c1, top_c2 = st.columns([3, 1])
        with top_c1:
            view_mode = st.radio(
                "Route Visualization Mode",
                options=["BinSense Optimized (Road Network + Dynamic Prediction)", "Naive Baseline (Straight-Line + Fixed 80% Threshold)"],
                horizontal=True,
            )
        with top_c2:
            st.markdown("<div class='primary-cta'>", unsafe_allow_html=True)
            if st.button("🚀 Dispatch Fleet", use_container_width=True):
                st.session_state.current_screen = "Fleet Dispatch"
                st.session_state.dispatched = True
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        is_binsense = "BinSense" in view_mode

        # Vehicle colors
        VEH_COLORS = VEHICLE_COLORS

        if is_binsense:
            # Per-vehicle metric cards
            v_cols = st.columns(len(routes))
            for i, r in enumerate(routes):
                with v_cols[i]:
                    color = VEH_COLORS[i % len(VEH_COLORS)]
                    st.markdown(
                        f"""
                        <div class="metric-card" style="border-top: 4px solid {color}; padding: 14px 16px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="font-family: 'Space Grotesk'; font-size: 1.1rem; color: {color};">{r['vehicle_label']}</strong>
                                <span style="font-size: 0.8rem; color: #A9B3AC;">{r['num_stops']} Stops</span>
                            </div>
                            <div style="margin-top: 8px;">
                                <div style="font-size: 1.6rem; font-weight: 700; font-family: 'Space Grotesk';">{r['distance_km']} km</div>
                                <div style="font-size: 0.82rem; color: #6E7A72;">Load: {r['load_liters']}L / {r['capacity_liters']}L ({r['utilization_pct']}%)</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Folium Map with road network paths
            depot_lat, depot_lon = DEPOT_LOCATION["latitude"], DEPOT_LOCATION["longitude"]
            rm = folium.Map(
                location=[depot_lat, depot_lon],
                zoom_start=14,
                tiles=MAP_TILES,
                control_scale=True,
            )

            # Depot
            folium.Marker(
                location=[depot_lat, depot_lon],
                tooltip="Depot Hub",
                icon=folium.Icon(color="white", icon="home", prefix="fa"),
            ).add_to(rm)

            # Plot routes on road network
            for i, r in enumerate(routes):
                color = VEH_COLORS[i % len(VEH_COLORS)]
                full_indices = r.get("full_path_indices", [])
                
                # Extract road-network coordinates
                if full_indices and matrix_result and "node_ids" in matrix_result:
                    node_seq = [matrix_result["node_ids"][idx] for idx in full_indices]
                    road_coords = get_route_geometry(G, node_seq)
                else:
                    road_coords = []

                if road_coords:
                    folium.PolyLine(
                        locations=road_coords,
                        color=color,
                        weight=5,
                        opacity=0.85,
                        dash_array="6, 8",
                        tooltip=f"{r.get('vehicle_label', 'Vehicle')}: {r.get('distance_km', 0)} km",
                    ).add_to(rm)

                # Add stop markers
                bin_stops = r.get("stop_bin_ids", r.get("stops_bin_ids", []))
                for stop_order, bin_id in enumerate(bin_stops, start=1):
                    bin_matches = scored_df[scored_df["bin_id"] == bin_id]
                    if not bin_matches.empty:
                        bin_row = bin_matches.iloc[0]
                        folium.CircleMarker(
                            location=[bin_row["latitude"], bin_row["longitude"]],
                            radius=7,
                            color="#FFFFFF",
                            weight=1.5,
                            fill=True,
                            fill_color=color,
                            fill_opacity=1.0,
                            tooltip=f"Stop {stop_order} ({r.get('vehicle_label', 'Vehicle')}): {bin_id} ({bin_row['current_fill_pct']:.0f}%)",
                        ).add_to(rm)

            st_folium(rm, width="100%", height=500)

        else:
            # Baseline View
            base_bin_count = baseline.get("num_bins_selected", baseline.get("num_bins", 0))
            base_dist = baseline.get("total_distance_km", 0)
            base_load = baseline.get("total_load", 0)

            b_c1, b_c2 = st.columns(2)
            with b_c1:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 4px solid var(--danger);">
                        <div class="metric-title">Baseline Selection (Naive 80% Threshold)</div>
                        <div class="metric-value">{base_bin_count} Bins</div>
                        <div class="metric-sub">Only services bins currently ≥80% fill. Ignores rapidly accumulating bins!</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with b_c2:
                st.markdown(
                    f"""
                    <div class="metric-card" style="border-left: 4px solid var(--amber);">
                        <div class="metric-title">Baseline Distance (Straight-Line TSP)</div>
                        <div class="metric-value">{base_dist} km</div>
                        <div class="metric-sub">Theoretical Euclidean distance. Real roads would add 30–40% extra detour.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Baseline Map (Straight line routes)
            depot_lat, depot_lon = DEPOT_LOCATION["latitude"], DEPOT_LOCATION["longitude"]
            bm = folium.Map(
                location=[depot_lat, depot_lon],
                zoom_start=14,
                tiles=MAP_TILES,
            )
            folium.Marker(
                location=[depot_lat, depot_lon],
                tooltip="Depot Hub",
                icon=folium.Icon(color="white", icon="home", prefix="fa"),
            ).add_to(bm)

            # Draw straight lines for baseline
            for r in baseline.get("routes", []):
                line_coords = [(depot_lat, depot_lon)]
                for bid in r.get("stop_bin_ids", []):
                    b_match = scored_df[scored_df["bin_id"] == bid]
                    if not b_match.empty:
                        row = b_match.iloc[0]
                        line_coords.append((row["latitude"], row["longitude"]))
                        folium.CircleMarker(
                            location=[row["latitude"], row["longitude"]],
                            radius=6,
                            color="#D9714E",
                            fill=True,
                            fill_color="#D9714E",
                            fill_opacity=0.8,
                            tooltip=f"Baseline Stop: {bid}",
                        ).add_to(bm)
                line_coords.append((depot_lat, depot_lon))

                folium.PolyLine(
                    locations=line_coords,
                    color="#D9714E",
                    weight=3,
                    opacity=0.7,
                    dash_array="4, 6",
                    tooltip=f"{r.get('vehicle_id', 'Vehicle')} (Straight line)",
                ).add_to(bm)

            st_folium(bm, width="100%", height=500)

        # Before / After Comparison Analytics
        st.markdown("### Operational Impact Comparison")
        bs_bins = comparison.get("binsense_bins", len(st.session_state.selected_bins_df) if st.session_state.selected_bins_df is not None else 0)
        bl_bins = comparison.get("baseline_bins", baseline.get("num_bins_selected", 0))
        diff_bins = comparison.get("bins_serviced_diff", bs_bins - bl_bins)
        bs_load = comparison.get("binsense_load_liters", comparison.get("binsense_total_load", 0))
        bl_load = baseline.get("total_load", 0)
        bs_dist = comparison.get("binsense_distance_km", 1.0)
        eff = bs_load / max(1.0, bs_dist)

        comp_c1, comp_c2, comp_c3 = st.columns(3)
        with comp_c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Bins Serviced Per Run</div>
                    <div class="metric-value" style="color: var(--leaf);">{bs_bins} vs {bl_bins}</div>
                    <div class="metric-sub">+{diff_bins} additional high-risk bins proactively cleared</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with comp_c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Collection Productivity</div>
                    <div class="metric-value" style="color: var(--leaf);">{bs_load} L</div>
                    <div class="metric-sub">Total waste collected vs {bl_load} L baseline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with comp_c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Route Efficiency (L / km)</div>
                    <div class="metric-value" style="color: var(--leaf);">{eff:.0f} L/km</div>
                    <div class="metric-sub">Liters collected per kilometer driven</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────
# 9. Screen 5 — Fleet Dispatch & Tracking
# ─────────────────────────────────────────────────────────────

elif st.session_state.current_screen == "Fleet Dispatch":
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem;">Fleet Dispatch & Tracking</h1>
                <p style="color: #A9B3AC; margin: 4px 0 0 0;">Live tracking of dispatched collection units and cycle completion</p>
            </div>
            <div>
                <span class="badge badge-low" style="font-size: 0.85rem; padding: 6px 14px;">
                    ● Fleet Dispatched
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.cvrp_solution is None:
        st.info("No active routes have been dispatched yet. Build routes first.")
        if st.button("← Go to Route Optimizer"):
            st.session_state.current_screen = "Route Optimizer"
            st.rerun()
    else:
        routes = st.session_state.formatted_routes
        comp = st.session_state.comparison

        # Leaf Green Callout Banner
        st.markdown(
            f"""
            <div style="background: rgba(111,174,124,0.15); border: 1px solid var(--leaf); border-radius: 12px; padding: 20px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--leaf); font-weight: 700;">Optimization Performance</div>
                    <div style="font-family: 'Space Grotesk'; font-size: 1.8rem; font-weight: 700; color: #F1EFE6; margin-top: 4px;">
                        {comp['binsense_bins']} bins cleared across {len(routes)} optimized routes ({comp['binsense_distance_km']:.2f} km total)
                    </div>
                    <div style="font-size: 0.9rem; color: #A9B3AC; margin-top: 4px;">
                        Predictive dynamic routing prevented estimated <strong>{len(scored_df[scored_df['tier']=='critical'])} emergency overflows</strong>.
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family: 'Space Grotesk'; font-size: 2rem; font-weight: 700; color: var(--leaf);">{comp['binsense_load_liters']} L</div>
                    <div style="font-size: 0.8rem; color: #A9B3AC;">Total Payload Diverted</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Vehicle Tracking Cards
        st.markdown("### Active Collection Fleet")
        for i, r in enumerate(routes):
            color = VEHICLE_COLORS[i % len(VEHICLE_COLORS)]
            bin_stops = r.get("stops_bin_ids", r.get("stop_bin_ids", []))
            stops_str = " → ".join(bin_stops[:5])
            if len(bin_stops) > 5:
                stops_str += f" → +{len(bin_stops) - 5} more"

            st.markdown(
                f"""
                <div class="metric-card" style="border-left: 4px solid {color}; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-family: 'Space Grotesk'; font-size: 1.2rem; font-weight: 700; color: {color};">{r['vehicle_label']}</span>
                            <span style="background: rgba(111,174,124,0.2); color: #6FAE7C; font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; font-weight: 600; margin-left: 10px;">EN ROUTE</span>
                        </div>
                        <span style="font-size: 0.9rem; color: #A9B3AC;">Capacity: <strong>{r['load_liters']}L / {r['capacity_liters']}L</strong> ({r['utilization_pct']}%)</span>
                    </div>
                    <div style="margin: 10px 0; font-size: 0.88rem; color: #F1EFE6;">
                        <strong>Route Waypoints:</strong> Depot → {stops_str} → Depot
                    </div>
                    <div style="background: rgba(241,239,230,0.08); border-radius: 6px; height: 8px; overflow: hidden; margin-top: 8px;">
                        <div style="background: {color}; width: 65%; height: 100%; border-radius: 6px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6E7A72; margin-top: 4px;">
                        <span>Departed 22:00</span>
                        <span>In Progress: Stop {min(r['num_stops'], 3)} of {r['num_stops']}</span>
                        <span>Est. Return: 23:30</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Complete Collection Action
        st.markdown(
            """
            <div class="metric-card" style="text-align: center; padding: 24px;">
                <h3 style="margin: 0 0 8px 0;">Complete Cycle & Close Feedback Loop</h3>
                <p style="color: #A9B3AC; max-width: 600px; margin: 0 auto 16px auto; font-size: 0.9rem;">
                    Marking the cycle complete writes collection outcomes back into the historical time-series.
                    Collected bins are reset to empty (0-5%), advancing the simulation clock and refreshing predictions for the next operational shift.
                </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='primary-cta' style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
        if st.button("✅ Complete Collection Cycle & Reset Bins", key="complete_cycle_btn"):
            # Update state: reset collected bins
            collected_bin_ids = set()
            for r in routes:
                collected_bin_ids.update(r.get("stops_bin_ids", r.get("stop_bin_ids", [])))

            # Reset collected bins to 2-5% fill in current_state_df
            new_state = st.session_state.current_state_df.copy()
            rate_col = "accumulation_rate_pct_per_hr" if "accumulation_rate_pct_per_hr" in pred_df.columns else ("accumulation_rate" if "accumulation_rate" in pred_df.columns else None)
            rate_map = dict(zip(pred_df["bin_id"], pred_df[rate_col])) if rate_col else {}
            now_ts = pd.Timestamp.now()
            cycle_rng = np.random.default_rng(seed=42 + st.session_state.cycle_count)

            for bid in collected_bin_ids:
                mask = new_state["bin_id"] == bid
                new_state.loc[mask, "current_fill_pct"] = float(cycle_rng.uniform(2.0, 6.0))
                if "last_collected_at" in new_state.columns:
                    new_state.loc[mask, "last_collected_at"] = now_ts
                if "hours_since_collection" in new_state.columns:
                    new_state.loc[mask, "hours_since_collection"] = 0.0

            # Advance uncollected bins slightly (4 hours accumulation)
            for bid in set(new_state["bin_id"]) - collected_bin_ids:
                mask = new_state["bin_id"] == bid
                curr = float(new_state.loc[mask, "current_fill_pct"].values[0])
                rate = float(rate_map.get(bid, cycle_rng.uniform(0.8, 4.0)))
                new_state.loc[mask, "current_fill_pct"] = min(100.0, round(curr + rate * 4.0, 1))
                if "hours_since_collection" in new_state.columns:
                    new_state.loc[mask, "hours_since_collection"] = round(float(new_state.loc[mask, "hours_since_collection"].values[0]) + 4.0, 1)

            st.session_state.current_state_df = new_state
            st.session_state.cycle_count += 1
            st.session_state.cvrp_solution = None
            st.session_state.include_ids = set()
            st.session_state.exclude_ids = set()
            st.session_state.current_screen = "Dashboard"

            st.success(f"Cycle completed successfully! {len(collected_bin_ids)} bins emptied. Returning to Dashboard...")
            time.sleep(1)
            st.rerun()
        st.markdown("</div></div>", unsafe_allow_html=True)
