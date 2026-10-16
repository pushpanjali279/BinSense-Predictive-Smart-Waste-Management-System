"""
BinSense — Bin Selection Module

Filters scored bins down to the set that should be serviced in the
current collection cycle, based on urgency tier and fleet capacity.
Supports manual include/exclude overrides.

Usage:
    from src.select import select_bins
"""

import pandas as pd

from src.config import (
    SELECTION_MIN_TIER,
    TIER_ORDER,
    NUM_VEHICLES,
    VEHICLE_CAPACITY_LITERS,
    FLEET_CAPACITY_BUFFER,
)


def select_bins(
    scored_df,
    fleet_capacity_liters=None,
    min_tier=SELECTION_MIN_TIER,
    include_ids=None,
    exclude_ids=None,
    capacity_buffer=FLEET_CAPACITY_BUFFER,
):
    """
    Select bins for the current collection cycle.

    Selection logic:
    1. Include all bins at or above `min_tier` urgency.
    2. Apply manual include/exclude overrides.
    3. If total demand exceeds fleet capacity (scaled by capacity_buffer), drop lowest-priority bins.
    4. Sort by priority (highest first).

    Parameters
    ----------
    scored_df : pd.DataFrame
        Bins with priority_score and tier columns (from priority engine).
    fleet_capacity_liters : float, optional
        Total fleet capacity. Defaults to NUM_VEHICLES * VEHICLE_CAPACITY_LITERS.
    min_tier : str
        Minimum tier to auto-include ("critical", "high", "medium", "low").
    include_ids : list[str], optional
        Bin IDs to force-include regardless of tier.
    exclude_ids : list[str], optional
        Bin IDs to force-exclude regardless of tier.
    capacity_buffer : float
        Fraction of fleet capacity to budget (e.g. 0.85 for 85%).

    Returns
    -------
    dict
        {
            "selected": pd.DataFrame,    # bins selected for collection
            "deferred": pd.DataFrame,     # bins deferred to next cycle
            "total_demand_liters": float, # total waste volume to collect
            "fleet_capacity_liters": float,
            "effective_capacity_liters": float,
            "capacity_exceeded": bool,
            "deferred_count": int,
        }
    """
    if fleet_capacity_liters is None:
        fleet_capacity_liters = NUM_VEHICLES * VEHICLE_CAPACITY_LITERS

    effective_capacity = fleet_capacity_liters * capacity_buffer

    include_ids = set(include_ids or [])
    exclude_ids = set(exclude_ids or [])

    # Determine the tier index cutoff
    tier_index = TIER_ORDER.index(min_tier)
    qualifying_tiers = set(TIER_ORDER[: tier_index + 1])  # critical, high, ... up to min_tier

    # Start with tier-based selection
    df = scored_df.copy()
    df["auto_selected"] = df["tier"].isin(qualifying_tiers)

    # Apply manual overrides
    df["selected"] = df.apply(
        lambda row: (
            True if row["bin_id"] in include_ids
            else (False if row["bin_id"] in exclude_ids
                  else row["auto_selected"])
        ),
        axis=1,
    )

    selected = df[df["selected"]].copy()
    not_selected = df[~df["selected"]].copy()

    # Sort selected by priority (highest first)
    selected = selected.sort_values("priority_score", ascending=False)

    # Estimate demand per bin (proportional to current fill and capacity)
    selected["estimated_demand_liters"] = (
        selected["current_fill_pct"] / 100.0 * selected["capacity_liters"]
    ).round(0)

    # Check fleet capacity constraint
    total_demand = selected["estimated_demand_liters"].sum()
    capacity_exceeded = total_demand > effective_capacity

    deferred = pd.DataFrame()
    deferred_count = 0

    if capacity_exceeded:
        # Drop lowest-priority bins until within effective capacity
        cumulative = 0
        keep_mask = []
        for _, row in selected.iterrows():
            if cumulative + row["estimated_demand_liters"] <= effective_capacity:
                cumulative += row["estimated_demand_liters"]
                keep_mask.append(True)
            else:
                keep_mask.append(False)

        deferred = selected[~pd.Series(keep_mask, index=selected.index)]
        selected = selected[pd.Series(keep_mask, index=selected.index)]
        deferred_count = len(deferred)
        total_demand = selected["estimated_demand_liters"].sum()

    # Also add demand column to not_selected for completeness
    if not not_selected.empty:
        not_selected["estimated_demand_liters"] = (
            not_selected["current_fill_pct"] / 100.0 * not_selected["capacity_liters"]
        ).round(0)
        deferred = pd.concat([deferred, not_selected], ignore_index=True)

    return {
        "selected": selected.reset_index(drop=True),
        "deferred": deferred.reset_index(drop=True),
        "total_demand_liters": total_demand,
        "fleet_capacity_liters": fleet_capacity_liters,
        "effective_capacity_liters": effective_capacity,
        "capacity_exceeded": capacity_exceeded,
        "deferred_count": deferred_count,
    }


if __name__ == "__main__":
    from src.simulate import (
        generate_bins, generate_fill_history,
        get_current_state, _load_road_graph,
    )
    from src.predict import predict_all_bins
    from src.priority import score_all_bins

    print("BinSense - Bin Selection Test")
    print("-" * 40)

    G = _load_road_graph()
    bins_df = generate_bins(G)
    history_df = generate_fill_history(bins_df)
    state_df = get_current_state(bins_df, history_df)
    pred_df = predict_all_bins(state_df, history_df)
    scored_df = score_all_bins(pred_df)

    result = select_bins(scored_df)

    print(f"Selected {len(result['selected'])} bins for collection:")
    if not result["selected"].empty:
        print(result["selected"][
            ["bin_id", "tier", "priority_score", "current_fill_pct",
             "estimated_demand_liters"]
        ].to_string(index=False))

    print(f"\nTotal demand: {result['total_demand_liters']:.0f} L")
    print(f"Fleet capacity: {result['fleet_capacity_liters']:.0f} L")
    print(f"Capacity exceeded: {result['capacity_exceeded']}")
    if result["deferred_count"] > 0:
        print(f"Deferred bins: {result['deferred_count']}")
