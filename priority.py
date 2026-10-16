"""
BinSense — Dynamic Priority Engine

Computes a composite priority score per bin using multiple signals,
then classifies bins into urgency tiers: Critical / High / Medium / Low.

All weights and thresholds are config-driven (no hardcoding).

Usage:
    from src.priority import score_bin, score_all_bins
"""

import numpy as np
import pandas as pd

from src.config import PRIORITY_WEIGHTS, TIER_THRESHOLDS, TIER_ORDER


def _normalize(value, min_val=0.0, max_val=100.0):
    """Normalize a value to [0, 100] range."""
    if max_val == min_val:
        return 50.0
    return max(0.0, min(100.0, (value - min_val) / (max_val - min_val) * 100.0))


def _classify_tier(score, thresholds=TIER_THRESHOLDS):
    """Map a numeric priority score to an urgency tier string."""
    if score >= thresholds["critical"]:
        return "critical"
    elif score >= thresholds["high"]:
        return "high"
    elif score >= thresholds["medium"]:
        return "medium"
    else:
        return "low"


def score_bin(
    current_fill_pct,
    predicted_fill_pct,
    accumulation_rate,
    time_to_overflow_hours,
    hours_since_collection,
    weights=PRIORITY_WEIGHTS,
    thresholds=TIER_THRESHOLDS,
    max_accumulation_rate=6.0,
    max_hours_since_collection=72.0,
):
    """
    Compute the composite priority score and urgency tier for a single bin.

    Parameters
    ----------
    current_fill_pct : float
        Current fill level (0-100).
    predicted_fill_pct : float
        Predicted fill level at horizon (0-100).
    accumulation_rate : float
        Estimated fill rate in %/hour.
    time_to_overflow_hours : float
        Estimated hours until the bin reaches 100%.
    hours_since_collection : float
        Hours since the bin was last emptied.
    weights : dict
        Priority weight configuration.
    thresholds : dict
        Tier threshold configuration.
    max_accumulation_rate : float
        Upper bound for rate normalization.
    max_hours_since_collection : float
        Upper bound for hours-since-collection normalization.

    Returns
    -------
    dict
        {
            "priority_score": float (0-100),
            "tier": str ("critical"/"high"/"medium"/"low"),
            "components": dict  # breakdown of each weighted factor
        }
    """
    # Normalize each factor to [0, 100]
    norm_current = _normalize(current_fill_pct, 0, 100)
    norm_predicted = _normalize(predicted_fill_pct, 0, 100)
    norm_rate = _normalize(accumulation_rate, 0, max_accumulation_rate)

    # Overflow risk: inversely proportional to time-to-overflow
    # Lower time-to-overflow = higher risk
    if time_to_overflow_hours <= 0:
        norm_overflow = 100.0
    elif time_to_overflow_hours == float("inf"):
        norm_overflow = 0.0
    else:
        # Map: 0h → 100, 24h → ~15, 48h → ~5, inf → 0
        norm_overflow = min(100.0, max(0.0, 100.0 * np.exp(-time_to_overflow_hours / 12.0) * 2.5))

    norm_hours = _normalize(hours_since_collection, 0, max_hours_since_collection)

    # Compute weighted composite score
    w = weights
    components = {
        "current_fill": w["w1_current_fill"] * norm_current,
        "predicted_fill": w["w2_predicted_fill"] * norm_predicted,
        "accumulation_rate": w["w3_accumulation_rate"] * norm_rate,
        "overflow_risk": w["w4_overflow_risk"] * norm_overflow,
        "time_since_collection": w["w5_time_since_collection"] * norm_hours,
    }

    priority_score = sum(components.values())
    priority_score = round(max(0.0, min(100.0, priority_score)), 1)

    tier = _classify_tier(priority_score, thresholds)

    return {
        "priority_score": priority_score,
        "tier": tier,
        "components": components,
    }


def score_all_bins(
    bins_with_predictions_df,
    weights=PRIORITY_WEIGHTS,
    thresholds=TIER_THRESHOLDS,
):
    """
    Score and tier-classify all bins.

    Parameters
    ----------
    bins_with_predictions_df : pd.DataFrame
        Must have columns: current_fill_pct, predicted_fill_pct,
        accumulation_rate, time_to_overflow_hours, hours_since_collection.
    weights : dict
        Priority weights (from config or UI override).
    thresholds : dict
        Tier thresholds (from config or UI override).

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with: priority_score, tier
    """
    scores = []
    tiers = []

    for _, row in bins_with_predictions_df.iterrows():
        result = score_bin(
            current_fill_pct=row.get("current_fill_pct", 0),
            predicted_fill_pct=row.get("predicted_fill_pct", 0),
            accumulation_rate=row.get("accumulation_rate", 0),
            time_to_overflow_hours=row.get("time_to_overflow_hours", float("inf")),
            hours_since_collection=row.get("hours_since_collection", 0),
            weights=weights,
            thresholds=thresholds,
        )
        scores.append(result["priority_score"])
        tiers.append(result["tier"])

    result_df = bins_with_predictions_df.copy()
    result_df["priority_score"] = scores
    result_df["tier"] = pd.Categorical(tiers, categories=TIER_ORDER, ordered=True)

    return result_df


def get_tier_summary(scored_df):
    """
    Generate a summary of bins per urgency tier.

    Returns
    -------
    dict
        {"critical": count, "high": count, "medium": count, "low": count}
    """
    counts = scored_df["tier"].value_counts()
    return {tier: int(counts.get(tier, 0)) for tier in TIER_ORDER}


if __name__ == "__main__":
    from src.simulate import (
        generate_bins, generate_fill_history,
        get_current_state, _load_road_graph,
    )
    from src.predict import predict_all_bins

    print("BinSense - Priority Engine Test")
    print("-" * 40)

    G = _load_road_graph()
    bins_df = generate_bins(G)
    history_df = generate_fill_history(bins_df)
    state_df = get_current_state(bins_df, history_df)
    pred_df = predict_all_bins(state_df, history_df)

    print("Scoring all bins...")
    scored_df = score_all_bins(pred_df)

    print(scored_df[["bin_id", "zone", "current_fill_pct", "predicted_fill_pct",
                      "priority_score", "tier"]]
          .sort_values("priority_score", ascending=False)
          .to_string(index=False))

    print("\n--- Tier Summary ---")
    summary = get_tier_summary(scored_df)
    for tier, count in summary.items():
        print(f"  {tier:10s}: {count}")

    print(f"\nScore range: {scored_df['priority_score'].min():.1f} - "
          f"{scored_df['priority_score'].max():.1f}")
    print(f"Mean score: {scored_df['priority_score'].mean():.1f}")
