"""
BinSense — Fill-Level Prediction Module

Forecasts each bin's fill level over a configurable future horizon
using linear regression on recent fill history.

Usage:
    from src.predict import predict_fill, predict_all_bins
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.config import PREDICTION_HORIZON_HOURS, PREDICTION_LOOKBACK_HOURS


def predict_fill(bin_history, horizon_hours=PREDICTION_HORIZON_HOURS,
                 lookback_hours=PREDICTION_LOOKBACK_HOURS):
    """
    Predict a single bin's fill level at a future horizon.

    Uses linear regression on the most recent `lookback_hours` of fill
    readings to estimate the accumulation rate, then extrapolates.

    Parameters
    ----------
    bin_history : pd.DataFrame
        Fill history for ONE bin with columns: timestamp, fill_pct.
        Must be sorted by timestamp ascending.
    horizon_hours : float
        How many hours ahead to forecast.
    lookback_hours : float
        How many hours of recent history to use for the regression.

    Returns
    -------
    dict
        {
            "predicted_fill_pct": float,   # predicted fill at horizon (capped 0-100)
            "time_to_overflow_hours": float, # estimated hours until 100% (inf if rate <= 0)
            "accumulation_rate": float,      # estimated %/hour from regression
            "current_fill_pct": float,       # latest reading
            "confidence": str,               # "high"/"medium"/"low" based on data quality
        }
    """
    if bin_history.empty:
        return {
            "predicted_fill_pct": 0.0,
            "time_to_overflow_hours": float("inf"),
            "accumulation_rate": 0.0,
            "current_fill_pct": 0.0,
            "confidence": "low",
        }

    # Sort and get recent readings within lookback window
    hist = bin_history.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    hist = hist.sort_values("timestamp")
    latest_time = hist["timestamp"].max()
    cutoff = latest_time - pd.Timedelta(hours=lookback_hours)
    recent = hist[hist["timestamp"] >= cutoff].copy()

    if len(recent) < 2:
        # Not enough data for regression — fall back to last known value
        current = hist["fill_pct"].iloc[-1]
        return {
            "predicted_fill_pct": min(current, 100.0),
            "time_to_overflow_hours": float("inf"),
            "accumulation_rate": 0.0,
            "current_fill_pct": current,
            "confidence": "low",
        }

    # Detect collection resets: find the last significant drop in the recent window
    fills = recent["fill_pct"].values
    last_reset_idx = 0
    for i in range(1, len(fills)):
        if fills[i] < fills[i - 1] - 20:  # drop > 20% = collection event
            last_reset_idx = i

    # Use only post-reset data for regression (otherwise the reset confuses the trend)
    recent = recent.iloc[last_reset_idx:]

    if len(recent) < 2:
        current = hist["fill_pct"].iloc[-1]
        return {
            "predicted_fill_pct": min(current, 100.0),
            "time_to_overflow_hours": float("inf"),
            "accumulation_rate": 0.0,
            "current_fill_pct": current,
            "confidence": "low",
        }

    # Convert timestamps to hours since first reading (for regression)
    t0 = recent["timestamp"].min()
    recent = recent.copy()
    recent["hours"] = (recent["timestamp"] - t0).dt.total_seconds() / 3600.0

    X = recent["hours"].values.reshape(-1, 1)
    y = recent["fill_pct"].values

    # Fit linear regression: fill_pct = slope * hours + intercept
    model = LinearRegression()
    model.fit(X, y)

    accumulation_rate = float(model.coef_[0])  # %/hour
    current_fill = float(y[-1])
    current_hours = float(X[-1][0])

    # Predict fill at horizon
    future_hours = current_hours + horizon_hours
    predicted_fill = float(model.predict([[future_hours]])[0])
    predicted_fill = max(0.0, min(100.0, predicted_fill))  # clamp to [0, 100]

    # Time to overflow (100%)
    if accumulation_rate > 0.01:
        remaining = 100.0 - current_fill
        time_to_overflow = remaining / accumulation_rate
        time_to_overflow = max(0.0, time_to_overflow)
    else:
        time_to_overflow = float("inf")

    # Confidence based on data points and R²
    r_squared = model.score(X, y)
    if len(recent) >= 10 and r_squared >= 0.7:
        confidence = "high"
    elif len(recent) >= 5 and r_squared >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "predicted_fill_pct": round(predicted_fill, 1),
        "time_to_overflow_hours": round(time_to_overflow, 1),
        "accumulation_rate": round(accumulation_rate, 3),
        "current_fill_pct": round(current_fill, 1),
        "confidence": confidence,
    }


def predict_all_bins(bins_df, history_df, horizon_hours=PREDICTION_HORIZON_HOURS,
                     lookback_hours=PREDICTION_LOOKBACK_HOURS):
    """
    Run predictions for all bins.

    Parameters
    ----------
    bins_df : pd.DataFrame
        Bin dataset with at least bin_id column.
    history_df : pd.DataFrame
        Full fill history with columns: bin_id, timestamp, fill_pct.
    horizon_hours : float
        Forecast horizon in hours.
    lookback_hours : float
        Lookback window for regression.

    Returns
    -------
    pd.DataFrame
        bins_df enriched with prediction columns:
        predicted_fill_pct, time_to_overflow_hours, accumulation_rate,
        current_fill_pct, confidence
    """
    predictions = []

    for bin_id in bins_df["bin_id"]:
        bin_hist = history_df[history_df["bin_id"] == bin_id].copy()
        pred = predict_fill(
            bin_hist,
            horizon_hours=horizon_hours,
            lookback_hours=lookback_hours,
        )
        pred["bin_id"] = bin_id
        predictions.append(pred)

    pred_df = pd.DataFrame(predictions)

    # Drop current_fill_pct from predictions if it already exists in bins_df
    # to avoid _x/_y suffix collision during merge
    if "current_fill_pct" in bins_df.columns and "current_fill_pct" in pred_df.columns:
        pred_df = pred_df.drop(columns=["current_fill_pct"])

    # Merge predictions with bin data
    result = bins_df.merge(pred_df, on="bin_id")

    # Add accumulation_rate_pct_per_hr alias for consistency
    if "accumulation_rate" in result.columns:
        result["accumulation_rate_pct_per_hr"] = result["accumulation_rate"]

    return result


if __name__ == "__main__":
    from src.simulate import generate_bins, generate_fill_history, _load_road_graph

    print("BinSense - Prediction Module Test")
    print("-" * 40)

    G = _load_road_graph()
    bins_df = generate_bins(G)
    history_df = generate_fill_history(bins_df)

    print(f"Running predictions for {len(bins_df)} bins "
          f"(horizon={PREDICTION_HORIZON_HOURS}h, "
          f"lookback={PREDICTION_LOOKBACK_HOURS}h)...\n")

    result = predict_all_bins(bins_df, history_df)

    print(result[["bin_id", "zone", "current_fill_pct", "predicted_fill_pct",
                   "time_to_overflow_hours", "accumulation_rate", "confidence"]]
          .to_string(index=False))

    print(f"\n--- Prediction Summary ---")
    print(f"Bins predicted to overflow within {PREDICTION_HORIZON_HOURS}h: "
          f"{(result['predicted_fill_pct'] >= 95).sum()}")
    print(f"Average time-to-overflow: "
          f"{result[result['time_to_overflow_hours'] < float('inf')]['time_to_overflow_hours'].mean():.1f}h")
    print(f"Confidence: {result['confidence'].value_counts().to_dict()}")
