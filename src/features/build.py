"""Feature engineering for parking availability prediction."""
import pandas as pd
import numpy as np
from typing import Optional


def _temporal_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    t = pd.to_datetime(df[ts_col])
    df = df.copy()
    df["hour"] = t.dt.hour
    df["day_of_week"] = t.dt.dayofweek
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    df["month"] = t.dt.month
    return df


def _lag_features(
    zone_occ: pd.DataFrame,
    lags: list[int],
    target_col: str = "occupancy_rate",
) -> pd.DataFrame:
    """Add lagged occupancy per zone. Expects sorted by zone_id, timestamp."""
    zone_occ = zone_occ.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    out = []
    for zid, g in zone_occ.groupby("zone_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        for lag in lags:
            g[f"lag_{lag}"] = g[target_col].shift(lag)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _same_hour_prev_week(zone_occ: pd.DataFrame, target_col: str = "occupancy_rate") -> pd.DataFrame:
    """Add occupancy at same hour, 7 days ago (approximate)."""
    zone_occ = zone_occ.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    zone_occ["timestamp_prev_week"] = zone_occ["timestamp"] - pd.Timedelta(days=7)
    merged = zone_occ.merge(
        zone_occ[["zone_id", "timestamp", target_col]].rename(
            columns={"timestamp": "timestamp_prev_week", target_col: "same_hour_prev_week"}
        ),
        on=["zone_id", "timestamp_prev_week"],
        how="left",
    )
    return merged.drop(columns=["timestamp_prev_week"], errors="ignore")


def _traffic_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Peak hour flags as traffic proxy when no real traffic data."""
    df = df.copy()
    df["peak_morning"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
    df["peak_evening"] = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(int)
    return df


def _events_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder: event_day = 0 for all. Replace with real event calendar join."""
    df = df.copy()
    df["event_day"] = 0
    return df


def build_features(
    zone_occupancy: pd.DataFrame,
    lags: list[int] = [1, 2, 3, 4],
    same_hour_prev_week: bool = True,
    traffic_proxy: bool = True,
    events_proxy: bool = True,
    target_col: str = "occupancy_rate",
) -> pd.DataFrame:
    """Build full feature matrix from zone-level occupancy. Drops rows with NaN from lags."""
    df = _temporal_features(zone_occupancy, "timestamp")
    df = _lag_features(df, lags, target_col=target_col)
    if same_hour_prev_week:
        df = _same_hour_prev_week(df, target_col=target_col)
    if traffic_proxy:
        df = _traffic_proxy(df)
    if events_proxy:
        df = _events_proxy(df)
    # Drop rows where lags are NaN (start of series)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df = df.dropna(subset=lag_cols)
    return df


FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_4",
    "same_hour_prev_week",
    "peak_morning",
    "peak_evening",
    "event_day",
]


def build_features_for_inference(
    zone_id: int,
    timestamp: pd.Timestamp,
    last_known_occupancy: Optional[list[float]],
    zone_hist: Optional[pd.DataFrame],
    same_hour_prev_week_val: Optional[float] = None,
) -> dict:
    """Build single-row feature dict for API inference when we don't have full history.
    last_known_occupancy: [most recent, ..., oldest] e.g. [0.7, 0.65, 0.6, 0.55] for lags 1..4
    """
    t = pd.to_datetime(timestamp)
    d = {
        "zone_id": zone_id,
        "hour": t.hour,
        "day_of_week": t.dayofweek,
        "is_weekend": 1 if t.dayofweek >= 5 else 0,
        "month": t.month,
        "peak_morning": 1 if 7 <= t.hour <= 9 else 0,
        "peak_evening": 1 if 17 <= t.hour <= 19 else 0,
        "event_day": 0,
    }
    if last_known_occupancy:
        for i, v in enumerate(last_known_occupancy[:4], start=1):
            d[f"lag_{i}"] = v
    else:
        for i in range(1, 5):
            d[f"lag_{i}"] = 0.5  # neutral default
    if same_hour_prev_week_val is not None:
        d["same_hour_prev_week"] = same_hour_prev_week_val
    else:
        d["same_hour_prev_week"] = 0.5
    return d
