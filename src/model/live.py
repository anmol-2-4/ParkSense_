"""Real-time (or latest-observed) occupancy for inference.

Loads the latest zone occupancy snapshot written at training time so predictions
for "now" use recent data as lags. Can be replaced or augmented with a live
feed (e.g. LADOT real-time API) in production.
"""
from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd

class LiveOccupancyProvider:
    """Provides latest observed occupancy per zone for use as model inputs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._df: Optional[pd.DataFrame] = None
        self._lock = Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._df = None
            return
        try:
            self._df = pd.read_csv(self.path)
        except Exception:
            self._df = None

    def update(self, df: pd.DataFrame) -> None:
        """Replace the live dataframe (used by simulated or real-time feeds)."""
        with self._lock:
            self._df = df

    def get_latest(self, zone_id: int) -> Optional[dict]:
        """Return last_known_occupancy (list of 4 lags) and same_hour_prev_week for a zone."""
        with self._lock:
            if self._df is None:
                return None
            row = self._df[self._df["zone_id"] == zone_id]
            if row.empty:
                return None
            row = row.iloc[0]
            return {
                "last_known_occupancy": [
                    float(row.get("lag_1", 0.5)),
                    float(row.get("lag_2", 0.5)),
                    float(row.get("lag_3", 0.5)),
                    float(row.get("lag_4", 0.5)),
                ],
                "same_hour_prev_week": float(row.get("same_hour_prev_week", 0.5)),
            }

    def get_latest_observed_rate(self, zone_id: int) -> Optional[float]:
        """Return the most recent occupancy rate (lag_1) for display as 'current observed'."""
        info = self.get_latest(zone_id)
        if not info or not info["last_known_occupancy"]:
            return None
        return info["last_known_occupancy"][0]

    @property
    def available(self) -> bool:
        with self._lock:
            return self._df is not None and len(self._df) > 0


class SimulatedLiveFeed:
    """Replay historical zone occupancy as a simulated real-time feed."""

    def __init__(self, zone_occ: pd.DataFrame, bucket_minutes: int, history_days: int = 1):
        df = zone_occ.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["zone_id", "timestamp"])

        self._zone_series = {
            int(zid): sub.set_index("timestamp")["occupancy_rate"].sort_index()
            for zid, sub in df.groupby("zone_id")
        }
        self._timestamps = sorted(df["timestamp"].unique())
        if not self._timestamps:
            self._idx = 0
        else:
            steps_per_day = int((24 * 60) / max(bucket_minutes, 1))
            history_steps = max(1, steps_per_day * max(history_days, 1))
            self._idx = max(0, len(self._timestamps) - history_steps)

    def next_snapshot(self) -> tuple[pd.DataFrame, pd.Timestamp | None]:
        if not self._timestamps:
            return pd.DataFrame(columns=["zone_id", "lag_1", "lag_2", "lag_3", "lag_4", "same_hour_prev_week"]), None

        ts = self._timestamps[self._idx]
        self._idx = (self._idx + 1) % len(self._timestamps)

        rows = []
        prev_week_ts = ts - pd.Timedelta(days=7)
        for zid, series in self._zone_series.items():
            history = series.loc[:ts].tail(4)
            rates = history.tolist()
            while len(rates) < 4:
                rates.insert(0, 0.5)
            lag_1 = rates[-1] if rates else 0.5
            lag_2 = rates[-2] if len(rates) >= 2 else 0.5
            lag_3 = rates[-3] if len(rates) >= 3 else 0.5
            lag_4 = rates[-4] if len(rates) >= 4 else 0.5
            same_hour_prev_week = float(series.loc[prev_week_ts]) if prev_week_ts in series.index else 0.5
            rows.append(
                {
                    "zone_id": zid,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "lag_4": lag_4,
                    "same_hour_prev_week": same_hour_prev_week,
                }
            )
        return pd.DataFrame(rows), ts
