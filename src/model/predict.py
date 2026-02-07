"""Load trained model and run inference with confidence intervals."""
import json
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from ..config import get_config, get_project_root
from ..features.build import build_features_for_inference, FEATURE_COLS


class ParkingPredictor:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        with open(self.model_dir / "model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(self.model_dir / "quantile_models.pkl", "rb") as f:
            self.quantile_models = pickle.load(f)
        with open(self.model_dir / "feature_cols.json") as f:
            self.feature_cols = json.load(f)
        self.zone_meta = pd.read_csv(self.model_dir / "zone_metadata.csv")
        try:
            with open(self.model_dir / "metrics.json") as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            self.metrics = {}

    def _get_zone_hist(self, zone_id: int, current_ts: pd.Timestamp) -> pd.DataFrame | None:
        """Load latest occupancy history for a zone from processed data. Optional."""
        return None

    def predict(
        self,
        zone_id: int,
        timestamp: datetime | pd.Timestamp | None = None,
        last_known_occupancy: list[float] | None = None,
        same_hour_prev_week: float | None = None,
    ) -> dict:
        """Predict occupancy rate and confidence interval for a zone at a time."""
        ts = pd.Timestamp(timestamp or datetime.utcnow())
        feat = build_features_for_inference(
            zone_id=zone_id,
            timestamp=ts,
            last_known_occupancy=last_known_occupancy,
            zone_hist=None,
            same_hour_prev_week_val=same_hour_prev_week,
        )
        X = pd.DataFrame([feat])[self.feature_cols]
        # Fill missing cols with 0.5
        for c in self.feature_cols:
            if c not in X.columns:
                X[c] = 0.5
        X = X[self.feature_cols]

        point = float(np.clip(self.model.predict(X)[0], 0, 1))
        lower = float(np.clip(self.quantile_models.get(0.1, self.model).predict(X)[0], 0, 1))
        upper = float(np.clip(self.quantile_models.get(0.9, self.model).predict(X)[0], 0, 1))
        # Ensure lower <= point <= upper
        lower, upper = min(lower, point), max(upper, point)

        capacity = int(self.zone_meta[self.zone_meta["zone_id"] == zone_id]["capacity"].iloc[0])
        free_spaces = int(round((1 - point) * capacity))
        free_lower = int(round((1 - upper) * capacity))
        free_upper = int(round((1 - lower) * capacity))

        # Plain-language summary (standout)
        width = upper - lower
        if width < 0.2:
            summary = "High confidence—good time to plan to park here."
        elif width < 0.4:
            summary = "Moderate confidence—check again closer to arrival."
        else:
            summary = "Uncertain—check again closer to time or try another zone."

        return {
            "zone_id": zone_id,
            "timestamp": ts.isoformat(),
            "occupancy_rate": round(point, 3),
            "free_spaces": free_spaces,
            "capacity": capacity,
            "confidence_interval_90": {"lower": round(lower, 3), "upper": round(upper, 3)},
            "free_spaces_interval": {"lower": free_lower, "upper": free_upper},
            "summary": summary,
            "latest_observed_occupancy": None,  # API can override from LiveOccupancyProvider
        }

    def predict_all_zones(self, timestamp: datetime | pd.Timestamp | None = None) -> list[dict]:
        """Predict for every zone (e.g. for map). No per-zone history; uses defaults."""
        ts = timestamp or datetime.utcnow()
        out = []
        for zid in self.zone_meta["zone_id"].unique():
            out.append(self.predict(zone_id=int(zid), timestamp=ts))
        return out

    def get_zone_metadata(self) -> list[dict]:
        return self.zone_meta.to_dict("records")


def load_predictor(model_dir: Path | None = None) -> ParkingPredictor:
    root = get_project_root()
    cfg = get_config()
    if model_dir is None:
        model_dir = root / cfg["data"]["model_dir"]
    return ParkingPredictor(model_dir)
