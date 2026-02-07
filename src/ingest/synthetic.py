"""Generate synthetic parking occupancy and inventory for prototype.
   Replace with LADOT downloaders for real data."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_synthetic_inventory(
    n_spaces: int = 500,
    n_clusters: int = 24,
    center_lat: float = 34.05,
    center_lon: float = -118.25,
    radius_deg: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a meter/space inventory with space_id, lat, lon, zone (to be set later)."""
    rng = np.random.default_rng(seed)
    # Cluster centers
    centers_lat = center_lat + rng.uniform(-radius_deg, radius_deg, n_clusters)
    centers_lon = center_lon + rng.uniform(-radius_deg, radius_deg, n_clusters)
    # Assign each space to a cluster and add jitter
    cluster_id = rng.integers(0, n_clusters, size=n_spaces)
    lat = centers_lat[cluster_id] + rng.uniform(-0.005, 0.005, n_spaces)
    lon = centers_lon[cluster_id] + rng.uniform(-0.005, 0.005, n_spaces)
    space_ids = [f"SP_{i:05d}" for i in range(n_spaces)]
    return pd.DataFrame({
        "space_id": space_ids,
        "lat": lat,
        "lon": lon,
        "cluster_id": cluster_id,
    })


def generate_synthetic_occupancy(
    inventory: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    bucket_minutes: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate occupancy (0 or 1) per space per time bucket with realistic patterns."""
    rng = np.random.default_rng(seed)
    space_ids = inventory["space_id"].tolist()
    buckets = pd.date_range(start_date, end_date, freq=f"{bucket_minutes}min")
    # Base occupancy by hour (peak higher) and day (weekday higher in business areas)
    records = []
    for ts in buckets:
        hour = ts.hour
        dow = ts.dayofweek
        is_weekend = dow >= 5
        # Peak hours 8-10, 17-19
        peak = (8 <= hour <= 10) or (17 <= hour <= 19)
        base_occ = 0.4 + 0.2 * peak + 0.15 * (not is_weekend) + rng.uniform(-0.1, 0.1)
        base_occ = np.clip(base_occ, 0.1, 0.95)
        for sid in space_ids:
            occ = 1 if rng.random() < base_occ else 0
            records.append({"space_id": sid, "timestamp": ts, "occupied": occ})
    return pd.DataFrame(records)


def ensure_data_dirs(raw_dir: Path, processed_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


def run_synthetic_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    n_spaces: int = 500,
    n_zones: int = 24,
    days_historical: int = 60,
    bucket_minutes: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic inventory and occupancy and save to data dirs. Returns (inventory, occupancy)."""
    base = Path(__file__).resolve().parent.parent.parent
    raw = base / raw_dir
    processed = base / processed_dir
    ensure_data_dirs(raw, processed)

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_historical)

    inventory = generate_synthetic_inventory(n_spaces=n_spaces, n_clusters=n_zones, seed=seed)
    inventory.to_csv(raw / "inventory.csv", index=False)

    occupancy = generate_synthetic_occupancy(
        inventory, start_date, end_date, bucket_minutes=bucket_minutes, seed=seed
    )
    # Save in chunks or sample for large data
    occupancy.to_parquet(processed / "occupancy.parquet", index=False)

    return inventory, occupancy
