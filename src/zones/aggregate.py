"""Zone definition and occupancy aggregation."""
import pandas as pd
from pathlib import Path


def build_zone_occupancy(
    occupancy: pd.DataFrame,
    inventory: pd.DataFrame,
    zone_col: str = "cluster_id",
) -> pd.DataFrame:
    """Aggregate occupancy to zone-level per time bucket.
    occupancy: columns space_id, timestamp, occupied
    inventory: columns space_id, ... zone_col (zone id)
    Returns: zone_id, timestamp, total_spaces, occupied_count, occupancy_rate
    """
    occ = occupancy.merge(
        inventory[["space_id", zone_col]].rename(columns={zone_col: "zone_id"}),
        on="space_id",
        how="left",
    )
    agg = (
        occ.groupby(["zone_id", "timestamp"])
        .agg(occupied_count=("occupied", "sum"), total_spaces=("occupied", "count"))
        .reset_index()
    )
    agg["occupancy_rate"] = agg["occupied_count"] / agg["total_spaces"].clip(lower=1)
    return agg


def get_zone_metadata(inventory: pd.DataFrame, zone_col: str = "cluster_id") -> pd.DataFrame:
    """Zone metadata: zone_id, lat_centroid, lon_centroid, capacity."""
    meta = (
        inventory.groupby(zone_col)
        .agg(
            lat_centroid=("lat", "mean"),
            lon_centroid=("lon", "mean"),
            capacity=("space_id", "count"),
        )
        .reset_index()
        .rename(columns={zone_col: "zone_id"})
    )
    return meta


def load_and_aggregate(
    raw_dir: Path,
    processed_dir: Path,
    zone_col: str = "cluster_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load inventory and occupancy from disk, return (zone_occupancy, zone_metadata)."""
    inventory = pd.read_csv(raw_dir / "inventory.csv")
    occupancy = pd.read_parquet(processed_dir / "occupancy.parquet")
    zone_occ = build_zone_occupancy(occupancy, inventory, zone_col=zone_col)
    zone_meta = get_zone_metadata(inventory, zone_col=zone_col)
    return zone_occ, zone_meta
