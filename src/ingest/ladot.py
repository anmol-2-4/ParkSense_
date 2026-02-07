"""Ingest LADOT parking data from data.lacity.org (Socrata)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from collections import deque
import time
import logging
import json

import httpx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


@dataclass
class LADOTConfig:
    base_url: str = "https://data.lacity.org/resource"
    inventory_id: str = "s49e-q6j2"
    live_id: str = "e7h6-4a3e"
    archive_id: str = "cj8s-ivry"
    archive_csv_url: str = "https://data.lacity.org/api/views/cj8s-ivry/rows.csv?accessType=DOWNLOAD"
    max_rows: int = 200_000
    page_size: int = 50_000
    history_days: int = 30
    live_history_minutes: int = 60
    live_poll_seconds: int = 60
    app_token: Optional[str] = None


def _pick_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _pick_col_contains(cols: list[str], patterns: list[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in cols}
    for pat in patterns:
        for lc, orig in lowered.items():
            if pat in lc:
                return orig
    return None


def _fetch_socrata(
    resource_id: str,
    base_url: str,
    params: dict,
    max_rows: int,
    page_size: int,
    app_token: Optional[str] = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    headers = {"X-App-Token": app_token} if app_token else None

    while True:
        page_params = {**params, "$limit": page_size, "$offset": offset}
        url = f"{base_url}/{resource_id}.json"
        last_exc = None
        for attempt in range(3):
            try:
                r = httpx.get(url, params=page_params, headers=headers, timeout=30)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(1 + attempt)
                batch = None
        if batch is None:
            raise last_exc  # type: ignore[misc]
        if not batch:
            break
        rows.extend(batch)
        if max_rows and len(rows) >= max_rows:
            rows = rows[:max_rows]
            break
        offset += page_size

    return pd.DataFrame(rows)


def _normalize_inventory(df: pd.DataFrame, n_zones: int, seed: int) -> pd.DataFrame:
    cols = list(df.columns)
    space_col = _pick_col(cols, ["space_id", "spaceid", "spaceid_1", "meter_id", "meterid", "space"])
    lat_col = _pick_col(cols, ["latitude", "lat", "y", "y_coordinate"])
    lon_col = _pick_col(cols, ["longitude", "lon", "x", "x_coordinate"])
    if not space_col:
        space_col = _pick_col_contains(cols, ["space", "meter", "stall"])

    has_latlng = "latlng" in cols
    if not lat_col and not has_latlng:
        lat_col = _pick_col_contains(cols, ["lat"])
    if not lon_col and not has_latlng:
        lon_col = _pick_col_contains(cols, ["lon", "lng", "long"])

    if not space_col:
        raise ValueError("Inventory dataset missing space_id column.")

    if has_latlng:
        inv = df[[space_col, "latlng"]].copy()

        def _extract_latlng(v):
            if isinstance(v, dict):
                lat = v.get("latitude") or v.get("lat")
                lon = v.get("longitude") or v.get("lon") or v.get("lng")
                return lat, lon
            if isinstance(v, str):
                try:
                    obj = json.loads(v)
                    if isinstance(obj, dict):
                        lat = obj.get("latitude") or obj.get("lat")
                        lon = obj.get("longitude") or obj.get("lon") or obj.get("lng")
                        return lat, lon
                except Exception:
                    return None, None
            return None, None

        coords = inv["latlng"].apply(_extract_latlng)
        inv["lat"] = coords.apply(lambda x: x[0])
        inv["lon"] = coords.apply(lambda x: x[1])
        inv = inv.drop(columns=["latlng"])
        inv.columns = ["space_id", "lat", "lon"]
    elif lat_col and lon_col:
        inv = df[[space_col, lat_col, lon_col]].copy()
        inv.columns = ["space_id", "lat", "lon"]
    else:
        raise ValueError("Inventory dataset missing lat/lon columns.")
    inv["space_id"] = inv["space_id"].astype(str)
    inv["lat"] = pd.to_numeric(inv["lat"], errors="coerce")
    inv["lon"] = pd.to_numeric(inv["lon"], errors="coerce")
    inv = inv.dropna(subset=["lat", "lon"])

    if len(inv) == 0:
        raise ValueError("Inventory dataset has no valid coordinates.")

    n_clusters = min(int(n_zones), len(inv))
    if n_clusters <= 0:
        n_clusters = 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    inv["cluster_id"] = kmeans.fit_predict(inv[["lat", "lon"]])
    return inv


def _normalize_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    space_col = _pick_col(cols, ["space_id", "spaceid", "spaceid_1", "meter_id", "meterid", "space"])
    time_col = _pick_col(cols, ["event_time", "event_time_utc", "timestamp", "date_time", "occupancy_date", "eventtime"])
    occ_col = _pick_col(cols, ["occupancystate", "occupancy_state", "occupied", "occ", "occupancy"])

    if not space_col:
        space_col = _pick_col_contains(cols, ["space", "meter", "stall"])
    if not time_col:
        time_col = _pick_col_contains(cols, ["time", "date", "timestamp", "event", "updated"])
    if not occ_col:
        occ_col = _pick_col_contains(cols, ["occup", "status", "state", "avail"])

    if not space_col or not time_col or not occ_col:
        logger.error(
            "Occupancy dataset missing required columns. Found columns: %s",
            cols,
        )
        raise ValueError("Occupancy dataset missing space_id/time/occupancy columns.")

    out = df[[space_col, time_col, occ_col]].copy()
    out.columns = ["space_id", "timestamp", "occupied_raw"]
    out["space_id"] = out["space_id"].astype(str)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)

    def _to_occ(v) -> int:
        if pd.isna(v):
            return 0
        if isinstance(v, (int, float, np.integer, np.floating)):
            return int(float(v) > 0)
        s = str(v).strip().lower()
        return 1 if s in {"1", "true", "occupied", "occ", "busy", "full"} else 0

    out["occupied"] = out["occupied_raw"].map(_to_occ).astype(int)
    out = out.dropna(subset=["timestamp"])
    return out[["space_id", "timestamp", "occupied"]]


def download_inventory(cfg: LADOTConfig, n_zones: int, seed: int) -> pd.DataFrame:
    params = {
        "$select": "spaceid,latlng",
        "$where": "latlng IS NOT NULL",
    }
    df = _fetch_socrata(
        resource_id=cfg.inventory_id,
        base_url=cfg.base_url,
        params=params,
        max_rows=cfg.max_rows,
        page_size=cfg.page_size,
        app_token=cfg.app_token,
    )
    return _normalize_inventory(df, n_zones=n_zones, seed=seed)


def download_archive(cfg: LADOTConfig, days: int) -> pd.DataFrame:
    # Socrata API may not expose archive rows; fall back to CSV download
    try:
        time_col = "eventtime"
        start = datetime.utcnow() - timedelta(days=days)
        where = f"{time_col} >= '{start.isoformat()}'"
        df = _fetch_socrata(
            resource_id=cfg.archive_id,
            base_url=cfg.base_url,
            params={
                "$select": "spaceid,eventtime,occupancystate",
                "$where": where,
                "$order": f"{time_col} asc",
            },
            max_rows=cfg.max_rows,
            page_size=cfg.page_size,
            app_token=cfg.app_token,
        )
        if len(df) > 0:
            return _normalize_occupancy(df)
    except Exception:
        logger.info("Socrata archive query unavailable; falling back to CSV download.")

    # CSV fallback (large): read only first max_rows for hackathon-scale training
    try:
        url = cfg.archive_csv_url
        header = pd.read_csv(url, nrows=0)
        cols = list(header.columns)
        space_col = _pick_col(cols, ["spaceid", "space_id", "spaceid_1", "meter_id", "meterid"])
        time_col = _pick_col(cols, ["eventtime", "event_time", "timestamp", "date_time"])
        occ_col = _pick_col(cols, ["occupancystate", "occupancy_state", "occupied", "occupancy"])
        if not space_col or not time_col or not occ_col:
            raise ValueError("Archive CSV missing required columns.")

        df = pd.read_csv(
            url,
            usecols=[space_col, time_col, occ_col],
            nrows=cfg.max_rows,
            low_memory=False,
        )
        return _normalize_occupancy(df)
    except Exception as e:
        logger.warning("Archive CSV fallback failed: %s", e)
        return pd.DataFrame(columns=["space_id", "timestamp", "occupied"])


def sample_live_history(cfg: LADOTConfig) -> pd.DataFrame:
    """Build a short historical window by polling the live feed."""
    minutes = max(5, int(cfg.live_history_minutes))
    poll_seconds = max(10, int(cfg.live_poll_seconds))
    end_time = datetime.utcnow() + timedelta(minutes=minutes)

    rows = []
    while datetime.utcnow() < end_time:
        try:
            df = _fetch_socrata(
                resource_id=cfg.live_id,
                base_url=cfg.base_url,
                params={"$select": "spaceid,eventtime,occupancystate"},
                max_rows=cfg.max_rows,
                page_size=cfg.page_size,
                app_token=cfg.app_token,
            )
            norm = _normalize_occupancy(df)
            if len(norm):
                rows.append(norm)
        except Exception as e:
            logger.warning("Live history poll failed: %s", e)
        time.sleep(poll_seconds)

    if not rows:
        return pd.DataFrame(columns=["space_id", "timestamp", "occupied"])
    return pd.concat(rows, ignore_index=True)


def run_ladot_pipeline(
    raw_dir: str,
    processed_dir: str,
    n_zones: int,
    seed: int,
    history_days: int,
    cfg: LADOTConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = Path(__file__).resolve().parent.parent.parent
    raw = base / raw_dir
    processed = base / processed_dir
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    inventory = download_inventory(cfg, n_zones=n_zones, seed=seed)
    inventory.to_csv(raw / "inventory.csv", index=False)

    occupancy = download_archive(cfg, days=history_days)
    if occupancy.empty:
        logger.info("Archive data unavailable; sampling live feed for training history.")
        occupancy = sample_live_history(cfg)
    occupancy.to_parquet(processed / "occupancy.parquet", index=False)
    return inventory, occupancy


class LADOTLiveUpdater:
    """Fetch real-time LADOT occupancy and build zone-level lags for inference."""

    def __init__(self, inventory: pd.DataFrame, cfg: LADOTConfig, n_zones: int):
        self.inventory = inventory[["space_id", "cluster_id"]].copy()
        self.cfg = cfg
        self.n_zones = int(n_zones)
        self._history: dict[int, deque] = {
            zid: deque(maxlen=4) for zid in range(self.n_zones)
        }

    def _fetch_live(self) -> pd.DataFrame:
        df = _fetch_socrata(
            resource_id=self.cfg.live_id,
            base_url=self.cfg.base_url,
            params={"$select": "spaceid,eventtime,occupancystate"},
            max_rows=self.cfg.max_rows,
            page_size=self.cfg.page_size,
            app_token=self.cfg.app_token,
        )
        return _normalize_occupancy(df)

    def next_snapshot(self) -> pd.DataFrame:
        occ = self._fetch_live()
        if occ.empty:
            logger.warning("LADOT live feed returned 0 rows")
        inv = self.inventory.copy()
        merged = occ.merge(inv, on="space_id", how="inner")
        if merged.empty:
            logger.warning(
                "LADOT live merge empty (occ=%d, inv=%d). Example occ ids: %s; inv ids: %s",
                len(occ),
                len(inv),
                occ["space_id"].head(5).tolist() if not occ.empty else [],
                inv["space_id"].head(5).tolist() if not inv.empty else [],
            )
            return pd.DataFrame(columns=["zone_id", "lag_1", "lag_2", "lag_3", "lag_4", "same_hour_prev_week"])

        agg = (
            merged.groupby("cluster_id")
            .agg(occupied_count=("occupied", "sum"), total_spaces=("occupied", "count"))
            .reset_index()
            .rename(columns={"cluster_id": "zone_id"})
        )
        agg["occupancy_rate"] = agg["occupied_count"] / agg["total_spaces"].clip(lower=1)

        for _, row in agg.iterrows():
            zid = int(row["zone_id"])
            rate = float(row["occupancy_rate"])
            if zid not in self._history:
                self._history[zid] = deque(maxlen=4)
            self._history[zid].append(rate)

        rows = []
        for zid in range(self.n_zones):
            hist = list(self._history.get(zid, deque()))
            while len(hist) < 4:
                hist.insert(0, 0.5)
            rows.append(
                {
                    "zone_id": zid,
                    "lag_1": hist[-1],
                    "lag_2": hist[-2],
                    "lag_3": hist[-3],
                    "lag_4": hist[-4],
                    "same_hour_prev_week": 0.5,
                }
            )
        return pd.DataFrame(rows)
