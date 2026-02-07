"""Load config from YAML or env."""
from pathlib import Path
from copy import deepcopy
from typing import Any

_DEFAULT_CONFIG = {
    "city": "la",
    "time_bucket_minutes": 30,
    "n_zones": 24,
    "train_val_split_weeks": 4,
    "random_seed": 42,
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "model_dir": "data/models",
        "source": "ladot",  # ladot | synthetic
    },
    "features": {
        "temporal": True,
        "lags": [1, 2, 3, 4],
        "same_hour_prev_week": True,
        "traffic_proxy": True,
        "events_proxy": True,
    },
    "model": {
        "type": "lightgbm",
        "objective": "regression",
        "quantile_alpha": [0.1, 0.5, 0.9],
        "num_leaves": 31,
        "n_estimators": 200,
        "learning_rate": 0.05,
    },
    "ladot": {
        "base_url": "https://data.lacity.org/resource",
        "inventory_id": "s49e-q6j2",
        "live_id": "e7h6-4a3e",
        "archive_id": "cj8s-ivry",
        "archive_csv_url": "https://data.lacity.org/api/views/cj8s-ivry/rows.csv?accessType=DOWNLOAD",
        "max_rows": 200000,
        "page_size": 50000,
        "history_days": 30,
        "live_history_minutes": 10,
        "live_poll_seconds": 20,
        "app_token": None,
    },
    "realtime": {
        "mode": "ladot_live",  # ladot_live | simulated | snapshot
        "refresh_seconds": 30,
        "history_days": 1,
    },
}


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def _load_yaml_config() -> dict[str, Any]:
    root = get_project_root()
    cfg_path = root / "config" / "default.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_config() -> dict[str, Any]:
    cfg = deepcopy(_DEFAULT_CONFIG)
    yaml_cfg = _load_yaml_config()
    if yaml_cfg:
        _deep_update(cfg, yaml_cfg)
    return cfg


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent
