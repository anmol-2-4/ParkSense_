"""Training pipeline: load data, build features, train LightGBM with quantile regression."""
import json
import logging
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

from ..config import get_config, get_project_root
from ..ingest.synthetic import run_synthetic_pipeline
from ..ingest.ladot import run_ladot_pipeline, LADOTConfig
from ..zones.aggregate import load_and_aggregate
from ..features.build import build_features, FEATURE_COLS


def train_model(
    data_dir: Path | None = None,
    model_dir: Path | None = None,
    split_weeks: int = 4,
    seed: int = 42,
    generate_if_missing: bool = True,
) -> dict:
    """Train point predictor and quantile models. Returns metrics dict."""
    root = get_project_root()
    cfg = get_config()
    if data_dir is None:
        data_dir = root / cfg["data"]["raw_dir"]
    if model_dir is None:
        model_dir = root / cfg["data"]["model_dir"]
    processed_dir = root / cfg["data"]["processed_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we have data
    source = cfg["data"].get("source", "synthetic")
    inventory_path = data_dir / "inventory.csv"
    occupancy_path = (root / cfg["data"]["processed_dir"]) / "occupancy.parquet"
    needs_download = not inventory_path.exists() or not occupancy_path.exists()

    if source == "ladot" and inventory_path.exists():
        try:
            sample = pd.read_csv(inventory_path, nrows=5)
            if "space_id" in sample.columns and sample["space_id"].astype(str).str.startswith("SP_").any():
                needs_download = True
                logger.info("Existing inventory looks synthetic; re-downloading LADOT data.")
        except Exception:
            needs_download = True

    if source == "ladot" and occupancy_path.exists():
        try:
            occ_sample = pd.read_parquet(occupancy_path, columns=["space_id"]).head(5)
            if occ_sample["space_id"].astype(str).str.startswith("SP_").any():
                needs_download = True
                logger.info("Existing occupancy looks synthetic; re-downloading LADOT data.")
        except Exception:
            needs_download = True
    if source == "synthetic" and inventory_path.exists():
        try:
            sample = pd.read_csv(inventory_path, nrows=5)
            if "space_id" not in sample.columns or not sample["space_id"].astype(str).str.startswith("SP_").any():
                needs_download = True
                logger.info("Existing inventory looks non-synthetic; regenerating synthetic data.")
        except Exception:
            needs_download = True

    if source == "synthetic" and occupancy_path.exists():
        try:
            occ_sample = pd.read_parquet(occupancy_path, columns=["space_id"]).head(5)
            if not occ_sample["space_id"].astype(str).str.startswith("SP_").any():
                needs_download = True
                logger.info("Existing occupancy looks non-synthetic; regenerating synthetic data.")
        except Exception:
            needs_download = True

    if needs_download and generate_if_missing:
        if source == "ladot":
            logger.info("Downloading LADOT open data (Los Angeles)")
            ladot_cfg = LADOTConfig(**cfg.get("ladot", {}))
            run_ladot_pipeline(
                raw_dir=str(cfg["data"]["raw_dir"]),
                processed_dir=str(cfg["data"]["processed_dir"]),
                n_zones=cfg["n_zones"],
                seed=seed,
                history_days=int(cfg.get("ladot", {}).get("history_days", 30)),
                cfg=ladot_cfg,
            )
        else:
            logger.info("Generating synthetic data (no inventory found)")
            run_synthetic_pipeline(
                raw_dir=str(cfg["data"]["raw_dir"]),
                processed_dir=str(cfg["data"]["processed_dir"]),
                n_zones=cfg["n_zones"],
                days_historical=90,
                bucket_minutes=cfg["time_bucket_minutes"],
                seed=seed,
            )
        data_dir = root / cfg["data"]["raw_dir"]
        processed_dir = root / cfg["data"]["processed_dir"]

    zone_occ, zone_meta = load_and_aggregate(data_dir, processed_dir)
    zone_occ["timestamp"] = pd.to_datetime(zone_occ["timestamp"])
    if zone_occ.empty:
        raise ValueError(
            "No occupancy data available for training. Ensure LADOT archive/live history download succeeded."
        )

    # Time-based split
    cutoff = zone_occ["timestamp"].max() - pd.Timedelta(weeks=split_weeks)
    train_df = zone_occ[zone_occ["timestamp"] <= cutoff]
    val_df = zone_occ[zone_occ["timestamp"] > cutoff]

    lags = cfg["features"]["lags"]
    same_hour_prev_week = cfg["features"]["same_hour_prev_week"]
    min_per_zone = zone_occ.groupby("zone_id").size().min()
    if min_per_zone < (max(lags) + 1):
        logger.warning(
            "Not enough history per zone for lags=%s (min rows=%d). Reducing lags for training.",
            lags,
            min_per_zone,
        )
        if min_per_zone >= 2:
            lags = [1]
            same_hour_prev_week = False
        else:
            raise ValueError("Insufficient live history for training. Increase live_history_minutes.")

    feat_train = build_features(
        train_df,
        lags=lags,
        same_hour_prev_week=same_hour_prev_week,
        traffic_proxy=cfg["features"]["traffic_proxy"],
        events_proxy=cfg["features"]["events_proxy"],
    )
    feat_val = build_features(
        val_df,
        lags=lags,
        same_hour_prev_week=same_hour_prev_week,
        traffic_proxy=cfg["features"]["traffic_proxy"],
        events_proxy=cfg["features"]["events_proxy"],
    )

    # Ensure we only use columns that exist
    use_cols = [c for c in FEATURE_COLS if c in feat_train.columns]
    X_train = feat_train[use_cols]
    y_train = feat_train["occupancy_rate"]
    X_val = feat_val[use_cols]
    y_val = feat_val["occupancy_rate"]

    # Zone id for later
    zone_train = feat_train["zone_id"]
    zone_val = feat_val["zone_id"]

    params = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": cfg["model"]["num_leaves"],
        "learning_rate": cfg["model"]["learning_rate"],
        "n_estimators": cfg["model"]["n_estimators"],
        "random_state": seed,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])

    pred_val = np.clip(model.predict(X_val), 0, 1)
    mae = mean_absolute_error(y_val, pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred_val))

    # Quantile models for confidence intervals
    quantile_models = {}
    for alpha in cfg["model"].get("quantile_alpha", [0.1, 0.5, 0.9]):
        qparams = {**params, "objective": "quantile", "alpha": alpha}
        qmodel = lgb.LGBMRegressor(**{k: v for k, v in qparams.items() if k != "metric"})
        qmodel.set_params(objective="quantile", alpha=alpha)
        qmodel.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(20, verbose=False)])
        quantile_models[alpha] = qmodel

    # Persist
    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / "quantile_models.pkl", "wb") as f:
        pickle.dump(quantile_models, f)
    with open(model_dir / "feature_cols.json", "w") as f:
        json.dump(use_cols, f, indent=2)
    zone_meta.to_csv(model_dir / "zone_metadata.csv", index=False)

    # Persist latest occupancy per zone for real-time inference (lags + same_hour_prev_week)
    zone_occ_sorted = zone_occ.sort_values(["zone_id", "timestamp"])
    latest_rows = []
    for zid in zone_occ_sorted["zone_id"].unique():
        sub = zone_occ_sorted[zone_occ_sorted["zone_id"] == zid].tail(4)
        rates = sub["occupancy_rate"].tolist()
        while len(rates) < 4:
            rates.insert(0, 0.5)
        # lag_1 = most recent, lag_4 = oldest of the four
        lag_1 = rates[-1] if rates else 0.5
        lag_2 = rates[-2] if len(rates) >= 2 else 0.5
        lag_3 = rates[-3] if len(rates) >= 3 else 0.5
        lag_4 = rates[-4] if len(rates) >= 4 else 0.5
        most_recent_ts = sub["timestamp"].iloc[-1]
        prev_week_ts = most_recent_ts - pd.Timedelta(days=7)
        prev_week = zone_occ_sorted[(zone_occ_sorted["zone_id"] == zid) & (zone_occ_sorted["timestamp"] <= prev_week_ts)].tail(1)
        same_hour_prev_week = float(prev_week["occupancy_rate"].iloc[0]) if len(prev_week) else 0.5
        latest_rows.append({"zone_id": zid, "lag_1": lag_1, "lag_2": lag_2, "lag_3": lag_3, "lag_4": lag_4, "same_hour_prev_week": same_hour_prev_week})
    pd.DataFrame(latest_rows).to_csv(model_dir / "latest_zone_occupancy.csv", index=False)
    logger.info("Saved latest zone occupancy for real-time inference (%d zones)", len(latest_rows))

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "n_train": len(X_train),
        "n_val": len(X_val),
    }

    # Baseline: same hour previous week
    if "same_hour_prev_week" in feat_val.columns:
        baseline_mae = mean_absolute_error(y_val, feat_val["same_hour_prev_week"].fillna(0.5))
        metrics["baseline_mae"] = round(baseline_mae, 4)
        metrics["improvement_vs_baseline_pct"] = round((1 - mae / baseline_mae) * 100, 1)

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Training complete: MAE=%.4f, improvement vs baseline=%.1f%%", mae, metrics.get("improvement_vs_baseline_pct", 0))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    m = train_model()
    print("Metrics:", m)
