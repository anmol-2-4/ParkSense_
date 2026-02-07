"""FastAPI application for parking availability predictions.

Serves zone-wise predictions with confidence intervals and a commuter UI.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from threading import Event, Thread

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..config import get_config, get_project_root
from ..model.predict import load_predictor, ParkingPredictor
from ..model.live import LiveOccupancyProvider, SimulatedLiveFeed
from ..zones.aggregate import load_and_aggregate
from ..ingest.ladot import LADOTConfig, LADOTLiveUpdater
from .schemas import (
    HealthResponse,
    ZonesResponse,
    ZoneMetadata,
    PredictionResponse,
    PredictAllResponse,
    MetricsResponse,
    ErrorDetail,
)

logger = logging.getLogger(__name__)
uvicorn_logger = logging.getLogger("uvicorn.error")
API_VERSION = "0.1.0"

predictor: ParkingPredictor | None = None
live_provider: LiveOccupancyProvider | None = None
_live_thread: Thread | None = None
_live_stop: Event | None = None
_live_status = {
    "mode": "snapshot",
    "last_update": None,
    "last_error": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, live_provider, _live_thread, _live_stop
    root = get_project_root()
    cfg = get_config()
    model_dir = root / cfg["data"]["model_dir"]
    if (model_dir / "model.pkl").exists():
        try:
            predictor = load_predictor(model_dir)
            logger.info("Model loaded from %s", model_dir)
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            predictor = None
    else:
        logger.warning("No model artifact at %s; run training first", model_dir / "model.pkl")
        predictor = None
    latest_path = model_dir / "latest_zone_occupancy.csv"
    realtime_cfg = cfg.get("realtime", {})
    mode = realtime_cfg.get("mode", "snapshot")
    _live_status["mode"] = mode
    uvicorn_logger.info("Realtime mode: %s", mode)

    if mode == "ladot_live":
        latest_path = model_dir / "latest_zone_occupancy.csv"
        live_provider = LiveOccupancyProvider(latest_path)
        data_dir = root / cfg["data"]["raw_dir"]
        try:
            inventory = pd.read_csv(data_dir / "inventory.csv")
            ladot_cfg = LADOTConfig(**cfg.get("ladot", {}))
            updater = LADOTLiveUpdater(
                inventory=inventory,
                cfg=ladot_cfg,
                n_zones=cfg["n_zones"],
            )
        except Exception as e:
            logger.exception("Failed to start LADOT live feed: %s", e)
            live_provider = LiveOccupancyProvider(latest_path) if latest_path.exists() else None
        else:
            refresh_seconds = max(5, int(realtime_cfg.get("refresh_seconds", 30)))
            _live_stop = Event()

            def _run_feed():
                uvicorn_logger.info("LADOT live feed thread running")
                while _live_stop and not _live_stop.is_set():
                    try:
                        df = updater.next_snapshot()
                        if not df.empty:
                            live_provider.update(df)
                            _live_status["last_update"] = datetime.utcnow().isoformat()
                            _live_status["last_error"] = None
                            uvicorn_logger.info("LADOT live update: %d zones", len(df))
                    except Exception as e:
                        _live_status["last_error"] = str(e)
                        uvicorn_logger.exception("LADOT live feed update failed: %s", e)
                    _live_stop.wait(refresh_seconds)

            _live_thread = Thread(target=_run_feed, daemon=True)
            _live_thread.start()
            logger.info("LADOT live feed started (refresh=%ss)", refresh_seconds)
    elif mode == "simulated":
        live_provider = LiveOccupancyProvider(latest_path)
        data_dir = root / cfg["data"]["raw_dir"]
        processed_dir = root / cfg["data"]["processed_dir"]
        try:
            zone_occ, _ = load_and_aggregate(data_dir, processed_dir)
            feed = SimulatedLiveFeed(
                zone_occ=zone_occ,
                bucket_minutes=cfg["time_bucket_minutes"],
                history_days=int(realtime_cfg.get("history_days", 1)),
            )
        except Exception as e:
            logger.exception("Failed to start simulated live feed: %s", e)
            live_provider = LiveOccupancyProvider(latest_path) if latest_path.exists() else None
        else:
            refresh_seconds = max(1, int(realtime_cfg.get("refresh_seconds", 10)))
            _live_stop = Event()

            def _run_feed():
                while _live_stop and not _live_stop.is_set():
                    df, _ = feed.next_snapshot()
                    if not df.empty:
                        live_provider.update(df)
                    _live_stop.wait(refresh_seconds)

            _live_thread = Thread(target=_run_feed, daemon=True)
            _live_thread.start()
            logger.info("Simulated live feed started (refresh=%ss)", refresh_seconds)
    else:
        live_provider = LiveOccupancyProvider(latest_path) if latest_path.exists() else None

    if live_provider and live_provider.available:
        uvicorn_logger.info("Real-time/latest occupancy data loaded for inference")
    yield
    if _live_stop:
        _live_stop.set()
    predictor = None
    live_provider = None
    _live_thread = None
    _live_stop = None
    _live_status["last_update"] = None
    _live_status["last_error"] = None


app = FastAPI(
    title="Parking Availability Prediction API",
    description=(
        "Zone-wise parking availability predictions with confidence intervals. "
        "Designed for smart-city deployment and commuter applications."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        503: {"model": ErrorDetail, "description": "Model not loaded or service unavailable"},
        400: {"model": ErrorDetail, "description": "Invalid request"},
    },
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)


def _require_model() -> ParkingPredictor:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded. Run the training pipeline first.",
        )
    return predictor


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
def health() -> HealthResponse:
    """Service health and model availability."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        version=API_VERSION,
    )


@app.get(
    "/zones",
    response_model=ZonesResponse,
    summary="List zones",
    tags=["Zones"],
)
def list_zones() -> ZonesResponse:
    """Return all zones with centroid coordinates and capacity."""
    _require_model()
    zones = [
        ZoneMetadata(
            zone_id=int(z["zone_id"]),
            lat_centroid=float(z["lat_centroid"]),
            lon_centroid=float(z["lon_centroid"]),
            capacity=int(z["capacity"]),
        )
        for z in predictor.get_zone_metadata()
    ]
    return ZonesResponse(zones=zones)


@app.get(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict one zone",
    tags=["Predictions"],
)
def predict(
    zone_id: int = Query(..., ge=0, description="Zone identifier"),
    timestamp: datetime | None = Query(None, description="Time for prediction (UTC); default: now"),
) -> PredictionResponse:
    """Predict occupancy and free spaces for a single zone with 90% confidence interval.
    Uses latest observed occupancy as model inputs when available (real-time data)."""
    p = _require_model()
    ts = timestamp or datetime.utcnow()
    live = live_provider.get_latest(zone_id) if live_provider else None
    try:
        result = p.predict(
            zone_id=zone_id,
            timestamp=ts,
            last_known_occupancy=live["last_known_occupancy"] if live else None,
            same_hour_prev_week=live["same_hour_prev_week"] if live else None,
        )
        if live_provider:
            result["latest_observed_occupancy"] = live_provider.get_latest_observed_rate(zone_id)
        return PredictionResponse(**result)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get(
    "/predict/all",
    response_model=PredictAllResponse,
    summary="Predict all zones",
    tags=["Predictions"],
)
def predict_all(
    timestamp: datetime | None = Query(None, description="Time for prediction (UTC); default: now"),
) -> PredictAllResponse:
    """Predict availability for every zone (e.g. for map or dashboard).
    Uses latest observed occupancy per zone when available (real-time data)."""
    p = _require_model()
    ts = timestamp or datetime.utcnow()
    zone_ids = [int(z["zone_id"]) for z in p.get_zone_metadata()]
    predictions = []
    for zid in zone_ids:
        live = live_provider.get_latest(zid) if live_provider else None
        r = p.predict(
            zone_id=zid,
            timestamp=ts,
            last_known_occupancy=live["last_known_occupancy"] if live else None,
            same_hour_prev_week=live["same_hour_prev_week"] if live else None,
        )
        if live_provider:
            r["latest_observed_occupancy"] = live_provider.get_latest_observed_rate(zid)
        predictions.append(PredictionResponse(**r))
    return PredictAllResponse(predictions=predictions)


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Training metrics",
    tags=["System"],
)
def metrics() -> MetricsResponse:
    """Model training metrics and improvement over baseline."""
    _require_model()
    if not predictor.metrics:
        return MetricsResponse()
    return MetricsResponse(**predictor.metrics)


@app.get(
    "/realtime/status",
    summary="Real-time feed status",
    tags=["System"],
)
def realtime_status() -> dict:
    """Expose the real-time feed status for debugging/demo."""
    return {
        "mode": _live_status["mode"],
        "last_update": _live_status["last_update"],
        "last_error": _live_status["last_error"],
        "live_provider_available": live_provider.available if live_provider else False,
    }


# Static UI
ui_path = get_project_root() / "static"
if ui_path.exists():
    app.mount("/static", StaticFiles(directory=str(ui_path)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(ui_path / "index.html")
