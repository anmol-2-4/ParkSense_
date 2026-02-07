"""Pydantic schemas for API request/response contracts."""
from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the prediction model is loaded")
    version: str = Field(default="0.1.0", description="API version")


class ZoneMetadata(BaseModel):
    zone_id: int
    lat_centroid: float
    lon_centroid: float
    capacity: int


class ZonesResponse(BaseModel):
    zones: list[ZoneMetadata]


class ConfidenceInterval(BaseModel):
    lower: float = Field(..., ge=0, le=1, description="Lower bound of 90% interval (occupancy rate)")
    upper: float = Field(..., ge=0, le=1, description="Upper bound of 90% interval (occupancy rate)")


class FreeSpacesInterval(BaseModel):
    lower: int = Field(..., ge=0)
    upper: int = Field(..., ge=0)


class PredictionResponse(BaseModel):
    zone_id: int
    timestamp: str = Field(..., description="ISO 8601 timestamp for which the prediction applies")
    occupancy_rate: float = Field(..., ge=0, le=1, description="Predicted occupancy rate (0–1)")
    free_spaces: int = Field(..., ge=0, description="Predicted number of free spaces")
    capacity: int = Field(..., ge=0)
    confidence_interval_90: ConfidenceInterval
    free_spaces_interval: FreeSpacesInterval
    summary: str = Field(..., description="Plain-language interpretation for commuters")
    latest_observed_occupancy: float | None = Field(
        None,
        description="Most recent observed occupancy rate for this zone when using real-time/latest data",
    )


class PredictAllResponse(BaseModel):
    predictions: list[PredictionResponse]


class MetricsResponse(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    n_train: int | None = None
    n_val: int | None = None
    baseline_mae: float | None = None
    improvement_vs_baseline_pct: float | None = None


class ErrorDetail(BaseModel):
    detail: str
    code: str | None = None
