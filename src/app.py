"""FastAPI service exposing the energy-demand model.

Run locally:  uvicorn src.app:app --reload
Interactive docs: http://127.0.0.1:8000/docs
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config
from src.model import ModelBundle, explain_one, load_bundle, predict_one
from src.monitoring import IGNORED_PATHS, UNMATCHED, RequestStats, get_request_logger

logger = logging.getLogger("api")
request_log = get_request_logger()

EXAMPLE_REQUEST = {
    "timestamp": "2019-10-15T14:00:00",
    "T2M_toc": 30.1,
    "T2M_san": 29.4,
    "T2M_dav": 27.8,
    "QV2M_toc": 0.0189,
    "QV2M_san": 0.0185,
    "QV2M_dav": 0.0178,
    "TQL_toc": 0.05,
    "TQL_san": 0.08,
    "TQL_dav": 0.10,
    "W2M_toc": 12.0,
    "W2M_san": 6.5,
    "W2M_dav": 3.2,
    "holiday": 0,
    "school": 1,
}


def _weather(variable: str, description: str) -> float:
    lo, hi = config.RANGES[variable]
    return Field(
        ..., ge=lo, le=hi, allow_inf_nan=False, description=f"{description} (valid {lo} to {hi})"
    )


class PredictionRequest(BaseModel):
    """Raw inputs for one hour. Time features are derived from ``timestamp``."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    timestamp: datetime = Field(..., description="Hour to predict, Panama local time (ISO 8601)")

    T2M_toc: float = _weather("T2M", "Temperature at 2 m, Tocumen (deg C)")
    T2M_san: float = _weather("T2M", "Temperature at 2 m, Santiago (deg C)")
    T2M_dav: float = _weather("T2M", "Temperature at 2 m, David (deg C)")
    QV2M_toc: float = _weather("QV2M", "Humidity at 2 m, Tocumen (specific humidity, kg/kg)")
    QV2M_san: float = _weather("QV2M", "Humidity at 2 m, Santiago (specific humidity, kg/kg)")
    QV2M_dav: float = _weather("QV2M", "Humidity at 2 m, David (specific humidity, kg/kg)")
    TQL_toc: float = _weather("TQL", "Liquid precipitation, Tocumen (l/m2)")
    TQL_san: float = _weather("TQL", "Liquid precipitation, Santiago (l/m2)")
    TQL_dav: float = _weather("TQL", "Liquid precipitation, David (l/m2)")
    W2M_toc: float = _weather("W2M", "Wind speed at 2 m, Tocumen (m/s)")
    W2M_san: float = _weather("W2M", "Wind speed at 2 m, Santiago (m/s)")
    W2M_dav: float = _weather("W2M", "Wind speed at 2 m, David (m/s)")

    holiday: int = Field(..., ge=0, le=1, description="1 = public holiday, 0 = regular day")
    school: int = Field(..., ge=0, le=1, description="1 = school period, 0 = vacations")

    @field_validator("timestamp")
    @classmethod
    def drop_timezone(cls, value: datetime) -> datetime:
        """Treat all timestamps as Panama local time."""
        return value.replace(tzinfo=None)


class PredictionResponse(BaseModel):
    predicted_demand_mwh: float = Field(..., description="Predicted national load in MWh")
    unit: str = "MWh"
    model_type: str
    feature_set: str
    mlflow_run_id: str


class FeatureContribution(BaseModel):
    feature: str
    value: float = Field(..., description="Value of the feature used for this prediction")
    contribution_mwh: float = Field(
        ..., description="How much this feature moved the prediction away from the baseline (MWh)"
    )


class ExplanationResponse(BaseModel):
    """prediction = baseline + sum of all contributions (up to rounding)."""

    predicted_demand_mwh: float
    baseline_mwh: float = Field(..., description="Average prediction on typical training hours")
    unit: str = "MWh"
    method: str = Field(..., description="shapley_exact, or occlusion for very large models")
    contributions: list[FeatureContribution] = Field(..., description="Largest effect first")
    model_type: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup (the service still starts if it is missing)."""
    try:
        app.state.bundle = load_bundle()
        logger.info("Loaded model: %s", app.state.bundle.meta["model_type"])
    except FileNotFoundError as exc:
        app.state.bundle = None
        logger.error("%s", exc)
    yield


app = FastAPI(
    title="Energy Demand Forecasting API",
    description="Predicts hourly national electricity demand (MWh) for Panama "
    "from weather and calendar inputs.",
    version="2.0.0",
    lifespan=lifespan,
)
app.state.stats = RequestStats()


@app.middleware("http")
async def record_request_stats(request: Request, call_next):
    """Time every request, log one line and add it to the /stats counters."""
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - started
        route = request.scope.get("route")
        endpoint = route.path if route is not None else UNMATCHED
        if request.url.path not in IGNORED_PATHS:
            request.app.state.stats.record(endpoint, status_code, duration)
            request_log.info(
                "%s %s -> %s in %.1f ms",
                request.method,
                request.url.path,
                status_code,
                1000 * duration,
            )


def get_bundle(request: Request) -> ModelBundle:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return bundle


BundleDep = Annotated[ModelBundle, Depends(get_bundle)]


@app.get("/", tags=["info"])
def root() -> dict:
    return {
        "service": "energy-demand-forecasting",
        "docs": "/docs",
        "predict": "POST /predict",
        "explain": "POST /explain",
        "stats": "GET /stats",
    }


@app.get("/health", tags=["info"])
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": request.app.state.bundle is not None}


@app.get("/model-info", tags=["info"])
def model_info(bundle: BundleDep) -> dict:
    keys = (
        "model_type",
        "feature_set",
        "features",
        "params",
        "cv",
        "test",
        "mlflow_run_id",
        "trained_at_utc",
    )
    return {k: bundle.meta[k] for k in keys if k in bundle.meta}


@app.get("/stats", tags=["monitoring"])
def stats(request: Request) -> dict:
    """Request count and latency since the service started (health checks excluded)."""
    return request.app.state.stats.snapshot()


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PredictionRequest, bundle: BundleDep) -> PredictionResponse:
    """Predict national electricity demand for the given hour and conditions."""
    value = predict_one(bundle, payload.model_dump())
    return PredictionResponse(
        predicted_demand_mwh=round(value, 2),
        model_type=bundle.meta["model_type"],
        feature_set=bundle.meta["feature_set"],
        mlflow_run_id=bundle.meta["mlflow_run_id"],
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["prediction"])
def explain(payload: PredictionRequest, bundle: BundleDep) -> ExplanationResponse:
    """Predict demand and show how much each feature pushed it above or below typical demand."""
    try:
        result = explain_one(bundle, payload.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ExplanationResponse(
        predicted_demand_mwh=round(result["prediction_mwh"], 2),
        baseline_mwh=round(result["baseline_mwh"], 2),
        method=result["method"],
        contributions=[
            FeatureContribution(
                feature=c["feature"],
                value=c["value"],
                contribution_mwh=round(c["contribution_mwh"], 2),
            )
            for c in result["contributions"]
        ],
        model_type=bundle.meta["model_type"],
    )
