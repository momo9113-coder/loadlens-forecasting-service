from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import ForecastModel, load_model


_LOAD_DEFAULT = object()


class Observation(BaseModel):
    timestamp: datetime
    load: float
    temperature: float = 0.0
    humidity: float = 0.0
    wind_speed: float = 0.0
    general_diffuse_flow: float = 0.0
    diffuse_flow: float = 0.0


class ForecastRequest(BaseModel):
    history: list[Observation] = Field(min_length=30)
    horizon: int = Field(default=12, ge=1, le=144)


class ForecastPoint(BaseModel):
    timestamp: datetime
    forecast: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    forecasts: list[ForecastPoint]
    metrics: dict[str, float]


def create_app(model: ForecastModel | None | object = _LOAD_DEFAULT) -> FastAPI:
    app = FastAPI(title="LoadLens Forecasting Service", version="1.0.0")
    loaded_model: ForecastModel | None = model if isinstance(model, ForecastModel) else None
    load_default = model is _LOAD_DEFAULT
    load_attempted = not load_default

    def get_model() -> ForecastModel | None:
        nonlocal loaded_model, load_attempted
        if not load_attempted:
            load_attempted = True
            model_path = Path(os.getenv("LOADLENS_MODEL", "artifacts/model.joblib"))
            if model_path.exists():
                loaded_model = load_model(model_path)
        return loaded_model

    if load_default:
        model_path = Path(os.getenv("LOADLENS_MODEL", "artifacts/model.joblib"))
        app.state.model_path = str(model_path)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model_loaded": get_model() is not None}

    @app.get("/metrics")
    def metrics() -> dict[str, float]:
        active_model = get_model()
        if active_model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not loaded")
        return active_model.metrics

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        active_model = get_model()
        if active_model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not loaded")
        history = pd.DataFrame([observation.model_dump() for observation in request.history])
        try:
            result = active_model.forecast(history, request.horizon)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        forecasts = [ForecastPoint(**row) for row in result.to_dict(orient="records")]
        return ForecastResponse(forecasts=forecasts, metrics=active_model.metrics)

    return app


app = create_app()
