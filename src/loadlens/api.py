from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import ForecastModel, load_model


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


def create_app(model: ForecastModel | None = None) -> FastAPI:
    app = FastAPI(title="LoadLens Forecasting Service", version="0.1.0")
    loaded_model = model
    if loaded_model is None:
        model_path = Path(os.getenv("LOADLENS_MODEL", "artifacts/model.joblib"))
        if model_path.exists():
            loaded_model = load_model(model_path)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "model_loaded": loaded_model is not None}

    @app.get("/metrics")
    def metrics() -> dict[str, float]:
        if loaded_model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not loaded")
        return loaded_model.metrics

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(request: ForecastRequest) -> ForecastResponse:
        if loaded_model is None:
            raise HTTPException(status_code=503, detail="Model artifact is not loaded")
        history = pd.DataFrame([observation.model_dump() for observation in request.history])
        try:
            result = loaded_model.forecast(history, request.horizon)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        forecasts = [ForecastPoint(**row) for row in result.to_dict(orient="records")]
        return ForecastResponse(forecasts=forecasts, metrics=loaded_model.metrics)

    return app


app = create_app()
