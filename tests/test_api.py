from fastapi.testclient import TestClient

from loadlens.api import create_app
from loadlens.features import FeatureConfig
from loadlens.model import ForecastModel
from test_features import fixture_frame


def test_api_contract() -> None:
    frame = fixture_frame(180)
    model = ForecastModel.fit(frame, config=FeatureConfig(steps_per_day=12, lags=(0, 1, 3, 12), rolling_windows=(3, 12)))
    client = TestClient(create_app(model))
    response = client.post("/forecast", json={
        "history": frame.tail(30).assign(timestamp=lambda x: x["timestamp"].astype(str)).to_dict(orient="records"),
        "horizon": 3,
    })
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecasts"]) == 3
    assert "validation_rmse" in body["metrics"]


def test_health_does_not_require_model() -> None:
    client = TestClient(create_app(None))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
