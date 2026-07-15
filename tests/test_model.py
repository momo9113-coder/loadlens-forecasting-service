import numpy as np

from loadlens.features import FeatureConfig
from loadlens.model import ForecastModel
from test_features import fixture_frame


def test_model_forecast_returns_interval() -> None:
    frame = fixture_frame(180)
    config = FeatureConfig(steps_per_day=12, lags=(0, 1, 3, 12), rolling_windows=(3, 12))
    model = ForecastModel.fit(frame, config=config)
    result = model.forecast(frame.tail(30), horizon=4)
    assert len(result) == 4
    assert np.all(result["lower"] <= result["forecast"])
    assert np.all(result["forecast"] <= result["upper"])
    assert model.metrics["validation_rows"] > 0
