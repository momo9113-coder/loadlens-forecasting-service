import numpy as np
import pandas as pd

from loadlens.features import FeatureConfig, build_feature_row, make_supervised


def fixture_frame(rows: int = 80) -> pd.DataFrame:
    timestamp = pd.date_range("2026-01-01", periods=rows, freq="10min")
    return pd.DataFrame({
        "timestamp": timestamp,
        "load": 100 + np.sin(np.arange(rows) / 4) * 10,
        "temperature": np.linspace(15, 20, rows),
        "humidity": 60.0,
        "wind_speed": 2.0,
        "general_diffuse_flow": 4.0,
        "diffuse_flow": 2.0,
    })


def test_supervised_features_have_no_future_target_values() -> None:
    frame = fixture_frame()
    config = FeatureConfig(steps_per_day=12, lags=(0, 1, 3, 12), rolling_windows=(3, 12))
    features, target = make_supervised(frame, config)
    assert len(features) == len(target)
    assert "load_lag_0" in features
    assert features["load_lag_0"].iloc[-1] != target.iloc[-1]
    assert features.notna().all().all()


def test_feature_row_requires_history_for_longest_lag() -> None:
    config = FeatureConfig(steps_per_day=12, lags=(0, 1, 12), rolling_windows=(3, 12))
    row = build_feature_row(fixture_frame(20), config)
    assert list(row.columns)
    assert row.shape == (1, len(row.columns))
