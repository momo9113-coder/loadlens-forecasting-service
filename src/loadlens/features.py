from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    steps_per_day: int = 144
    horizon_steps: int = 1
    lags: tuple[int, ...] = (0, 1, 6, 24, 144)
    rolling_windows: tuple[int, ...] = (6, 24, 144)


def _base_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "load"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["load"] = pd.to_numeric(result["load"], errors="coerce")
    result = result.dropna(subset=["timestamp", "load"]).sort_values("timestamp").reset_index(drop=True)
    for column in ["temperature", "humidity", "wind_speed", "general_diffuse_flow", "diffuse_flow"]:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").interpolate(limit_direction="both").fillna(0.0)
    return result


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    minutes = result["timestamp"].dt.hour * 60 + result["timestamp"].dt.minute
    day_fraction = minutes / (24 * 60)
    week_fraction = (result["timestamp"].dt.dayofweek * 24 * 60 + minutes) / (7 * 24 * 60)
    result["day_sin"] = np.sin(2 * np.pi * day_fraction)
    result["day_cos"] = np.cos(2 * np.pi * day_fraction)
    result["week_sin"] = np.sin(2 * np.pi * week_fraction)
    result["week_cos"] = np.cos(2 * np.pi * week_fraction)
    result["is_weekend"] = (result["timestamp"].dt.dayofweek >= 5).astype(float)
    return result


def make_supervised(frame: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> tuple[pd.DataFrame, pd.Series]:
    result = add_calendar_features(_base_frame(frame))
    feature_columns: list[str] = []
    for lag in config.lags:
        name = f"load_lag_{lag}"
        result[name] = result["load"].shift(lag)
        feature_columns.append(name)
    for window in config.rolling_windows:
        name_mean = f"load_roll_mean_{window}"
        name_std = f"load_roll_std_{window}"
        result[name_mean] = result["load"].rolling(window, min_periods=window).mean()
        result[name_std] = result["load"].rolling(window, min_periods=window).std().fillna(0.0)
        feature_columns.extend([name_mean, name_std])

    observed_columns = [
        "temperature",
        "humidity",
        "wind_speed",
        "general_diffuse_flow",
        "diffuse_flow",
        "day_sin",
        "day_cos",
        "week_sin",
        "week_cos",
        "is_weekend",
    ]
    feature_columns.extend(observed_columns)
    result["target"] = result["load"].shift(-config.horizon_steps)
    usable = result.dropna(subset=feature_columns + ["target"]).reset_index(drop=True)
    return usable[feature_columns], usable["target"]


def build_feature_row(history: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    result = _base_frame(history)
    if result.empty:
        raise ValueError("At least one observation is required")
    result = add_calendar_features(result)
    row: dict[str, float] = {}
    for lag in config.lags:
        if lag >= len(result):
            raise ValueError(f"At least {lag + 1} observations are required for lag {lag}")
        row[f"load_lag_{lag}"] = float(result["load"].iloc[-1 - lag])
    for window in config.rolling_windows:
        if len(result) < window:
            raise ValueError(f"At least {window} observations are required for rolling features")
        values = result["load"].iloc[-window:]
        row[f"load_roll_mean_{window}"] = float(values.mean())
        row[f"load_roll_std_{window}"] = float(values.std(ddof=1) if len(values) > 1 else 0.0)
    last = result.iloc[-1]
    minutes = last["timestamp"].hour * 60 + last["timestamp"].minute
    day_fraction = minutes / (24 * 60)
    week_fraction = (last["timestamp"].dayofweek * 24 * 60 + minutes) / (7 * 24 * 60)
    row.update({
        "temperature": float(last["temperature"]),
        "humidity": float(last["humidity"]),
        "wind_speed": float(last["wind_speed"]),
        "general_diffuse_flow": float(last["general_diffuse_flow"]),
        "diffuse_flow": float(last["diffuse_flow"]),
        "day_sin": float(np.sin(2 * np.pi * day_fraction)),
        "day_cos": float(np.cos(2 * np.pi * day_fraction)),
        "week_sin": float(np.sin(2 * np.pi * week_fraction)),
        "week_cos": float(np.cos(2 * np.pi * week_fraction)),
        "is_weekend": float(last["timestamp"].dayofweek >= 5),
    })
    return pd.DataFrame([row])
