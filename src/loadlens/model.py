from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import DEFAULT_DATA_DIR, download_and_prepare, load_processed
from .features import FeatureConfig, build_feature_row, make_supervised


def make_estimator(random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=random_state,
    )


def fit_with_calibration(
    features: pd.DataFrame,
    target: pd.Series,
    calibration_fraction: float,
    random_state: int,
) -> tuple[HistGradientBoostingRegressor, float, int]:
    if not 0 < calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must be between zero and 0.5")
    calibration_start = int(len(features) * (1 - calibration_fraction))
    if calibration_start < 100 or calibration_start >= len(features):
        raise ValueError("calibration_fraction leaves too few training or calibration rows")

    calibration_model = make_estimator(random_state)
    calibration_model.fit(features.iloc[:calibration_start], target.iloc[:calibration_start])
    calibration_predictions = calibration_model.predict(features.iloc[calibration_start:])
    calibration_actual = target.iloc[calibration_start:].to_numpy()
    interval_width = float(np.quantile(np.abs(calibration_actual - calibration_predictions), 0.9))

    # Refit the point model on every observation available before the test origin.
    estimator = make_estimator(random_state)
    estimator.fit(features, target)
    return estimator, interval_width, len(calibration_actual)


@dataclass
class ForecastModel:
    estimator: Any
    config: FeatureConfig
    feature_columns: list[str]
    interval_width: float
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        config: FeatureConfig = FeatureConfig(),
        validation_fraction: float = 0.2,
        calibration_fraction: float = 0.125,
        random_state: int = 20260715,
    ) -> "ForecastModel":
        features, target = make_supervised(frame, config)
        if len(features) < 100:
            raise ValueError("At least 100 supervised rows are required")
        split = int(len(features) * (1 - validation_fraction))
        if split <= 0 or split >= len(features):
            raise ValueError("validation_fraction leaves no train or validation rows")
        estimator, interval_width, calibration_rows = fit_with_calibration(
            features.iloc[:split],
            target.iloc[:split],
            calibration_fraction=calibration_fraction,
            random_state=random_state,
        )
        predictions = estimator.predict(features.iloc[split:])
        actual = target.iloc[split:].to_numpy()
        persistence = features.iloc[split:]["load_lag_0"].to_numpy()
        model_metrics = {
            "calibration_rows": float(calibration_rows),
            "validation_rows": float(len(actual)),
            "validation_mae": float(mean_absolute_error(actual, predictions)),
            "validation_rmse": float(mean_squared_error(actual, predictions) ** 0.5),
            "persistence_mae": float(mean_absolute_error(actual, persistence)),
            "persistence_rmse": float(mean_squared_error(actual, persistence) ** 0.5),
            "interval_width_p90_abs_residual": interval_width,
            "interval_coverage": float(np.mean((actual >= predictions - interval_width) & (actual <= predictions + interval_width))),
        }
        return cls(
            estimator=estimator,
            config=config,
            feature_columns=list(features.columns),
            interval_width=interval_width,
            metrics=model_metrics,
        )

    def predict_next(self, history: pd.DataFrame) -> float:
        row = build_feature_row(history, self.config)
        row = row.reindex(columns=self.feature_columns, fill_value=0.0)
        return float(self.estimator.predict(row)[0])

    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        working = history.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
        working = working.sort_values("timestamp").reset_index(drop=True)
        if working.empty:
            raise ValueError("history must contain at least one observation")
        if len(working) <= max(max(self.config.lags), max(self.config.rolling_windows)):
            raise ValueError("history is too short for the configured features")
        delta = working["timestamp"].diff().dropna().median()
        if pd.isna(delta) or delta <= pd.Timedelta(0):
            delta = pd.Timedelta(minutes=10)
        rows: list[dict[str, object]] = []
        for _ in range(horizon):
            value = self.predict_next(working)
            timestamp = working["timestamp"].iloc[-1] + delta
            last = working.iloc[-1]
            next_row = {
                "timestamp": timestamp,
                "load": value,
                "temperature": float(last.get("temperature", 0.0)),
                "humidity": float(last.get("humidity", 0.0)),
                "wind_speed": float(last.get("wind_speed", 0.0)),
                "general_diffuse_flow": float(last.get("general_diffuse_flow", 0.0)),
                "diffuse_flow": float(last.get("diffuse_flow", 0.0)),
            }
            rows.append({
                "timestamp": timestamp,
                "forecast": value,
                "lower": value - self.interval_width,
                "upper": value + self.interval_width,
            })
            working = pd.concat([working, pd.DataFrame([next_row])], ignore_index=True)
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_model(path: Path) -> ForecastModel:
    model = joblib.load(path)
    if not isinstance(model, ForecastModel):
        raise TypeError(f"Unexpected model artifact type: {type(model)!r}")
    return model


def train_from_disk(data_dir: Path = DEFAULT_DATA_DIR, output: Path = Path("artifacts/model.joblib")) -> ForecastModel:
    processed = data_dir / "processed" / "tetouan_load.csv"
    if not processed.exists():
        processed = download_and_prepare(data_dir)
    model = ForecastModel.fit(load_processed(processed))
    model.save(output)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the LoadLens forecasting model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.joblib"))
    args = parser.parse_args()
    model = train_from_disk(args.data_dir, args.output)
    for key, value in model.metrics.items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
