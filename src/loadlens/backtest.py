from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import FeatureConfig, make_supervised_with_timestamps
from .model import fit_with_calibration


@dataclass(frozen=True)
class RollingOriginConfig:
    folds: int = 5
    initial_train_fraction: float = 0.5
    test_fraction: float = 0.1
    calibration_fraction: float = 0.125
    random_state: int = 20260715

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least two")
        if not 0 < self.initial_train_fraction < 1:
            raise ValueError("initial_train_fraction must be between zero and one")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        if self.initial_train_fraction + self.folds * self.test_fraction > 1.000001:
            raise ValueError("rolling-origin folds exceed the available observations")
        if not 0 < self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be between zero and 0.5")


def rolling_origin_backtest(
    frame: pd.DataFrame,
    feature_config: FeatureConfig = FeatureConfig(),
    backtest_config: RollingOriginConfig = RollingOriginConfig(),
) -> tuple[pd.DataFrame, dict[str, object]]:
    backtest_config.validate()
    features, target, target_timestamps = make_supervised_with_timestamps(frame, feature_config)
    total_rows = len(features)
    initial_train_rows = int(total_rows * backtest_config.initial_train_fraction)
    first_test_end = int(total_rows * (backtest_config.initial_train_fraction + backtest_config.test_fraction))
    if initial_train_rows < 125 or first_test_end <= initial_train_rows:
        raise ValueError("backtest configuration leaves too few train or test rows")

    fold_rows: list[dict[str, object]] = []
    all_actual: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_persistence: list[np.ndarray] = []
    all_covered: list[np.ndarray] = []
    all_widths: list[np.ndarray] = []

    for fold_index in range(backtest_config.folds):
        train_end = int(
            total_rows
            * (backtest_config.initial_train_fraction + fold_index * backtest_config.test_fraction)
        )
        test_end = int(
            total_rows
            * (backtest_config.initial_train_fraction + (fold_index + 1) * backtest_config.test_fraction)
        )
        if test_end > total_rows:
            raise ValueError("backtest fold exceeds available observations")

        train_features = features.iloc[:train_end]
        train_target = target.iloc[:train_end]
        test_features = features.iloc[train_end:test_end]
        actual = target.iloc[train_end:test_end].to_numpy()
        estimator, interval_width, calibration_rows = fit_with_calibration(
            train_features,
            train_target,
            calibration_fraction=backtest_config.calibration_fraction,
            random_state=backtest_config.random_state + fold_index,
        )
        predictions = estimator.predict(test_features)
        persistence = test_features["load_lag_0"].to_numpy()
        covered = (actual >= predictions - interval_width) & (actual <= predictions + interval_width)
        model_rmse = float(mean_squared_error(actual, predictions) ** 0.5)
        persistence_rmse = float(mean_squared_error(actual, persistence) ** 0.5)

        fold_rows.append({
            "fold": fold_index + 1,
            "train_rows": train_end,
            "calibration_rows": calibration_rows,
            "test_rows": len(actual),
            "test_start": pd.Timestamp(target_timestamps.iloc[train_end]).isoformat(),
            "test_end": pd.Timestamp(target_timestamps.iloc[test_end - 1]).isoformat(),
            "model_mae": float(mean_absolute_error(actual, predictions)),
            "model_rmse": model_rmse,
            "persistence_mae": float(mean_absolute_error(actual, persistence)),
            "persistence_rmse": persistence_rmse,
            "rmse_improvement_pct": float((persistence_rmse - model_rmse) / persistence_rmse * 100),
            "interval_width_p90_abs_residual": interval_width,
            "interval_coverage": float(np.mean(covered)),
        })
        all_actual.append(actual)
        all_predictions.append(predictions)
        all_persistence.append(persistence)
        all_covered.append(covered)
        all_widths.append(np.full(len(actual), interval_width))

    actual = np.concatenate(all_actual)
    predictions = np.concatenate(all_predictions)
    persistence = np.concatenate(all_persistence)
    covered = np.concatenate(all_covered)
    widths = np.concatenate(all_widths)
    model_rmse = float(mean_squared_error(actual, predictions) ** 0.5)
    persistence_rmse = float(mean_squared_error(actual, persistence) ** 0.5)
    folds = pd.DataFrame(fold_rows)
    summary: dict[str, object] = {
        "method": "expanding-window rolling-origin evaluation with a pre-test calibration block",
        "feature_horizon_steps": feature_config.horizon_steps,
        "folds": backtest_config.folds,
        "evaluated_rows": int(len(actual)),
        "evaluation_start": folds.iloc[0]["test_start"],
        "evaluation_end": folds.iloc[-1]["test_end"],
        "model_mae": float(mean_absolute_error(actual, predictions)),
        "model_rmse": model_rmse,
        "persistence_mae": float(mean_absolute_error(actual, persistence)),
        "persistence_rmse": persistence_rmse,
        "rmse_improvement_pct": float((persistence_rmse - model_rmse) / persistence_rmse * 100),
        "fold_rmse_wins": int((folds["model_rmse"] < folds["persistence_rmse"]).sum()),
        "mean_interval_width_p90_abs_residual": float(np.mean(widths)),
        "interval_coverage": float(np.mean(covered)),
        "config": asdict(backtest_config),
    }
    return folds, summary


def write_backtest_report(
    folds: pd.DataFrame,
    summary: dict[str, object],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_path = output_dir / "rolling_origin_folds.csv"
    summary_path = output_dir / "rolling_origin_summary.json"
    chart_path = output_dir / "rolling_origin_rmse.png"
    folds.to_csv(folds_path, index=False, float_format="%.6f")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(folds))
    width = 0.36
    axis.bar(x - width / 2, folds["model_rmse"], width, label="HistGradientBoosting", color="#2563eb")
    axis.bar(x + width / 2, folds["persistence_rmse"], width, label="Persistence", color="#6b7280")
    axis.set_xticks(x, [f"Fold {int(fold)}" for fold in folds["fold"]])
    axis.set_ylabel("RMSE")
    axis.set_title("Rolling-origin RMSE by fold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    return folds_path, summary_path, chart_path
