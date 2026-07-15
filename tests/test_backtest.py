from pathlib import Path

from loadlens.backtest import RollingOriginConfig, rolling_origin_backtest, write_backtest_report
from loadlens.features import FeatureConfig
from test_features import fixture_frame


def test_rolling_origin_backtest_is_ordered_and_writes_evidence(tmp_path: Path) -> None:
    frame = fixture_frame(420)
    feature_config = FeatureConfig(steps_per_day=12, lags=(0, 1, 3, 12), rolling_windows=(3, 12))
    backtest_config = RollingOriginConfig(folds=3, initial_train_fraction=0.4, test_fraction=0.2)
    folds, summary = rolling_origin_backtest(frame, feature_config, backtest_config)

    assert len(folds) == 3
    assert folds["test_start"].is_monotonic_increasing
    assert (folds["model_rmse"] > 0).all()
    assert summary["evaluated_rows"] == int(folds["test_rows"].sum())
    assert 0 <= summary["interval_coverage"] <= 1

    folds_path, summary_path, chart_path = write_backtest_report(folds, summary, tmp_path)
    assert folds_path.exists()
    assert summary_path.exists()
    assert chart_path.exists()


def test_rolling_origin_backtest_does_not_absorb_unused_tail() -> None:
    frame = fixture_frame(420)
    feature_config = FeatureConfig(steps_per_day=12, lags=(0, 1, 3, 12), rolling_windows=(3, 12))
    backtest_config = RollingOriginConfig(folds=2, initial_train_fraction=0.4, test_fraction=0.2)
    folds, summary = rolling_origin_backtest(frame, feature_config, backtest_config)

    assert summary["evaluated_rows"] == int(folds["test_rows"].sum())
    assert summary["evaluated_rows"] < int(len(frame) * 0.5)
