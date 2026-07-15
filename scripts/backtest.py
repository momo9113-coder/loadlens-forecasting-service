from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loadlens.backtest import RollingOriginConfig, rolling_origin_backtest, write_backtest_report  # noqa: E402
from loadlens.data import DEFAULT_DATA_DIR, download_and_prepare, load_processed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LoadLens rolling-origin backtest.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    processed = args.data_dir / "processed" / "tetouan_load.csv"
    if not processed.exists():
        processed = download_and_prepare(args.data_dir)
    folds, summary = rolling_origin_backtest(
        load_processed(processed),
        backtest_config=RollingOriginConfig(folds=args.folds),
    )
    folds_path, summary_path, chart_path = write_backtest_report(folds, summary, args.output_dir)
    print(folds.to_string(index=False))
    print(json_summary(summary))
    print(f"folds={folds_path}")
    print(f"summary={summary_path}")
    print(f"chart={chart_path}")


def json_summary(summary: dict[str, object]) -> str:
    keys = (
        "evaluated_rows",
        "model_mae",
        "model_rmse",
        "persistence_mae",
        "persistence_rmse",
        "rmse_improvement_pct",
        "fold_rmse_wins",
        "interval_coverage",
    )
    return "\n".join(f"{key}={summary[key]}" for key in keys)


if __name__ == "__main__":
    main()
