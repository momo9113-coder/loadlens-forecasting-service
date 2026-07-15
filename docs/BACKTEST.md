# Rolling-Origin Evaluation

Evaluation date: 2026-07-15.

## Question

Does the one-step HistGradientBoosting model consistently improve on a last-observation persistence forecast when the prediction origin advances through time?

## Protocol

- Source: 52,416 normalized UCI observations at 10-minute intervals, 2017-01-01 through 2017-12-30.
- Target: Zone 1 load one step (10 minutes) ahead.
- Split: five non-overlapping test folds after an initial 50% training window; each fold contributes roughly 10% of the supervised rows.
- Training: expanding window, re-fitted independently at every fold.
- Interval calibration: the last 12.5% of each fold's available training history calibrates the p90 absolute-residual half-width. The point model is then re-fitted on all pre-test history.
- Leakage control: no test target is used for feature construction, fitting, model selection, or interval-width calibration.
- Baseline: persistence, equal to the most recently observed load.

The exact configuration is versioned in `reports/rolling_origin_summary.json`; fold-level outputs are in `reports/rolling_origin_folds.csv`.

## Pooled Result

Across 26,136 test observations from 2017-07-02 12:00 through 2017-12-30 23:50:

| Metric | HistGradientBoosting | Persistence |
|---|---:|---:|
| MAE | 288.62 | 411.99 |
| RMSE | 456.17 | 612.91 |

The pooled RMSE reduction was 25.57%. The model beat persistence on RMSE in four of five folds. Fold-specific calibrated bands covered 89.08% of test targets in aggregate, compared with a nominal p90 absolute-residual target; this is empirical coverage, not a distribution-free or operational guarantee.

![Model and persistence RMSE by fold](../reports/rolling_origin_rmse.png)

## Fold Stability

| Fold | Test period | Model RMSE | Persistence RMSE | RMSE change | Interval coverage |
|---:|---|---:|---:|---:|---:|
| 1 | 2017-07-02 to 2017-08-07 | 710.11 | 670.18 | -5.96% | 76.43% |
| 2 | 2017-08-07 to 2017-09-13 | 410.40 | 665.92 | +38.37% | 96.81% |
| 3 | 2017-09-13 to 2017-10-19 | 360.13 | 616.91 | +41.62% | 89.94% |
| 4 | 2017-10-19 to 2017-11-24 | 399.05 | 575.63 | +30.68% | 87.28% |
| 5 | 2017-11-24 to 2017-12-30 | 280.81 | 523.27 | +46.34% | 94.95% |

Positive change means lower model RMSE than persistence. Fold 1 is a genuine failure case: the model lagged persistence by 5.96% and the interval under-covered. This indicates that aggregate performance masks seasonal or regime sensitivity. It motivates drift monitoring, more robust calibration, and testing forecast horizons beyond one step before operational use.

## Reproduce

```powershell
python scripts/download_data.py
python scripts/backtest.py
python -m pytest
```
