# Backtest evidence

Run the expanding-window evaluation from the repository root:

```powershell
python scripts/backtest.py
```

`rolling_origin_folds.csv` contains fold-level metrics. `rolling_origin_summary.json` contains metrics pooled across all test observations, and `rolling_origin_rmse.png` visualizes model-versus-baseline stability. Each fold uses only earlier observations for model fitting and a pre-test calibration block for the empirical interval; test observations are never used to select the model or interval half-width.
