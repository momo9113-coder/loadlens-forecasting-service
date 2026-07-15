# LoadLens Forecasting Service

LoadLens is a reproducible short-horizon electricity-load forecasting service. It turns a public smart-city power-consumption dataset into a small, testable product with a FastAPI service and a Streamlit demonstration.

The project is intentionally scoped to one target zone and a one-step recursive forecast. The model is compared with a persistence baseline, evaluated with a chronological holdout, and served with an empirical prediction interval. It is not a claim of operational grid forecasting performance.

## Public demo

The intended public entry point is Streamlit Community Cloud:

`https://<deployment-name>.streamlit.app`

The URL will be added after the first public deployment. The application is stateless and may retrain on first start when no model artifact is present.

## Data

The default source is the UCI `Tetouan City power consumption` dataset:

<https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city>

The download script records the archive SHA-256 and writes the normalized data under `data/processed/`. Raw data and trained artifacts are ignored by Git. Tests use generated in-memory fixtures and never require network access.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/download_data.py
python scripts/train.py
python -m uvicorn loadlens.api:app --reload
streamlit run app/streamlit_app.py
```

The API exposes:

- `GET /health` - process and model status;
- `GET /metrics` - holdout metrics saved with the model;
- `POST /forecast` - recursive forecasts from a short observation history.

Run tests with:

```powershell
python -m pytest
```

## Evaluation design

- Chronological holdout only; random splits are not used.
- Features use observations available at the forecast origin: current/lagged load, calendar cycles, and observed weather fields.
- Reported metrics are MAE, RMSE, persistence-baseline comparison, and empirical interval coverage.
- The interval is an empirical residual band, not an unconditional probability guarantee.

## First reproducible run (2026-07-15)

Using the 52,416-row UCI snapshot from 2017-01-01 through 2017-12-30, with the last 20% held out chronologically:

| Metric | HistGradientBoosting | Persistence baseline |
|---|---:|---:|
| MAE | 249.46 | 369.43 |
| RMSE | 371.24 | 550.07 |

The empirical p90 absolute-residual interval had width 532.78 and 89.995% holdout coverage. These numbers describe this dated snapshot and configuration; they are not a leaderboard claim or a production guarantee.

## Repository layout

```text
app/                    Streamlit public demo
src/loadlens/           data, features, model, and API modules
scripts/                download and training entry points
tests/                  unit and API contract tests
data/                   ignored raw/processed data directories
artifacts/              ignored model files
```

## Limitations

The free public deployment may sleep after inactivity and has no persistent local disk. The demo therefore avoids a database and can rebuild its model from the public snapshot. Weather forecasts are not modeled; the last observed weather values are carried forward during recursive inference. A production system would need online monitoring, drift checks, access control, and a stronger uncertainty calibration protocol.

## License

Code is released under the MIT license. The dataset remains subject to the UCI dataset terms; it is not redistributed in this repository.
