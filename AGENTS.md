# LoadLens Project Handoff

Last updated: 2026-07-15.

## Scope

This repository is the end-to-end applied-product project in a three-repository MSc application portfolio. Keep it focused on reproducible load forecasting, evaluation, API behavior, and the public Streamlit experience. Do not turn it into a general portfolio monorepo.

## Read First

1. `README.md`
2. `AGENTS.md`
3. `docs/DEPLOYMENT.md`
4. `src/loadlens/data.py`
5. `src/loadlens/features.py`
6. `src/loadlens/model.py`
7. `src/loadlens/api.py`
8. `app/streamlit_app.py`
9. `tests/`

## Current State

- GitHub: `https://github.com/momo9113-coder/loadlens-forecasting-service`
- Live app: `https://loadlens-forecasting-service-xx64wapp6mrljcncsedvaah.streamlit.app/`
- Stable release: `https://github.com/momo9113-coder/loadlens-forecasting-service/releases/tag/v1.0.0`
- Default branch: `main`
- Dataset: UCI Power Consumption of Tetouan City, 52,416 normalized rows.
- Primary rolling-origin result: pooled RMSE 456.17 versus persistence RMSE 612.91 across 26,136 test rows; 25.57% lower RMSE and four wins in five folds.
- Independently calibrated empirical bands covered 89.08% in aggregate; fold 1 under-covered at 76.43% and lost to persistence by 5.96% RMSE.
- Final dated holdout: model RMSE 371.24 versus persistence RMSE 550.07; calibrated p90 half-width 518.21 and 89.53% untouched-holdout coverage.
- Local tests: 7 passing as of 2026-07-15.
- GitHub Actions `tests`: passing on the current public `main` branch.

Treat these as application evidence, not production claims. Read `docs/BACKTEST.md` before quoting a metric; do not omit the losing first fold when discussing stability.

## Architecture

```text
UCI zip -> normalized CSV -> leakage-aware features -> sklearn model
                                                   |-> FastAPI
                                                   `-> Streamlit
```

- `data.py` downloads, hashes, extracts, and normalizes the public snapshot.
- `features.py` creates lag, rolling, calendar, and observed-weather features.
- `model.py` performs chronological evaluation, interval construction, persistence comparison, serialization, and recursive forecasts.
- `api.py` exposes health, metrics, and forecast contracts. Default model loading is lazy so tests do not depend on local artifacts.
- `streamlit_app.py` downloads/trains on a cold start when no ignored local artifact is available.

## Local Workflow

Run commands from the repository root.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe scripts/train.py
.\.venv\Scripts\python.exe -m uvicorn loadlens.api:app --reload
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

If `.venv` does not exist, create it and install `requirements-dev.txt`. Python 3.12 is used in CI; the existing Windows development environment was created with Python 3.14.3.

## Data And Secrets

- UCI download requires no API key.
- No Streamlit secret is required by the current app.
- `data/raw/`, `data/processed/`, `artifacts/`, `.venv/`, caches, and `.streamlit/secrets.toml` are ignored.
- Never commit a token, browser cookie, credential file, private dataset, or local model artifact without an explicit size/license review.
- GitHub and Streamlit authentication are managed outside the repository. See `docs/DEPLOYMENT.md`; do not add credential values to it.

## Guardrails

- Do not use a random train/test split for this time-series task.
- Do not call the empirical residual band a guaranteed probabilistic interval.
- Do not claim general grid-operations performance from this single public dataset.
- Keep API and app logic thin; forecasting logic belongs in importable modules with tests.
- Before changing a published metric, reproduce the dated data snapshot and evaluation protocol.

## Next Priorities

1. Validate recursive forecasts by horizon rather than extrapolating one-step results.
2. Investigate fold 1 regime sensitivity and horizon-specific interval calibration.
3. Re-verify the pinned environment when upgrading Python or direct dependencies.
4. Keep the README screenshot synchronized after material UI changes.
5. Keep `v1.0.0` stable; create a new release only after a material, re-verified change.
