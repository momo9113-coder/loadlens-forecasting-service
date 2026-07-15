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
- Default branch: `main`
- Dataset: UCI Power Consumption of Tetouan City, 52,416 normalized rows.
- First dated holdout result: model RMSE 371.24 versus persistence RMSE 550.07.
- Empirical p90 absolute-residual band: width 532.78, holdout coverage 89.995%.
- Local tests: 5 passing as of 2026-07-15.
- GitHub Actions `tests`: passing on the current public `main` branch.

Treat these as first-run evidence, not final or production claims. The current evaluation is one chronological holdout, not a complete rolling-origin study.

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

If `.venv` does not exist, create it and install `.[dev]`. Python 3.12 is used in CI; the existing Windows development environment was created with Python 3.14.3.

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

1. Replace the single holdout with rolling-origin evaluation and record fold-level metrics.
2. Pin or lock deployment dependencies so future package upgrades cannot silently break the app.
3. Add a real screenshot or short GIF to the README as a fallback for Streamlit hibernation.
4. Improve interval calibration and document coverage by forecast horizon.
5. Tag an application-ready release only after the above evidence is stable.
