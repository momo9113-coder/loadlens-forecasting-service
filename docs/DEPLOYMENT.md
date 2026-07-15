# Deployment And Operations

Last verified: 2026-07-15.

## Public Services

- GitHub repository: `https://github.com/momo9113-coder/loadlens-forecasting-service`
- Streamlit application: `https://loadlens-forecasting-service-xx64wapp6mrljcncsedvaah.streamlit.app/`
- Streamlit source: repository `main`, file `app/streamlit_app.py`
- CI: `.github/workflows/test.yml`

## Authentication Model

The repository contains no cloud key.

- GitHub pushes use the developer's Git credential helper or GitHub CLI outside the repository.
- Streamlit Community Cloud is linked to GitHub through browser OAuth.
- The app has no `.streamlit/secrets.toml` dependency and no configured application secret.
- Never copy OAuth tokens, browser cookies, device codes, or credential-manager output into this repository.

## Deployment Behavior

Pushing to `main` updates the source repository and runs CI. Streamlit Community Cloud watches the linked branch and redeploys the configured app. The deployment installs `requirements.txt`, which installs the local package and development extra.

On a cold start, the app:

1. checks for an ignored local model and processed CSV;
2. downloads the public UCI zip when those files do not exist;
3. normalizes the data and trains the first model;
4. caches the model and dataframe for the running process.

The filesystem is ephemeral. A container restart can repeat this process.

## Free-Tier Behavior

Streamlit Community Cloud currently has no fixed free-trial expiry, but an app without traffic for 12 hours hibernates. Any viewer can wake it from the sleeping page. Do not use automated traffic to defeat hibernation. Keep screenshots and quantitative results in the README so the repository remains reviewable during a cold start or outage.

## Verification

Local checks:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn loadlens.api:app --host 127.0.0.1 --port 8765
```

Remote checks:

1. Open the public Streamlit URL in a signed-out/private browser.
2. Confirm the page title is `LoadLens` and the chart, forecast table, and metrics render.
3. Confirm the latest GitHub Actions `tests` run succeeded.
4. Recheck after dependency or Python-version changes.

## Recovery

- If CI fails, fix tests before redeploying.
- If Streamlit build fails, inspect Community Cloud logs; first verify Python and package compatibility.
- If the app starts but cannot load data, verify the UCI URL and checksum metadata path.
- If a dependency upgrade breaks deployment, pin the last verified versions and redeploy a reviewed commit.
- Roll back by reverting the breaking Git commit; do not upload secrets or raw data as a shortcut.
