from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loadlens.data import download_and_prepare, load_processed  # noqa: E402
from loadlens.model import ForecastModel, load_model  # noqa: E402


st.set_page_config(page_title="LoadLens", layout="wide")
st.title("LoadLens")
st.caption("Short-horizon electricity load forecasting with a reproducible baseline and empirical interval.")


@st.cache_resource(show_spinner=False)
def get_model() -> tuple[ForecastModel, pd.DataFrame]:
    artifact = ROOT / "artifacts" / "model.joblib"
    processed = ROOT / "data" / "processed" / "tetouan_load.csv"
    if artifact.exists() and processed.exists():
        return load_model(artifact), load_processed(processed)
    processed = download_and_prepare(ROOT / "data")
    frame = load_processed(processed)
    model = ForecastModel.fit(frame)
    return model, frame


try:
    with st.spinner("Loading the public dataset and model..."):
        model, frame = get_model()
except Exception as exc:  # pragma: no cover - the cloud surface must show an actionable message
    st.error(f"The demo could not initialize: {exc}")
    st.stop()

history_points = st.slider("History shown", min_value=145, max_value=min(1000, len(frame)), value=min(432, len(frame)))
horizon = st.slider("Forecast steps (10-minute intervals)", min_value=1, max_value=144, value=12)
history = frame.tail(history_points).copy()

st.subheader("Recent load")
st.line_chart(history.set_index("timestamp")["load"])

forecast = model.forecast(history, horizon)
left, right = st.columns(2)
with left:
    st.subheader("Forecast")
    st.dataframe(forecast, use_container_width=True, hide_index=True)
with right:
    st.subheader("Holdout metrics")
    st.json(model.metrics)

st.caption("The interval is an empirical residual band. It is not a calibrated operational guarantee.")
