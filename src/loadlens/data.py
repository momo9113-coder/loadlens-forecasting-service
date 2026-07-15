from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


DATASET_URL = "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip"
DEFAULT_DATA_DIR = Path("data")
DEFAULT_PROCESSED_NAME = "tetouan_load.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(raw_dir: Path, url: str = DATASET_URL) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "tetouan_power_consumption.zip"
    if not archive_path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "LoadLens/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    return archive_path


def extract_archive(archive_path: Path, raw_dir: Path) -> Path:
    extract_dir = raw_dir / "tetouan"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_dir)
    csv_files = sorted(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {archive_path}")
    return csv_files[0]


def _normalise_name(value: object) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _find_column(columns: list[str], *names: str) -> str | None:
    normalised = {_normalise_name(column): column for column in columns}
    for name in names:
        match = normalised.get(_normalise_name(name))
        if match:
            return match
    return None


def load_tetouan_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    timestamp_col = _find_column(raw.columns.tolist(), "DateTime", "Date Time", "timestamp")
    target_col = _find_column(raw.columns.tolist(), "Zone 1 Power Consumption")
    if timestamp_col is None or target_col is None:
        raise ValueError(f"Unexpected Tetouan columns: {list(raw.columns)}")

    aliases = {
        "temperature": ("Temperature",),
        "humidity": ("Humidity",),
        "wind_speed": ("Wind Speed",),
        "general_diffuse_flow": ("general diffuse flows", "general diffuse flow"),
        "diffuse_flow": ("diffuse flows", "diffuse flow"),
    }
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[timestamp_col], errors="coerce"),
        "load": pd.to_numeric(raw[target_col], errors="coerce"),
    })
    for output_name, candidates in aliases.items():
        source = _find_column(raw.columns.tolist(), *candidates)
        frame[output_name] = pd.to_numeric(raw[source], errors="coerce") if source else 0.0

    frame = (
        frame.dropna(subset=["timestamp", "load"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )
    numeric = [column for column in frame.columns if column not in {"timestamp", "load"}]
    frame[numeric] = frame[numeric].interpolate(limit_direction="both").fillna(0.0)
    if len(frame) < 100:
        raise ValueError("The dataset is unexpectedly small after cleaning")
    return frame


def download_and_prepare(data_dir: Path = DEFAULT_DATA_DIR, url: str = DATASET_URL) -> Path:
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    archive_path = download_archive(raw_dir, url=url)
    csv_path = extract_archive(archive_path, raw_dir)
    frame = load_tetouan_csv(csv_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / DEFAULT_PROCESSED_NAME
    frame.to_csv(processed_path, index=False)
    metadata = {
        "source_url": url,
        "archive_sha256": sha256_file(archive_path),
        "source_csv": str(csv_path.relative_to(raw_dir)),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
    }
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return processed_path


def load_processed(path: Path = DEFAULT_DATA_DIR / "processed" / DEFAULT_PROCESSED_NAME) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "load"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")
    return frame.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize the LoadLens public dataset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--url", default=DATASET_URL)
    args = parser.parse_args()
    output = download_and_prepare(args.data_dir, url=args.url)
    print(output)


if __name__ == "__main__":
    main()
