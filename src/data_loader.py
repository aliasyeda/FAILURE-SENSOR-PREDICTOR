"""Load and prepare the Microsoft Azure Predictive Maintenance dataset."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PREDICTION_HORIZON_HOURS,
    PROJECT_ROOT,
    RAW_DIR,
    SENSOR_COLS,
)


REQUIRED_FILES = [
    "PdM_telemetry.csv",
    "PdM_failures.csv",
    "PdM_machines.csv",
    "PdM_errors.csv",
]


def ensure_raw_data() -> Path:
    """Download raw CSVs if missing."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in REQUIRED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        script = PROJECT_ROOT / "scripts" / "download_data.py"
        print(f"Missing {missing}; running {script} ...")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Dataset download failed. Run: python scripts/download_data.py"
            )
    return RAW_DIR


def load_raw_tables() -> dict[str, pd.DataFrame]:
    """Load telemetry, failures, machines, and errors."""
    ensure_raw_data()
    telemetry = pd.read_csv(RAW_DIR / "PdM_telemetry.csv", parse_dates=["datetime"])
    failures = pd.read_csv(RAW_DIR / "PdM_failures.csv", parse_dates=["datetime"])
    machines = pd.read_csv(RAW_DIR / "PdM_machines.csv")
    errors = pd.read_csv(RAW_DIR / "PdM_errors.csv", parse_dates=["datetime"])

    telemetry = telemetry.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    failures = failures.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    errors = errors.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    return {
        "telemetry": telemetry,
        "failures": failures,
        "machines": machines,
        "errors": errors,
    }


def build_failure_labels(
    telemetry: pd.DataFrame,
    failures: pd.DataFrame,
    horizon_hours: int = PREDICTION_HORIZON_HOURS,
) -> pd.DataFrame:
    """
    Label each telemetry row: will_fail_within_horizon = 1 if a failure
    occurs for the same machine in (T, T + horizon_hours].

    Implementation: for each failure at time F, mark hours in
    (F - horizon, F) as positive. Uses only future failure times as labels,
    never as input features.
    """
    tel = telemetry.copy()
    tel["will_fail_within_horizon"] = 0
    horizon = pd.Timedelta(hours=horizon_hours)

    for _, row in failures.iterrows():
        mid = int(row["machineID"])
        fail_t = row["datetime"]
        mask = (
            (tel["machineID"] == mid)
            & (tel["datetime"] > fail_t - horizon)
            & (tel["datetime"] < fail_t)
        )
        tel.loc[mask, "will_fail_within_horizon"] = 1

    return tel


def attach_machine_meta(
    telemetry: pd.DataFrame, machines: pd.DataFrame
) -> pd.DataFrame:
    df = telemetry.merge(machines, on="machineID", how="left")
    if "model" in df.columns:
        df["model_code"] = pd.Categorical(df["model"]).codes.astype(np.int16)
    return df


def count_recent_errors(
    telemetry: pd.DataFrame,
    errors: pd.DataFrame,
    window_hours: int = 24,
) -> pd.DataFrame:
    """
    Count errors in the past window_hours including T (causal rolling sum).
    """
    tel = telemetry.copy()
    err_counts = (
        errors.groupby(["machineID", "datetime"], as_index=False)
        .size()
        .rename(columns={"size": "errors_at_hour"})
    )
    tel = tel.merge(err_counts, on=["machineID", "datetime"], how="left")
    tel["errors_at_hour"] = tel["errors_at_hour"].fillna(0).astype(np.int16)
    tel = tel.sort_values(["machineID", "datetime"])
    tel["error_count_24h"] = (
        tel.groupby("machineID", sort=False)["errors_at_hour"]
        .transform(lambda s: s.rolling(window=window_hours, min_periods=1).sum())
        .astype(np.int16)
    )
    return tel.drop(columns=["errors_at_hour"])


def load_labeled_dataset(
    horizon_hours: int = PREDICTION_HORIZON_HOURS,
    include_errors: bool = True,
) -> pd.DataFrame:
    """Full labeled telemetry frame ready for feature engineering."""
    tables = load_raw_tables()
    df = build_failure_labels(
        tables["telemetry"], tables["failures"], horizon_hours=horizon_hours
    )
    df = attach_machine_meta(df, tables["machines"])
    if include_errors:
        df = count_recent_errors(df, tables["errors"], window_hours=24)

    keep = (
        ["datetime", "machineID"]
        + [c for c in SENSOR_COLS if c in df.columns]
        + [c for c in ["age", "model_code", "error_count_24h"] if c in df.columns]
        + ["will_fail_within_horizon"]
    )
    df = df[keep].dropna().reset_index(drop=True)
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_machines": int(df["machineID"].nunique()),
        "datetime_min": str(df["datetime"].min()),
        "datetime_max": str(df["datetime"].max()),
        "positive_rate": float(df["will_fail_within_horizon"].mean()),
        "n_positives": int(df["will_fail_within_horizon"].sum()),
        "sensors": [c for c in SENSOR_COLS if c in df.columns],
        "horizon_hours": PREDICTION_HORIZON_HOURS,
    }


if __name__ == "__main__":
    data = load_labeled_dataset()
    print(dataset_summary(data))
    print(data.head())
