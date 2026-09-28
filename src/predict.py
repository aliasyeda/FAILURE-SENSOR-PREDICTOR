"""Inference helpers for the trained failure predictor."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH, ROLLING_WINDOW_HOURS, SENSOR_COLS  # noqa: E402
from src.feature_engineering import engineer_features  # noqa: E402


def load_artifact(model_path: Path | None = None) -> dict:
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train with: python src/train.py"
        )
    return joblib.load(path)


def predict_engineered_frame(df: pd.DataFrame, artifact: dict | None = None) -> pd.DataFrame:
    """
    Predict on a frame that already contains engineered feature columns
    (or enough history to engineer them).
    """
    artifact = artifact or load_artifact()
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    threshold = float(artifact["threshold"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # Attempt to engineer if raw sensors + datetime + machineID present
        need = {"datetime", "machineID", *SENSOR_COLS}
        if need.issubset(df.columns):
            eng = engineer_features(df, window_hours=artifact.get("rolling_window_hours", ROLLING_WINDOW_HOURS))
            work = eng
        else:
            raise ValueError(f"Missing feature columns: {missing[:8]}...")
    else:
        work = df

    X = work[feature_cols].astype(float)
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    out = work.copy()
    out["failure_probability"] = prob
    out["predicted"] = pred
    out["predicted_label"] = np.where(pred == 1, "failure_risk", "normal")
    return out


def predict_from_sensor_row(
    volt: float,
    rotate: float,
    pressure: float,
    vibration: float,
    age: float = 10.0,
    model_code: int = 0,
    error_count_24h: int = 0,
    artifact: dict | None = None,
) -> dict:
    """
    Single-row manual prediction.

    Without historical context, rolling statistics are approximated by the
    current reading (std≈0, diffs≈0). This is useful for demos; for production
    upload a time-series CSV so causal rolling features can be computed.
    """
    artifact = artifact or load_artifact()
    feature_cols = artifact["feature_cols"]
    threshold = float(artifact["threshold"])

    row = {}
    for col in SENSOR_COLS:
        val = {"volt": volt, "rotate": rotate, "pressure": pressure, "vibration": vibration}[col]
        row[f"{col}_current"] = val
        row[f"{col}_roll_mean_12h"] = val
        row[f"{col}_roll_std_12h"] = 0.0
        row[f"{col}_roll_min_12h"] = val
        row[f"{col}_roll_max_12h"] = val
        row[f"{col}_diff_1h"] = 0.0
        row[f"{col}_diff_12h"] = 0.0

    row["age"] = age
    row["model_code"] = model_code
    row["error_count_24h"] = error_count_24h

    X = pd.DataFrame([{c: row.get(c, 0.0) for c in feature_cols}])
    prob = float(artifact["model"].predict_proba(X)[0, 1])
    pred = int(prob >= threshold)
    importance = artifact.get("feature_importance", {})
    top_feats = list(importance.items())[:8]

    return {
        "failure_probability": prob,
        "predicted": pred,
        "predicted_label": "failure_risk" if pred == 1 else "normal",
        "threshold": threshold,
        "sensors": {
            "volt": volt,
            "rotate": rotate,
            "pressure": pressure,
            "vibration": vibration,
        },
        "approximation_note": (
            "Rolling features approximated from a single reading (no history). "
            "Upload a telemetry CSV for causal 12h features."
        ),
        "top_features": top_feats,
        "feature_values": {c: float(X.iloc[0][c]) for c in feature_cols},
    }


def predict_csv(path: str | Path, artifact: dict | None = None) -> pd.DataFrame:
    """Predict from a CSV with datetime, machineID, and sensor columns."""
    artifact = artifact or load_artifact()
    df = pd.read_csv(path, parse_dates=["datetime"] if "datetime" in pd.read_csv(path, nrows=0).columns else None)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return predict_engineered_frame(df, artifact)


if __name__ == "__main__":
    art = load_artifact()
    # Demo using mean-ish healthy-ish values from Azure PdM telemetry
    result = predict_from_sensor_row(
        volt=170.0,
        rotate=450.0,
        pressure=100.0,
        vibration=40.0,
        artifact=art,
    )
    print(result)
