"""Shared project paths and constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Hourly telemetry → 12 samples = 12 hours; 24 samples = 24 hours
ROLLING_WINDOW_HOURS = 12
PREDICTION_HORIZON_HOURS = 24

SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]

MODEL_PATH = MODELS_DIR / "failure_predictor.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
SAMPLE_PREDS_PATH = REPORTS_DIR / "sample_predictions.csv"
DEMO_SAMPLES_PATH = DATA_DIR / "demo_samples.csv"

# Chronological split fractions of the time axis (not random)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (remainder)
