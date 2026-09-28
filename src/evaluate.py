"""Evaluation metrics and report figure generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    SAMPLE_PREDS_PATH,
)


def save_training_figures(
    engineered: pd.DataFrame,
    labeled: pd.DataFrame,
    y_test,
    test_pred,
    test_prob,
    importance: dict,
    test_df: pd.DataFrame,
    metrics_test: dict,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = labeled["will_fail_within_horizon"].value_counts().sort_index()
    ax.bar(
        ["No failure within 24h (0)", "Failure within 24h (1)"],
        counts.values,
        color=["#4C78A8", "#E45756"],
    )
    ax.set_ylabel("Row count")
    ax.set_title("Failure-class distribution (full labeled telemetry)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=140)
    plt.close(fig)

    # 2. Sensor trends (one machine with failures if possible)
    machine_id = int(
        labeled.loc[labeled["will_fail_within_horizon"] == 1, "machineID"].mode().iloc[0]
    )
    mdf = labeled[labeled["machineID"] == machine_id].sort_values("datetime").tail(24 * 60)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, col, color in zip(
        axes,
        ["volt", "rotate", "pressure", "vibration"],
        ["#4C78A8", "#F58518", "#54A24B", "#E45756"],
    ):
        ax.plot(mdf["datetime"], mdf[col], color=color, linewidth=0.8)
        ax.set_ylabel(col)
        fails = mdf[mdf["will_fail_within_horizon"] == 1]
        if len(fails):
            ax.scatter(
                fails["datetime"],
                fails[col],
                s=8,
                c="#E45756",
                alpha=0.35,
                label="pre-failure window",
            )
    axes[0].set_title(f"Sensor trends — machine {machine_id} (last ~60 days)")
    axes[-1].set_xlabel("datetime")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "sensor_trends.png", dpi=140)
    plt.close(fig)

    # 3. Rolling feature example
    eng_m = engineered[engineered["machineID"] == machine_id].tail(24 * 30)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eng_m["datetime"], eng_m["vibration_current"], label="vibration (current)", alpha=0.7)
    ax.plot(
        eng_m["datetime"],
        eng_m["vibration_roll_mean_12h"],
        label="vibration 12h rolling mean",
        linewidth=2,
    )
    ax.plot(
        eng_m["datetime"],
        eng_m["vibration_roll_mean_12h"] + eng_m["vibration_roll_std_12h"],
        label="mean + std",
        linestyle="--",
        alpha=0.7,
    )
    ax.legend()
    ax.set_title(f"12-hour rolling vibration features — machine {machine_id}")
    ax.set_xlabel("datetime")
    ax.set_ylabel("vibration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rolling_feature_example.png", dpi=140)
    plt.close(fig)

    # 4. Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["No fail (0)", "Fail@24h (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion matrix (chronological test set)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # 5. Precision / Recall / F1 bars
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Precision", "Recall", "F1", "Accuracy"]
    vals = [
        metrics_test["precision"],
        metrics_test["recall"],
        metrics_test["f1"],
        metrics_test["accuracy"],
    ]
    bars = ax.bar(names, vals, color=["#4C78A8", "#E45756", "#54A24B", "#B279A2"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-set metrics (failure class for P/R/F1)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_bars.png", dpi=140)
    plt.close(fig)

    # 6. Predictions vs actual (probability ordered sample)
    plot_n = min(800, len(test_prob))
    order = np.argsort(test_prob)[::-1][:plot_n]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(
        range(plot_n),
        np.asarray(y_test)[order],
        s=12,
        alpha=0.5,
        label="actual",
        c="#4C78A8",
    )
    ax.plot(range(plot_n), np.asarray(test_prob)[order], color="#E45756", label="P(fail)")
    ax.set_title(f"Top-{plot_n} highest predicted risk vs actual (test)")
    ax.set_xlabel("rank by predicted probability")
    ax.set_ylabel("label / probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "predictions_vs_actual.png", dpi=140)
    plt.close(fig)

    # 7. Feature importance
    top = list(importance.items())[:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([k for k, _ in top][::-1], [v for _, v in top][::-1], color="#4C78A8")
    ax.set_xlabel("Permutation importance (mean F1 drop)")
    ax.set_title("Top-15 feature importance (validation subsample)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=140)
    plt.close(fig)

    print(f"Figures written to {FIGURES_DIR}")


def evaluate_saved_model() -> dict:
    """Re-score the chronological test set from a saved model artifact."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run src/train.py first.")

    artifact = joblib.load(MODEL_PATH)
    from src.data_loader import load_labeled_dataset
    from src.feature_engineering import chronological_split, engineer_features, get_feature_matrix
    from src.config import PREDICTION_HORIZON_HOURS, ROLLING_WINDOW_HOURS, TRAIN_FRAC, VAL_FRAC

    labeled = load_labeled_dataset(horizon_hours=PREDICTION_HORIZON_HOURS)
    engineered = engineer_features(labeled, window_hours=ROLLING_WINDOW_HOURS)
    splits = chronological_split(engineered, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC)
    test_df = splits["test"]
    feature_cols = artifact["feature_cols"]
    X_test, y_test, _ = get_feature_matrix(test_df, feature_cols)
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "threshold": threshold,
        "n_test": int(len(y_test)),
    }
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    print(json.dumps(metrics, indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "datetime": test_df["datetime"].astype(str).values,
        "machineID": test_df["machineID"].values,
        "actual": y_test.values,
        "predicted": pred,
        "failure_probability": prob,
    }
    pd.DataFrame(out).head(200).to_csv(SAMPLE_PREDS_PATH, index=False)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            full = json.load(f)
        full["metrics_test_reeval"] = metrics
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(full, f, indent=2)

    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
