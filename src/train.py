"""Train the failure prediction model with a chronological split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

# Allow running as `python src/train.py` from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREDICTION_HORIZON_HOURS,
    REPORTS_DIR,
    ROLLING_WINDOW_HOURS,
    TRAIN_FRAC,
    VAL_FRAC,
)
from src.data_loader import dataset_summary, load_labeled_dataset  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    chronological_split,
    engineer_features,
    get_feature_matrix,
)
from src.evaluate import save_training_figures  # noqa: E402


def build_model() -> HistGradientBoostingClassifier:
    """
    HistGradientBoosting: strong tabular model, handles large hourly panels,
    no external XGBoost dependency (Render-friendly).
    Class imbalance handled via sample_weight (fitted after split only).
    """
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        min_samples_leaf=50,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=42,
    )


def metrics_dict(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "support_positive": int((np.asarray(y_true) == 1).sum()),
        "support_negative": int((np.asarray(y_true) == 0).sum()),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def train(
    save_artifacts: bool = True,
    threshold: float | None = None,
) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading labeled dataset...")
    labeled = load_labeled_dataset(horizon_hours=PREDICTION_HORIZON_HOURS)
    summary = dataset_summary(labeled)
    print("Dataset:", summary)

    print("Engineering features (12h causal rolling)...")
    engineered = engineer_features(labeled, window_hours=ROLLING_WINDOW_HOURS)
    feature_cols = engineered.attrs["feature_cols"]
    print(f"n_features={len(feature_cols)}, n_rows={len(engineered)}")

    splits = chronological_split(engineered, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    print(
        f"Split (chronological): train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )
    print(f"  train_end={splits['train_end']}, val_end={splits['val_end']}")

    X_train, y_train, _ = get_feature_matrix(train_df, feature_cols)
    X_val, y_val, _ = get_feature_matrix(val_df, feature_cols)
    X_test, y_test, _ = get_feature_matrix(test_df, feature_cols)

    # Balance weights computed on TRAIN only (no leakage from val/test)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    model = build_model()
    print("Training HistGradientBoostingClassifier...")
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Choose decision threshold on validation to balance precision/recall
    # Prefer higher recall for failure class without collapsing precision.
    val_prob = model.predict_proba(X_val)[:, 1]
    if threshold is None:
        best_t, best_score = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 25):
            pred = (val_prob >= t).astype(int)
            rec = recall_score(y_val, pred, zero_division=0)
            prec = precision_score(y_val, pred, zero_division=0)
            f1 = f1_score(y_val, pred, zero_division=0)
            # Emphasize recall slightly for predictive maintenance
            score = 0.4 * prec + 0.6 * rec
            if f1 > 0 and score > best_score:
                best_score, best_t = score, float(t)
        threshold = best_t
    print(f"Selected decision threshold (from val): {threshold:.3f}")

    def eval_split(name, X, y):
        prob = model.predict_proba(X)[:, 1]
        pred = (prob >= threshold).astype(int)
        m = metrics_dict(y, pred, prob)
        print(f"\n=== {name} ===")
        print(classification_report(y, pred, digits=4, zero_division=0))
        return m, pred, prob

    train_m, _, _ = eval_split("TRAIN", X_train, y_train)
    val_m, _, _ = eval_split("VAL", X_val, y_val)
    test_m, test_pred, test_prob = eval_split("TEST", X_test, y_test)

    # Feature importance via permutation on a val subsample (stored for app)
    from sklearn.inspection import permutation_importance

    rng = np.random.RandomState(42)
    n_imp = min(5000, len(X_val))
    idx = rng.choice(len(X_val), size=n_imp, replace=False)
    print("Computing permutation importance on validation subsample...")
    imp = permutation_importance(
        model,
        X_val.iloc[idx],
        y_val.iloc[idx],
        n_repeats=5,
        random_state=42,
        scoring="f1",
        n_jobs=-1,
    )
    importance = {
        feature_cols[i]: float(imp.importances_mean[i]) for i in range(len(feature_cols))
    }
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "threshold": float(threshold),
        "horizon_hours": PREDICTION_HORIZON_HOURS,
        "rolling_window_hours": ROLLING_WINDOW_HOURS,
        "train_end": str(splits["train_end"]),
        "val_end": str(splits["val_end"]),
        "feature_importance": importance,
        "sensor_cols": ["volt", "rotate", "pressure", "vibration"],
        "model_name": "HistGradientBoostingClassifier",
    }

    report = {
        "dataset": summary,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "split": {
            "strategy": "chronological_global_datetime",
            "train_frac": TRAIN_FRAC,
            "val_frac": VAL_FRAC,
            "test_frac": round(1.0 - TRAIN_FRAC - VAL_FRAC, 2),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_end": str(splits["train_end"]),
            "val_end": str(splits["val_end"]),
        },
        "model": artifact["model_name"],
        "threshold": float(threshold),
        "horizon_hours": PREDICTION_HORIZON_HOURS,
        "rolling_window_hours": ROLLING_WINDOW_HOURS,
        "metrics_train": train_m,
        "metrics_val": val_m,
        "metrics_test": test_m,
        "feature_importance_top15": dict(list(importance.items())[:15]),
        "target_definition": (
            f"will_fail_within_horizon=1 if a component failure occurs for the "
            f"same machine in (T, T+{PREDICTION_HORIZON_HOURS}h]."
        ),
    }

    if save_artifacts:
        joblib.dump(artifact, MODEL_PATH)
        print(f"Saved model -> {MODEL_PATH}")

        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved metrics -> {METRICS_PATH}")

        # Sample predictions on test set
        sample = test_df[
            ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"]
        ].copy()
        sample["actual"] = y_test.values
        sample["predicted"] = test_pred
        sample["failure_probability"] = test_prob
        sample_path = REPORTS_DIR / "sample_predictions.csv"
        # Stratified-ish sample: some positives + some negatives
        pos = sample[sample["actual"] == 1].head(50)
        neg = sample[sample["actual"] == 0].head(50)
        pd.concat([pos, neg]).to_csv(sample_path, index=False)
        print(f"Saved sample predictions -> {sample_path}")

        # Demo samples for Streamlit (engineered feature rows)
        demo = test_df.copy()
        demo["predicted"] = test_pred
        demo["failure_probability"] = test_prob
        demo_pos = demo[demo["will_fail_within_horizon"] == 1].head(10)
        demo_neg = demo[demo["will_fail_within_horizon"] == 0].head(10)
        demo_out = pd.concat([demo_pos, demo_neg])
        demo_cols = (
            ["datetime", "machineID"]
            + artifact["sensor_cols"]
            + feature_cols
            + ["will_fail_within_horizon", "predicted", "failure_probability"]
        )
        demo_out[demo_cols].to_csv(ROOT / "data" / "demo_samples.csv", index=False)

        save_training_figures(
            engineered=engineered,
            labeled=labeled,
            y_test=y_test,
            test_pred=test_pred,
            test_prob=test_prob,
            importance=importance,
            test_df=test_df,
            metrics_test=test_m,
        )

    return report


if __name__ == "__main__":
    train()
