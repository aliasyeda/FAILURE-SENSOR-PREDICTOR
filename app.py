"""
Failure Sensor Predictor — Streamlit dashboard
AVIP 2026 Task 8: Predictive maintenance from industrial IoT telemetry.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "failure_predictor.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"
DEMO_PATH = ROOT / "data" / "demo_samples.csv"
SAMPLE_PREDS = ROOT / "reports" / "sample_predictions.csv"

st.set_page_config(
    page_title="Failure Sensor Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_demo():
    if DEMO_PATH.exists():
        return pd.read_csv(DEMO_PATH, parse_dates=["datetime"])
    if SAMPLE_PREDS.exists():
        return pd.read_csv(SAMPLE_PREDS, parse_dates=["datetime"])
    return None


def show_figure(name: str, caption: str | None = None):
    path = FIGURES_DIR / name
    if path.exists():
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        st.info(f"Figure not found: {name}")


def main():
    artifact = load_model()
    metrics = load_metrics()
    demo = load_demo()

    st.sidebar.title("Failure Sensor Predictor")
    page = st.sidebar.radio(
        "Navigate",
        ["Dashboard", "Predict", "Visualizations", "About"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("AVIP 2026 · Task 8 · Predictive Maintenance")

    if page == "Dashboard":
        render_dashboard(artifact, metrics)
    elif page == "Predict":
        render_predict(artifact, demo)
    elif page == "Visualizations":
        render_viz(metrics)
    else:
        render_about(metrics)


def render_dashboard(artifact, metrics):
    st.title("Failure Sensor Predictor")
    st.markdown(
        """
        **Industrial IoT predictive maintenance** — estimate whether a machine is
        approaching a component failure within the next **24 hours**, using hourly
        voltage, rotation, pressure, and vibration telemetry.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    test = (metrics or {}).get("metrics_test", {})
    c1.metric("Precision", f"{test.get('precision', float('nan')):.3f}" if test else "—")
    c2.metric("Recall", f"{test.get('recall', float('nan')):.3f}" if test else "—")
    c3.metric("F1", f"{test.get('f1', float('nan')):.3f}" if test else "—")
    c4.metric("Accuracy", f"{test.get('accuracy', float('nan')):.3f}" if test else "—")

    st.markdown("### Model & data")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"""
            - **Dataset:** Microsoft Azure PdM (hourly IoT telemetry, 2015)
            - **Horizon:** {(metrics or {}).get('horizon_hours', 24)} hours ahead
            - **Rolling window:** {(metrics or {}).get('rolling_window_hours', 12)} hours
            - **Model:** {(metrics or {}).get('model', 'HistGradientBoostingClassifier')}
            - **Threshold:** {(metrics or {}).get('threshold', '—')}
            """
        )
    with right:
        ds = (metrics or {}).get("dataset", {})
        split = (metrics or {}).get("split", {})
        st.markdown(
            f"""
            - **Rows (labeled):** {ds.get('n_rows', '—'):,}
            - **Machines:** {ds.get('n_machines', '—')}
            - **Positive rate:** {ds.get('positive_rate', float('nan')):.4f}
            - **Split:** chronological {split.get('train_frac', 0.7):.0%} /
              {split.get('val_frac', 0.15):.0%} /
              {split.get('test_frac', 0.15):.0%}
            - **Test rows:** {split.get('test_rows', '—'):,}
            """
        )

    st.markdown("### Why predictive maintenance?")
    st.write(
        "Unplanned downtime is expensive. A model that flags elevated failure risk "
        "hours before breakdown lets maintenance teams schedule inspections and "
        "parts replacement during planned windows instead of emergency stops."
    )

    if FIGURES_DIR.joinpath("metrics_bars.png").exists():
        st.markdown("### Held-out test metrics")
        show_figure("metrics_bars.png")


def render_predict(artifact, demo):
    st.title("Prediction")
    if artifact is None:
        st.error("Model artifact missing. Run `python src/train.py` first.")
        return

    mode = st.radio(
        "Input mode",
        ["Demo examples", "Manual sensor entry", "Upload CSV"],
        horizontal=True,
    )

    feature_cols = artifact["feature_cols"]
    threshold = float(artifact["threshold"])
    model = artifact["model"]
    importance = artifact.get("feature_importance", {})

    if mode == "Demo examples":
        if demo is None:
            st.warning("No demo_samples.csv found.")
            return
        labels = []
        for i, row in demo.iterrows():
            actual = int(row.get("will_fail_within_horizon", row.get("actual", -1)))
            labels.append(
                f"#{i} · machine {int(row['machineID'])} · "
                f"{row['datetime']} · actual={actual}"
            )
        choice = st.selectbox("Choose a labeled demo row (from chronological test set)", labels)
        idx = labels.index(choice)
        row = demo.iloc[idx]
        st.write("**Sensors**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("volt", f"{row['volt']:.2f}")
        sc2.metric("rotate", f"{row['rotate']:.2f}")
        sc3.metric("pressure", f"{row['pressure']:.2f}")
        sc4.metric("vibration", f"{row['vibration']:.2f}")

        if all(c in demo.columns for c in feature_cols):
            X = row[feature_cols].astype(float).to_frame().T
            prob = float(model.predict_proba(X)[0, 1])
        else:
            prob = float(row.get("failure_probability", 0.0))
        pred = int(prob >= threshold)
        _show_result(prob, pred, threshold, importance)

        if "will_fail_within_horizon" in row:
            st.caption(
                f"Ground-truth label on this demo row: "
                f"**{int(row['will_fail_within_horizon'])}** "
                f"(1 = failure within 24h)."
            )

    elif mode == "Manual sensor entry":
        st.info(
            "Single-row mode approximates 12h rolling stats from the current reading "
            "(std/diffs = 0). Prefer CSV upload when you have recent history."
        )
        c1, c2 = st.columns(2)
        with c1:
            volt = st.number_input("Voltage (volt)", value=170.0, step=1.0)
            rotate = st.number_input("Rotation (rotate)", value=450.0, step=1.0)
            pressure = st.number_input("Pressure", value=100.0, step=1.0)
            vibration = st.number_input("Vibration", value=40.0, step=0.5)
        with c2:
            age = st.number_input("Machine age (years)", value=10.0, step=1.0)
            model_code = st.number_input("Model code (0–3)", value=0, min_value=0, max_value=3)
            error_count_24h = st.number_input("Errors in last 24h", value=0, min_value=0)

        # Suggested presets
        st.markdown("**Quick presets**")
        p1, p2 = st.columns(2)
        if p1.button("Typical normal-ish readings"):
            volt, rotate, pressure, vibration = 170.0, 450.0, 100.0, 40.0
        if p2.button("Elevated vibration / voltage stress"):
            volt, rotate, pressure, vibration = 200.0, 350.0, 120.0, 55.0

        if st.button("Run prediction", type="primary"):
            from src.predict import predict_from_sensor_row

            result = predict_from_sensor_row(
                volt=volt,
                rotate=rotate,
                pressure=pressure,
                vibration=vibration,
                age=age,
                model_code=int(model_code),
                error_count_24h=int(error_count_24h),
                artifact=artifact,
            )
            _show_result(
                result["failure_probability"],
                result["predicted"],
                threshold,
                importance,
            )
            st.caption(result["approximation_note"])
            with st.expander("Sensor values used"):
                st.json(result["sensors"])

    else:
        st.markdown(
            "Upload a CSV with columns: `datetime`, `machineID`, `volt`, `rotate`, "
            "`pressure`, `vibration` (optional: `age`, `model` / `model_code`). "
            "Provide enough consecutive hours per machine for 12h rolling features."
        )
        uploaded = st.file_uploader("Telemetry CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if "datetime" not in df.columns:
                st.error("CSV must include a datetime column.")
                return
            df["datetime"] = pd.to_datetime(df["datetime"])
            if "machineID" not in df.columns:
                df["machineID"] = 1
            for col in ["volt", "rotate", "pressure", "vibration"]:
                if col not in df.columns:
                    st.error(f"Missing required column: {col}")
                    return

            # Attach defaults for meta if absent
            if "age" not in df.columns:
                df["age"] = 10
            if "model_code" not in df.columns:
                df["model_code"] = 0
            if "error_count_24h" not in df.columns:
                df["error_count_24h"] = 0
            # Dummy label column so engineer_features pipeline can run get path
            if "will_fail_within_horizon" not in df.columns:
                df["will_fail_within_horizon"] = 0

            from src.feature_engineering import engineer_features
            from src.predict import predict_engineered_frame

            try:
                eng = engineer_features(df)
                preds = predict_engineered_frame(eng, artifact)
            except Exception as exc:
                st.error(f"Could not score CSV: {exc}")
                return

            st.success(f"Scored {len(preds)} rows after feature engineering.")
            show = preds[
                [
                    c
                    for c in [
                        "datetime",
                        "machineID",
                        "volt",
                        "rotate",
                        "pressure",
                        "vibration",
                        "failure_probability",
                        "predicted",
                        "predicted_label",
                    ]
                    if c in preds.columns
                ]
            ].tail(100)
            st.dataframe(show, use_container_width=True)

            st.line_chart(
                show.set_index("datetime")[["failure_probability"]]
                if "datetime" in show.columns
                else show[["failure_probability"]]
            )


def _show_result(prob: float, pred: int, threshold: float, importance: dict):
    st.markdown("### Result")
    r1, r2, r3 = st.columns(3)
    r1.metric("Failure probability", f"{prob:.3f}")
    r2.metric("Predicted class", "FAILURE RISK" if pred == 1 else "NORMAL")
    r3.metric("Decision threshold", f"{threshold:.3f}")

    if pred == 1:
        st.error("Elevated risk: model predicts a failure within the 24-hour horizon.")
    else:
        st.success("Normal: model does not flag failure within the 24-hour horizon.")

    st.progress(min(max(prob, 0.0), 1.0))

    if importance:
        st.markdown("#### Top features (global permutation importance)")
        top = list(importance.items())[:10]
        st.bar_chart(pd.DataFrame({"importance": [v for _, v in top]}, index=[k for k, _ in top]))


def render_viz(metrics):
    st.title("Visualizations")
    tabs = st.tabs(
        [
            "Class balance",
            "Sensor trends",
            "Rolling features",
            "Confusion matrix",
            "Metrics",
            "Predictions",
            "Feature importance",
        ]
    )
    with tabs[0]:
        show_figure("class_distribution.png", "Label distribution on labeled telemetry")
    with tabs[1]:
        show_figure("sensor_trends.png")
    with tabs[2]:
        show_figure("rolling_feature_example.png")
    with tabs[3]:
        show_figure("confusion_matrix.png")
        if metrics and "metrics_test" in metrics:
            st.json(metrics["metrics_test"].get("confusion_matrix"))
    with tabs[4]:
        show_figure("metrics_bars.png")
    with tabs[5]:
        show_figure("predictions_vs_actual.png")
    with tabs[6]:
        show_figure("feature_importance.png")
        if metrics and "feature_importance_top15" in metrics:
            st.dataframe(
                pd.DataFrame(
                    list(metrics["feature_importance_top15"].items()),
                    columns=["feature", "importance"],
                ),
                use_container_width=True,
            )


def render_about(metrics):
    st.title("About this project")
    st.markdown(
        """
        ### AVIP 2026 — Task 8
        Build an industrial IoT **failure sensor predictor** that flags machines
        approaching failure using time-series sensor telemetry.

        ### Target
        Binary label: will a component failure occur within **24 hours** after time T?

        ### Leakage-safe design
        - Causal 12-hour rolling statistics only
        - Chronological train / validation / test split
        - Failure times used only as labels
        - Class weights fit on the training split only

        ### Stack
        Python · pandas · scikit-learn HistGradientBoosting · Streamlit · Render-ready
        """
    )
    if metrics:
        with st.expander("metrics.json"):
            st.json(metrics)


if __name__ == "__main__":
    main()
