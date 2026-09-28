# Failure Sensor Predictor

Industrial IoT predictive-maintenance system that flags machines approaching component failure **24 hours ahead**, using hourly sensor telemetry from the Microsoft Azure Predictive Maintenance sample dataset.

Built for **AVIP 2026 — Task 8: Failure Sensor Predictor**.

---

## Project overview

This project trains a classical ML model on engineered time-series features (12-hour causal rolling statistics) to predict whether a machine will experience a component failure within the next 24 hours. It includes a reproducible data pipeline, evaluation reports, Jupyter notebook, and a Render-ready Streamlit dashboard.

## Problem statement

Unplanned equipment downtime is costly. Given hourly sensor readings (voltage, rotation, pressure, vibration), can we detect machines that will fail within 24 hours so maintenance can be scheduled proactively?

## Dataset

| Item | Value |
|------|-------|
| **Name** | Microsoft Azure Predictive Maintenance (PdM) |
| **Source** | Microsoft sample / Azure ML experiments |
| **Mirror used** | [ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy](https://github.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/tree/master/data) |
| **Records** | 876,100 hourly telemetry rows |
| **Machines** | 100 |
| **Period** | 2015-01-01 → 2016-01-01 |
| **Failures** | 761 component replacement events |

See [`data/README.md`](data/README.md) for columns, download URLs, and license notes.

### Sensors

- `volt` — voltage  
- `rotate` — rotation / RPM-like measure  
- `pressure` — pressure  
- `vibration` — vibration  

## Target definition

At timestamp **T** for machine **M**:

```
will_fail_within_horizon = 1
  iff a failure for M occurs at time F with T < F ≤ T + 24 hours
```

This is a **genuine 24-hour-ahead** label because telemetry and failures are hourly and failure timestamps are explicit. The dataset supports this horizon without fabrication.

**Positive rate:** ~1.88% of labeled telemetry rows.

## Feature engineering

Causal (past-only) features per machine, computed in `src/feature_engineering.py`:

| Category | Features |
|----------|----------|
| 12h rolling | mean, std, min, max for each sensor |
| Differences | 1h and 12h diffs per sensor |
| Current | latest sensor reading at T |
| Machine meta | `age`, `model_code` |
| Errors | `error_count_24h` (past 24h non-fatal errors) |

**12-hour rolling window:** 12 consecutive hourly samples = 12 hours — meaningful for this dataset.

**Total features:** 31

## Model

**HistGradientBoostingClassifier** (scikit-learn)

- Strong tabular performance on large engineered feature matrices  
- No XGBoost dependency (Render-friendly)  
- Class imbalance handled via `sample_weight='balanced'` on **train only**  
- Decision threshold tuned on **validation** set (0.80) to emphasize failure recall  

Saved artifact: `models/failure_predictor.joblib` (~790 KB)

## Time-aware validation strategy

| Split | Fraction | Rows | Date range ends |
|-------|----------|------|-----------------|
| Train | 70% | 612,400 | ≤ 2015-09-13 |
| Validation | 15% | 131,200 | ≤ 2015-11-07 |
| Test | 15% | 131,300 | remainder |

- **Chronological global split** on datetime — no random shuffle  
- No oversampling before splitting  
- Threshold selected on validation; test is fully held out  

## Evaluation metrics (held-out test set)

| Metric | Value |
|--------|-------|
| **Precision** | 0.769 |
| **Recall** | 0.966 |
| **F1** | 0.856 |
| **Accuracy** | 0.994 |
| **ROC-AUC** | 0.998 |

Confusion matrix (test):

|  | Pred 0 | Pred 1 |
|--|--------|--------|
| **Actual 0** | 128,273 | 681 |
| **Actual 1** | 79 | 2,267 |

High accuracy reflects class imbalance; **recall (96.6%)** and **precision (76.9%)** on the failure class are the primary operational metrics.

Full metrics: [`reports/metrics.json`](reports/metrics.json)

## Example prediction

Manual sensor entry (typical normal-ish readings):

```python
from src.predict import predict_from_sensor_row

predict_from_sensor_row(volt=170.0, rotate=450.0, pressure=100.0, vibration=40.0)
# → predicted: normal, failure_probability ≈ 0.00005
```

Demo rows from the test set: [`data/demo_samples.csv`](data/demo_samples.csv)

## Project structure

```
Failure-Sensor-Predictor/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/
│   ├── README.md
│   └── demo_samples.csv
├── models/
│   └── failure_predictor.joblib
├── notebooks/
│   └── predictive_maintenance.ipynb
├── reports/
│   ├── metrics.json
│   ├── sample_predictions.csv
│   └── figures/
├── scripts/
│   └── download_data.py
└── src/
    ├── config.py
    ├── data_loader.py
    ├── feature_engineering.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Installation

```bash
git clone <your-repo-url>
cd Failure-Sensor-Predictor
python -m venv .venv

# Windows
.\.venv\Scripts\pip install -r requirements.txt

# macOS / Linux
source .venv/bin/activate && pip install -r requirements.txt
```

## Download data

```bash
python scripts/download_data.py
```

Downloads the official Microsoft PdM CSVs from the verified GitHub mirror into `data/raw/` (git-ignored).

## Training

```bash
python src/train.py
```

Produces:

- `models/failure_predictor.joblib`
- `reports/metrics.json`
- `reports/sample_predictions.csv`
- `reports/figures/*.png`
- `data/demo_samples.csv`

## Evaluation

```bash
python src/evaluate.py
```

Re-scores the chronological test set from the saved model.

## Prediction

```bash
python src/predict.py
```

## Running Streamlit

```bash
streamlit run app.py
```

Local URL: http://127.0.0.1:8501

The app includes:

- Dashboard with model metrics  
- Manual sensor entry, CSV upload, and demo examples  
- Visualizations (confusion matrix, feature importance, sensor trends)  

## Deployment (Render)

**Build command:**

```bash
pip install -r requirements.txt
python scripts/download_data.py   # optional if model committed
python src/train.py               # optional if model committed
```

**Start command:**

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

The trained model and reports are included in the repo so Render can serve predictions without retraining.

## Limitations

- **Historical benchmark, not live production** — the pipeline uses the real downloaded Microsoft Azure Predictive Maintenance (PdM) telemetry and failure-event CSVs. This is a public historical/benchmark dataset for predictive-maintenance research and tutorials, not live plant telemetry from a currently operating facility.
- **Sparse failures / class imbalance** — failure events are rare (~1.9% positive labels). Accuracy is therefore **not** the primary metric; precision, recall, and F1 for the failure class are more informative.
- **Hourly aggregation** — sensor values are hourly averages; sub-hour dynamics are not observed.
- **Feature importance ≠ causality** — `error_count_24h` has substantial permutation importance. Treat it as a predictive signal available in this dataset, not as proof that recent errors *cause* failures.
- **Manual single-row Streamlit predictions** — entering one observation approximates 12-hour rolling features (no historical window). Prefer CSV upload when recent telemetry history is available so causal rolling stats can be computed correctly.
- **Azure blob URLs** may be unreachable; this project downloads the same official `PdM_*` files from a verified public GitHub mirror (see [`data/README.md`](data/README.md)).

## Future improvements

- Per-machine or grouped cross-validation for robustness  
- Survival / time-to-failure models (Cox, RSF)  
- SHAP explanations per prediction  
- Separate models per failure component (`comp1`–`comp4`)  
- Online scoring API with rolling feature store  

---

## How This Project Works

1. **Download** the Microsoft Azure PdM CSVs (telemetry, failures, errors, machines).  
2. **Label** each hourly row: will this machine fail in the next 24 hours? Labels use future failure times only — never as input features.  
3. **Engineer features** with causal 12-hour rolling statistics, diffs, and past error counts per machine.  
4. **Split chronologically** (70/15/15) so the model never trains on future data.  
5. **Train** HistGradientBoosting with balanced sample weights on the train split.  
6. **Tune threshold** on validation to balance precision and recall (maintenance cares about catching failures).  
7. **Evaluate** on the latest 15% of time (test) — report precision, recall, F1, confusion matrix.  
8. **Deploy** via Streamlit: load the saved model, score manual inputs or uploaded CSVs, show metrics and plots.  

No future information leaks into features. The 24-hour target is real because failures have exact hourly timestamps.

---

## Interview Questions

**1. What problem does this project solve?**  
Predicting component failure 24 hours ahead from IoT sensor data so maintenance can be planned instead of reactive.

**2. Why the Microsoft Azure PdM dataset?**  
Public, reproducible, hourly telemetry with explicit failure timestamps — supports a genuine 24h prediction horizon.

**3. How is the target defined?**  
Label = 1 if a failure occurs for the same machine in (T, T+24h]. Equivalent to marking the 24 hours before each failure as positive.

**4. How do you prevent data leakage?**  
Causal rolling features, past-only error counts, chronological split, no random shuffle, class weights fit on train only, threshold tuned on validation.

**5. Why 12-hour rolling windows?**  
Telemetry is hourly; 12 samples = 12 hours of operating history — captures half-day trends without using future data.

**6. Why HistGradientBoosting over XGBoost?**  
Strong tabular performance, built into scikit-learn, lighter deployment on Render, handles large feature matrices efficiently.

**7. Why is recall more important than accuracy here?**  
Failures are rare (~2%); missing a failure (false negative) is costlier than a false alarm. Test recall for failure class is 96.6%.

**8. What is the main limitation of your best feature?**  
`error_count_24h` dominates importance — the model relies heavily on recent error logs, not only raw sensors.

**9. How would you validate this in production?**  
Track precision/recall over rolling time windows, alert fatigue rate, lead time to actual failure, and drift in sensor distributions.

**10. What would you improve next?**  
Component-specific models, proper feature store for rolling stats at inference, and SHAP-based per-prediction explanations.

---

## License

Code in this repository: MIT (or your chosen license).  
Dataset: Microsoft Azure Predictive Maintenance sample — see [`data/README.md`](data/README.md).
