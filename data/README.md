# Dataset: Microsoft Azure Predictive Maintenance (PdM)

## Dataset name

**Microsoft Azure Predictive Maintenance Sample Dataset** (also known as the Predictive Maintenance Modelling Guide datasets).

## Original source

Microsoft Azure Predictive Maintenance sample dataset from Microsoft’s Predictive Maintenance Modelling Guide (Fidan Boylu Uz / Microsoft).

Public historical / benchmark IoT telemetry and failure logs used for predictive-maintenance tutorials. Historically distributed via Azure AI Gallery / Azure ML sample experiments.

## Download / source URLs

### Verified working mirror (used by this project)

The original Azure blob endpoints are intermittently unreachable. This project downloads the **same Microsoft Azure PdM CSV files** from a public GitHub mirror:

- Repository: https://github.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy
- Data folder: https://github.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/tree/master/data

Direct raw URLs (reproducible):

- https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_telemetry.csv
- https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_errors.csv
- https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_maint.csv
- https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_failures.csv
- https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data/PdM_machines.csv

These files use the official `PdM_*` naming and match the Microsoft sample schema (hourly telemetry for 100 machines in 2015, ~876k rows).

### Original Microsoft sources (may be unavailable)

- Azure blob: `https://azuremlsampleexperiments.blob.core.windows.net/datasets/PdM_*.csv`
- Historical guide blob: `https://pdmmodelingguide.blob.core.windows.net/pdmdata/*.csv`
- Microsoft SQL-Server-R-Services-Samples (legacy GitHub paths may 404)

`scripts/download_data.py` tries the GitHub mirror first, then falls back to the Azure blob.

### Reproduce locally

```bash
python scripts/download_data.py
```

Files are stored under `data/raw/` (git-ignored). Do **not** commit the large raw CSVs.

## Files used

| File | Role |
|------|------|
| `PdM_telemetry.csv` | Hourly sensor averages per machine |
| `PdM_failures.csv` | Component failure / replacement events |
| `PdM_machines.csv` | Static machine attributes (model, age) |
| `PdM_errors.csv` | Non-fatal error events (past-only counts) |
| `PdM_maint.csv` | Maintenance log (not required for the default pipeline) |

## Columns (telemetry)

| Column | Description |
|--------|-------------|
| `datetime` | Timestamp (hourly) |
| `machineID` | Machine identifier (1–100) |
| `volt` | Voltage |
| `rotate` | Rotation / RPM-like operating measure |
| `pressure` | Pressure |
| `vibration` | Vibration |

### Failures

| Column | Description |
|--------|-------------|
| `datetime` | Failure / replacement time (hour-aligned) |
| `machineID` | Machine ID |
| `failure` | Failed component (`comp1`…`comp4`) |

### Machines

| Column | Description |
|--------|-------------|
| `machineID` | Machine ID |
| `model` | Machine model type |
| `age` | Machine age (years) |

## Sampling frequency

**Hourly** averages for calendar year **2015**, ~100 machines → on the order of **~876,000** telemetry rows.

## Target definition

At timestamp `T` for machine `M`:

```
will_fail_within_horizon = 1
  iff there exists a failure for M with datetime F satisfying
      T < F ≤ T + 24 hours
```

Equivalently: for each failure at `F`, label hours in `(F − 24h, F)` as positive.

This is a **genuine 24-hour-ahead** binary predictive-maintenance target because the dataset has real timestamps and failure event times at hourly resolution.

## Preprocessing

1. Parse datetimes; sort by `(machineID, datetime)`.
2. Build the 24h failure label (future events → label only).
3. Join machine `age` / `model_code`.
4. Count errors in the past 24 hours (causal rolling sum).
5. Engineer **12-hour** causal rolling mean/std/min/max, 1h & 12h diffs, current sensor values.
6. Drop early rows until the rolling window has enough history.
7. **Chronological** train/val/test split on global time (70% / 15% / 15%). No random shuffle.

## Leakage controls

- Rolling features use only past/present values at `T`.
- Error counts use only errors at or before `T`.
- Failure timestamps are used **only** for labels, never as input features.
- Train-only class weights; threshold tuned on validation only.
- Test set is the latest chronological segment.

## Limitations

- Historical benchmark / tutorial dataset — not live production telemetry from a currently operating plant.
- Failures are component replacements; operational “breakdown” semantics follow the Microsoft guide.
- Hourly aggregation hides sub-hour dynamics.
- Positive class is rare → emphasize precision, recall, and F1 for the failure class; accuracy alone is misleading.
- `error_count_24h` can dominate feature importance; interpret as a predictive signal in this dataset, not causal proof.
- Manual single-row inference in the app approximates rolling stats without history; CSV upload uses proper causal features.

## License / usage

Distributed by Microsoft as a public sample for predictive-maintenance tutorials. Reuse for education and portfolio projects is the common practice for this sample; verify current Microsoft sample terms if redistributing the raw files commercially. This repository does **not** vend the raw CSVs—only download scripts and documentation.
