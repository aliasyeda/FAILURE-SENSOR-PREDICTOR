"""Time-series feature engineering with causal (past-only) rolling windows."""

from __future__ import annotations

import pandas as pd

from src.config import ROLLING_WINDOW_HOURS, SENSOR_COLS


def _rolling_stats(
    group: pd.DataFrame,
    cols: list[str],
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    """
    Causal rolling stats: at time T, use rows [T-window+1, T] inclusive.
    pandas rolling is causal by default (looks backward).
    """
    out = pd.DataFrame(index=group.index)
    for col in cols:
        r = group[col].rolling(window=window, min_periods=min_periods)
        out[f"{col}_roll_mean_{window}h"] = r.mean()
        out[f"{col}_roll_std_{window}h"] = r.std()
        out[f"{col}_roll_min_{window}h"] = r.min()
        out[f"{col}_roll_max_{window}h"] = r.max()
        # Rate of change vs previous hour (past only)
        out[f"{col}_diff_1h"] = group[col].diff(1)
        # Change vs value ~window hours ago
        out[f"{col}_diff_{window}h"] = group[col].diff(window)
        # Recent raw value (current reading at T — available at prediction time)
        out[f"{col}_current"] = group[col]
    return out


def engineer_features(
    df: pd.DataFrame,
    window_hours: int = ROLLING_WINDOW_HOURS,
    sensor_cols: list[str] | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Build per-machine causal features.

    For hourly data, window_hours=12 → 12-hour rolling statistics
    (mean, std, min, max), 1h/12h differences, and current values.
    """
    if sensor_cols is None:
        sensor_cols = [c for c in SENSOR_COLS if c in df.columns]
    if min_periods is None:
        min_periods = max(3, window_hours // 3)

    df = df.sort_values(["machineID", "datetime"]).copy()
    feature_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("machineID", sort=False):
        feats = _rolling_stats(group, sensor_cols, window_hours, min_periods)
        feature_parts.append(feats)

    feat_df = pd.concat(feature_parts).sort_index()
    result = pd.concat([df, feat_df], axis=1)

    # Operating / machine-level statistics available at T (static or past)
    meta_cols = [c for c in ["age", "model_code", "error_count_24h"] if c in result.columns]

    feature_cols = list(feat_df.columns) + meta_cols
    # Drop rows where rolling window not yet filled enough
    result = result.dropna(subset=feature_cols).reset_index(drop=True)
    result.attrs["feature_cols"] = feature_cols
    return result


def get_feature_matrix(
    engineered: pd.DataFrame, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if feature_cols is None:
        feature_cols = engineered.attrs.get("feature_cols")
        if feature_cols is None:
            raise ValueError("feature_cols not provided and not found on dataframe")
    X = engineered[feature_cols].astype(float)
    y = engineered["will_fail_within_horizon"].astype(int)
    return X, y, feature_cols


def chronological_split(
    engineered: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """
    Time-aware split on global datetime (same cut for all machines).

    Train: earliest train_frac of the calendar span
    Val: next val_frac
    Test: remainder

    No random shuffle — prevents future leakage into training.
    """
    df = engineered.sort_values("datetime").reset_index(drop=True)
    t_min, t_max = df["datetime"].min(), df["datetime"].max()
    span = t_max - t_min
    train_end = t_min + span * train_frac
    val_end = t_min + span * (train_frac + val_frac)

    train = df[df["datetime"] <= train_end].copy()
    val = df[(df["datetime"] > train_end) & (df["datetime"] <= val_end)].copy()
    test = df[df["datetime"] > val_end].copy()

    return {
        "train": train,
        "val": val,
        "test": test,
        "train_end": train_end,
        "val_end": val_end,
    }


if __name__ == "__main__":
    from src.data_loader import load_labeled_dataset

    labeled = load_labeled_dataset()
    eng = engineer_features(labeled)
    splits = chronological_split(eng)
    print("Features:", eng.attrs["feature_cols"])
    print({k: len(v) for k, v in splits.items() if k in ("train", "val", "test")})
