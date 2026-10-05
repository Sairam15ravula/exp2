"""Sequence dataset preparation and cross-temperature splitting for PyTorch models.

Rule 1: No data leakage (disjoint battery splits).
Rule 2: No target leakage (capacity and initial_capacity strictly excluded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_battery.soh.dataset import validate_no_target_leakage

# Non-leaky features for sequence modeling
SEQUENCE_FEATURE_COLUMNS = [
    "cycle_number",
    "voltage_mean",
    "voltage_drop_rate",
    "temp_max",
    "temp_rise",
    "internal_resistance_proxy",
    "duration_s",
]


def create_sequence_windows(
    df: pd.DataFrame,
    window_size: int = 5,
    feature_columns: list[str] | None = None,
    target_column: str = "soh",
) -> tuple[np.ndarray, np.ndarray]:
    """Create temporal sliding sequence windows for LSTM/GRU models.

    For each battery independently (to prevent cross-battery sequence contamination),
    generates input windows X of shape (N, window_size, n_features) and targets y (N,).

    Args:
        df: Tabular dataframe containing battery_id, cycle_number, features, and target.
        window_size: Number of past cycles included in each temporal window.
        feature_columns: Feature columns to include. Defaults to SEQUENCE_FEATURE_COLUMNS.
        target_column: Target variable (e.g. 'soh' or 'rul').

    Returns:
        (X, y) as numpy arrays of shape (N, window_size, n_features) and (N,).
    """
    cols = feature_columns if feature_columns is not None else list(SEQUENCE_FEATURE_COLUMNS)
    validate_no_target_leakage(cols)

    if window_size < 1:
        raise ValueError("window_size must be >= 1.")

    X_list: list[np.ndarray] = []
    y_list: list[float] = []

    unique_batteries = sorted(df["battery_id"].unique())

    for b_id in unique_batteries:
        b_df = df[df["battery_id"] == b_id].sort_values("cycle_number").reset_index(drop=True)
        if len(b_df) < window_size:
            continue

        feats = b_df[cols].to_numpy(dtype=float)
        targets = b_df[target_column].to_numpy(dtype=float)

        for t in range(window_size - 1, len(b_df)):
            window = feats[t - window_size + 1 : t + 1]
            X_list.append(window)
            y_list.append(targets[t])

    if not X_list:
        return np.empty((0, window_size, len(cols))), np.empty((0,))

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def cross_temperature_split(
    df: pd.DataFrame,
    split_temp_threshold: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by operating temperature for cross-temperature validation.

    Rule 1: Evaluates whether a model trained on nominal room temperatures
    generalizes to elevated or stress thermal environments.

    Args:
        df: Tabular cycle dataset containing 'temp_mean' or 'temp_max'.
        split_temp_threshold: Threshold dividing nominal vs stress operating conditions.

    Returns:
        (train_df_nominal, test_df_thermal_stress)
    """
    temp_col = "temp_max" if "temp_max" in df.columns else "temp_mean"
    if temp_col not in df.columns:
        raise KeyError("Neither temp_max nor temp_mean present in dataframe.")

    # Group by battery: determine battery average peak operating temperature
    battery_temps = df.groupby("battery_id")[temp_col].mean()

    train_bats = battery_temps[battery_temps <= split_temp_threshold].index.tolist()
    test_bats = battery_temps[battery_temps > split_temp_threshold].index.tolist()

    train_df = df[df["battery_id"].isin(train_bats)].reset_index(drop=True)
    test_df = df[df["battery_id"].isin(test_bats)].reset_index(drop=True)

    # Disjoint check (Rule 1)
    assert set(train_bats).isdisjoint(set(test_bats)), "Data leakage in cross-temperature split."

    return train_df, test_df
