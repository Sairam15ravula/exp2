"""Dataset preparation and splitting for SOH estimation.

Rule 1: Split train/test BY BATTERY (leave-one-battery-out).
Rule 2: NO TARGET LEAKAGE. Capacity and initial_capacity must NOT be input features.
"""

from __future__ import annotations

from typing import Generator
import pandas as pd

from ev_battery.data.features import extract_features
from ev_battery.data.loader import CycleData

# Non-leaky input features allowed for SOH prediction
FEATURE_COLUMNS = [
    "cycle_number",
    "voltage_max",
    "voltage_min",
    "voltage_mean",
    "voltage_drop_rate",
    "temp_max",
    "temp_min",
    "temp_rise",
    "internal_resistance_proxy",
    "time_in_voltage_window",
    "duration_s",
]

# Explicitly forbidden column names or substrings to prevent target leakage
FORBIDDEN_TARGET_SUBSTRINGS = {"capacity", "initial_capacity", "soh", "target"}


def validate_no_target_leakage(feature_columns: list[str]) -> None:
    """Validate that feature columns contain no target leakage.

    Rule 2: SOH = capacity / initial_capacity. Therefore capacity and
    initial_capacity must NOT be input features.

    Raises:
        ValueError: If any feature name matches or contains forbidden target variables.
    """
    for col in feature_columns:
        col_lower = col.lower().strip()
        for forbidden in FORBIDDEN_TARGET_SUBSTRINGS:
            if forbidden in col_lower:
                raise ValueError(
                    f"Target leakage detected! Feature '{col}' contains forbidden target term '{forbidden}'."
                )


def prepare_soh_dataset(cycles: list[CycleData]) -> pd.DataFrame:
    """Prepare clean tabular dataset for SOH modeling from raw cycle records.

    Computes SOH = capacity / initial_capacity for each battery, where
    initial_capacity is the first observed discharge capacity for that battery.

    Args:
        cycles: List of CycleData records.

    Returns:
        DataFrame containing metadata (battery_id, cycle_number), target (soh, capacity),
        and all engineered features.
    """
    discharge_cycles = [c for c in cycles if c.cycle_type == "discharge"]
    if not discharge_cycles:
        return pd.DataFrame()

    # Find initial capacity per battery (first cycle with valid capacity)
    initial_capacities: dict[str, float] = {}
    for c in discharge_cycles:
        if c.battery_id not in initial_capacities and c.capacity is not None and c.capacity > 0:
            initial_capacities[c.battery_id] = float(c.capacity)

    rows: list[dict] = []
    for c in discharge_cycles:
        if c.capacity is None or c.battery_id not in initial_capacities:
            continue

        c_init = initial_capacities[c.battery_id]
        soh = float(c.capacity / c_init)

        feats = extract_features(c)
        feats["battery_id"] = c.battery_id
        feats["cycle_number"] = c.cycle_number
        feats["capacity"] = float(c.capacity)
        feats["initial_capacity"] = c_init
        feats["soh"] = soh
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def get_feature_matrix_and_target(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y, strictly enforcing no target leakage.

    Args:
        df: Prepared SOH dataframe.
        feature_columns: Feature names to include. Defaults to FEATURE_COLUMNS.

    Returns:
        (X, y) where X has only non-leaky features and y is SOH.
    """
    cols = feature_columns if feature_columns is not None else list(FEATURE_COLUMNS)
    validate_no_target_leakage(cols)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    X = df[cols].copy()
    y = df["soh"].copy()
    return X, y


def leave_one_battery_out_splits(
    df: pd.DataFrame,
) -> Generator[tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """Generate Leave-One-Battery-Out (LOBO) train/test splits.

    Rule 1: Split train/test BY BATTERY. Never use random row splits on cycle data.

    Yields:
        (held_out_battery_id, train_df, test_df)
    """
    unique_batteries = sorted(df["battery_id"].unique())
    if len(unique_batteries) < 2:
        raise ValueError("Leave-one-battery-out requires at least 2 distinct batteries.")

    for b_id in unique_batteries:
        train_df = df[df["battery_id"] != b_id].reset_index(drop=True)
        test_df = df[df["battery_id"] == b_id].reset_index(drop=True)

        # Enforce Rule 1 assertion: Disjoint batteries
        train_batteries = set(train_df["battery_id"])
        test_batteries = set(test_df["battery_id"])
        assert train_batteries.isdisjoint(test_batteries), "Data leakage! Battery sets overlap."

        yield str(b_id), train_df, test_df
