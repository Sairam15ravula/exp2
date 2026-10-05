"""Dataset preparation and splitting for Remaining Useful Life (RUL) prediction.

Rule 1: Split train/test BY BATTERY (leave-one-battery-out).
Rule 2: NO TARGET LEAKAGE. Capacity and initial_capacity must NOT be input features.
Rule 4: RUL is defined as cycles until SOH <= 80% (end of life).
        NOT cycles until the experiment ended.
"""

from __future__ import annotations

from typing import Generator
import pandas as pd

from ev_battery.data.loader import CycleData
from ev_battery.soh.dataset import (
    FEATURE_COLUMNS,
    prepare_soh_dataset,
    validate_no_target_leakage,
)

# Rule 4: End-of-life definition (80% State of Health)
EOL_SOH_THRESHOLD = 0.80

# Forbidden feature names preventing target leakage (Rule 2)
FORBIDDEN_RUL_TARGET_SUBSTRINGS = {"capacity", "initial_capacity", "soh", "rul", "target", "eol"}


def validate_no_rul_target_leakage(feature_columns: list[str]) -> None:
    """Validate that feature columns contain no target leakage for RUL prediction.

    Rule 2: Capacity, initial_capacity, SOH, and RUL must NOT be input features.
    """
    for col in feature_columns:
        col_lower = col.lower().strip()
        for forbidden in FORBIDDEN_RUL_TARGET_SUBSTRINGS:
            if forbidden in col_lower:
                raise ValueError(
                    f"Target leakage detected! Feature '{col}' contains forbidden target term '{forbidden}'."
                )


def find_eol_cycle(
    battery_df: pd.DataFrame,
    eol_threshold: float = EOL_SOH_THRESHOLD,
) -> int | None:
    """Find the first cycle index where SOH drops to or below the EOL threshold (0.80).

    Rule 4: RUL is defined relative to SOH <= 80%, NOT experiment end.

    Args:
        battery_df: DataFrame of discharge cycles for a single battery, containing 'soh' and 'cycle_number'.
        eol_threshold: EOL SOH threshold (default 0.80 = 80%).

    Returns:
        The first cycle_number where SOH <= eol_threshold, or None if battery never reached EOL.
    """
    eol_records = battery_df[battery_df["soh"] <= eol_threshold]
    if eol_records.empty:
        return None
    return int(eol_records["cycle_number"].iloc[0])


def prepare_rul_dataset(
    cycles: list[CycleData],
    eol_threshold: float = EOL_SOH_THRESHOLD,
) -> pd.DataFrame:
    """Prepare clean tabular dataset for RUL estimation from cycle telemetry.

    Rule 4: For each battery, finds the first cycle k_eol where SOH <= eol_threshold (80%).
    Then ground-truth RUL(k) = k_eol - k for all cycles k <= k_eol.
    Cycles after EOL are excluded.

    Args:
        cycles: List of CycleData records.
        eol_threshold: End-of-Life SOH threshold (default: 0.80).

    Returns:
        DataFrame with features, metadata (battery_id, cycle_number, eol_cycle), and target 'rul'.
    """
    soh_df = prepare_soh_dataset(cycles)
    if soh_df.empty:
        return pd.DataFrame()

    rul_rows: list[dict] = []
    unique_batteries = sorted(soh_df["battery_id"].unique())

    for b_id in unique_batteries:
        b_df = soh_df[soh_df["battery_id"] == b_id].sort_values("cycle_number").reset_index(drop=True)
        eol_cycle = find_eol_cycle(b_df, eol_threshold=eol_threshold)

        if eol_cycle is None:
            # Battery never reached EOL (80% SOH); cannot determine ground-truth RUL without censoring
            continue

        # Keep cycles up to and including the EOL cycle
        active_cycles = b_df[b_df["cycle_number"] <= eol_cycle].copy()
        for _, row in active_cycles.iterrows():
            c_num = int(row["cycle_number"])
            # Rule 4: RUL is remaining cycles until SOH <= 80%
            true_rul = float(eol_cycle - c_num)

            row_dict = row.to_dict()
            row_dict["eol_cycle"] = eol_cycle
            row_dict["rul"] = true_rul
            rul_rows.append(row_dict)

    if not rul_rows:
        return pd.DataFrame()

    return pd.DataFrame(rul_rows)


def get_rul_feature_matrix_and_target(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and RUL target y with strict target leakage validation."""
    cols = feature_columns if feature_columns is not None else list(FEATURE_COLUMNS)
    validate_no_rul_target_leakage(cols)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")

    X = df[cols].copy()
    y = df["rul"].copy()
    return X, y


def leave_one_battery_out_rul_splits(
    df: pd.DataFrame,
) -> Generator[tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """Generate Leave-One-Battery-Out (LOBO) splits for RUL dataset.

    Rule 1: Train and test sets must have completely disjoint batteries.
    """
    unique_batteries = sorted(df["battery_id"].unique())
    if len(unique_batteries) < 2:
        raise ValueError("LOBO splitting requires at least 2 distinct batteries with reached EOL.")

    for b_id in unique_batteries:
        train_df = df[df["battery_id"] != b_id].reset_index(drop=True)
        test_df = df[df["battery_id"] == b_id].reset_index(drop=True)

        # Enforce Rule 1 disjointness check
        train_bats = set(train_df["battery_id"])
        test_bats = set(test_df["battery_id"])
        assert train_bats.isdisjoint(test_bats), "Data leakage! Battery sets overlap in LOBO split."

        yield str(b_id), train_df, test_df
