"""Data quality validation for battery cycle data.

Checks for missing values, out-of-range values, and structural issues.
Returns a DataFrame of issues found — does not modify the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_battery.data.loader import CycleData
from ev_battery.units import (
    MAX_CELL_VOLTAGE,
    MAX_TEMPERATURE,
    MIN_CELL_VOLTAGE,
    MIN_TEMPERATURE,
)


def validate_cycle(cycle: CycleData) -> list[dict[str, str]]:
    """Validate a single cycle. Returns list of issue dicts.

    Each issue has: battery_id, cycle_number, issue_type, message.
    """
    issues: list[dict[str, str]] = []

    # Check for empty arrays
    if len(cycle.time) == 0:
        issues.append({
            "battery_id": cycle.battery_id,
            "cycle_number": str(cycle.cycle_number),
            "issue_type": "empty_time",
            "message": "Time array is empty",
        })

    if len(cycle.voltage) == 0:
        issues.append({
            "battery_id": cycle.battery_id,
            "cycle_number": str(cycle.cycle_number),
            "issue_type": "empty_voltage",
            "message": "Voltage array is empty",
        })

    # Check for NaN/Inf in voltage
    if np.any(~np.isfinite(cycle.voltage)):
        issues.append({
            "battery_id": cycle.battery_id,
            "cycle_number": str(cycle.cycle_number),
            "issue_type": "non_finite_voltage",
            "message": "Voltage contains NaN or Inf",
        })

    # Check voltage range
    if np.any(cycle.voltage < MIN_CELL_VOLTAGE) or np.any(cycle.voltage > MAX_CELL_VOLTAGE):
        issues.append({
            "battery_id": cycle.battery_id,
            "cycle_number": str(cycle.cycle_number),
            "issue_type": "voltage_out_of_range",
            "message": f"Voltage outside [{MIN_CELL_VOLTAGE}, {MAX_CELL_VOLTAGE}] V",
        })

    # Check temperature range
    if np.any(cycle.temperature < MIN_TEMPERATURE) or np.any(cycle.temperature > MAX_TEMPERATURE):
        issues.append({
            "battery_id": cycle.battery_id,
            "cycle_number": str(cycle.cycle_number),
            "issue_type": "temperature_out_of_range",
            "message": f"Temperature outside [{MIN_TEMPERATURE}, {MAX_TEMPERATURE}] °C",
        })

    # Check for negative capacity
    if cycle.capacity is not None and cycle.capacity < 0:
        issues.append({
            "battery_id": cycle.battery_id,
            "cycle_number": str(cycle.cycle_number),
            "issue_type": "negative_capacity",
            "message": f"Capacity is negative: {cycle.capacity}",
        })

    # Check array length consistency
    lengths = {len(cycle.time), len(cycle.voltage), len(cycle.current), len(cycle.temperature)}
    if len(lengths) > 1:
        issues.append({
            "battery_id": cycle.battery_id,
            "cycle_number": str(cycle.cycle_number),
            "issue_type": "array_length_mismatch",
            "message": f"Array lengths differ: time={len(cycle.time)}, voltage={len(cycle.voltage)}, current={len(cycle.current)}, temperature={len(cycle.temperature)}",
        })

    return issues


def validate_dataset(cycles: list[CycleData]) -> pd.DataFrame:
    """Validate entire dataset. Returns DataFrame of all issues found.

    Args:
        cycles: All cycles to validate.

    Returns:
        DataFrame with columns: battery_id, cycle_number, issue_type, message.
        Empty DataFrame means no issues found.
    """
    all_issues: list[dict[str, str]] = []
    for cycle in cycles:
        all_issues.extend(validate_cycle(cycle))
    return pd.DataFrame(all_issues, columns=["battery_id", "cycle_number", "issue_type", "message"])
