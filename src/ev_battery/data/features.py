"""Feature extraction from battery discharge cycles.

Features are designed to capture degradation signals without target leakage.
NO feature uses capacity or initial_capacity (rule 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ev_battery.data.loader import CycleData

# Voltage window for "time in window" feature (typical Li-ion operating range)
VOLTAGE_WINDOW_LOW = 3.0  # V
VOLTAGE_WINDOW_HIGH = 4.2  # V


def extract_features(cycle: CycleData) -> dict[str, float]:
    """Extract features from a single discharge cycle.

    Features:
        - voltage_max, voltage_min, voltage_mean: voltage-curve shape
        - voltage_drop_rate: how fast voltage falls (degradation signal)
        - temp_max, temp_min, temp_rise: thermal behaviour
        - internal_resistance_proxy: V_drop / I at start of discharge
        - time_in_voltage_window: time spent between 3.0V and 4.2V
        - duration_s: discharge duration

    Args:
        cycle: A discharge CycleData.

    Returns:
        Dictionary of feature names to values.
    """
    v = np.asarray(cycle.voltage, dtype=float)
    t = np.asarray(cycle.time, dtype=float)
    temp = np.asarray(cycle.temperature, dtype=float)
    curr = np.asarray(cycle.current, dtype=float)

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    # Voltage-curve shape
    voltage_max = float(np.max(v))
    voltage_min = float(np.min(v))
    voltage_mean = float(np.mean(v))
    voltage_drop_rate = (voltage_max - voltage_min) / duration if duration > 0 else 0.0

    # Thermal behaviour
    temp_max = float(np.max(temp))
    temp_min = float(np.min(temp))
    temp_rise = temp_max - temp_min

    # Internal resistance proxy: voltage drop from start to mid-discharge
    # divided by average current magnitude
    mid_idx = len(v) // 2
    if mid_idx > 0 and len(curr) > 0:
        avg_current = float(np.mean(np.abs(curr)))
        internal_resistance_proxy = (
            float(v[0] - v[mid_idx]) / avg_current if avg_current > 0 else 0.0
        )
    else:
        internal_resistance_proxy = 0.0

    # Time in voltage window (3.0V to 4.2V)
    if len(v) > 1 and len(t) > 1:
        in_window = (v >= VOLTAGE_WINDOW_LOW) & (v <= VOLTAGE_WINDOW_HIGH)
        time_in_voltage_window = float(np.sum(np.diff(t)[in_window[:-1]]))
    else:
        time_in_voltage_window = 0.0

    return {
        "voltage_max": voltage_max,
        "voltage_min": voltage_min,
        "voltage_mean": voltage_mean,
        "voltage_drop_rate": voltage_drop_rate,
        "temp_max": temp_max,
        "temp_min": temp_min,
        "temp_rise": temp_rise,
        "internal_resistance_proxy": internal_resistance_proxy,
        "time_in_voltage_window": time_in_voltage_window,
        "duration_s": duration,
    }


def extract_all_features(cycles: list[CycleData]) -> pd.DataFrame:
    """Extract features from all discharge cycles.

    Args:
        cycles: List of CycleData (any type; non-discharge are skipped).

    Returns:
        DataFrame with one row per discharge cycle, including battery_id and
        cycle_number for joining.
    """
    rows = []
    for c in cycles:
        if c.cycle_type != "discharge":
            continue
        feats = extract_features(c)
        feats["battery_id"] = c.battery_id
        feats["cycle_number"] = c.cycle_number
        rows.append(feats)
    return pd.DataFrame(rows)
