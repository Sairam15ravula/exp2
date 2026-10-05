"""Segment raw cycle data into a clean, analysis-ready cycle table.

The NASA dataset already separates cycles, but we filter to discharge cycles
only (those have capacity measurements) and build a flat DataFrame.
"""

from __future__ import annotations

import pandas as pd

from ev_battery.data.loader import CycleData


def segment_cycles(cycles: list[CycleData]) -> pd.DataFrame:
    """Convert list of CycleData into a clean cycle-level DataFrame.

    Only discharge cycles are kept (they have capacity measurements).
    Charge and impedance cycles are excluded from the cycle table.

    Args:
        cycles: Raw cycles from the loader.

    Returns:
        DataFrame with one row per discharge cycle.
    """
    rows = []
    for c in cycles:
        if c.cycle_type != "discharge":
            continue
        rows.append({
            "battery_id": c.battery_id,
            "cycle_number": c.cycle_number,
            "capacity": c.capacity,
            "duration_s": float(c.time[-1] - c.time[0]) if len(c.time) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def get_discharge_cycles(cycles: list[CycleData]) -> list[CycleData]:
    """Filter to discharge cycles only."""
    return [c for c in cycles if c.cycle_type == "discharge"]
