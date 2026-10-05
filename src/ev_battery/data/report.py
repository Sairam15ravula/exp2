"""Generate a data quality and summary report."""

from __future__ import annotations

import pandas as pd

from ev_battery.data.loader import CycleData
from ev_battery.data.validation import validate_dataset


def generate_report(cycles: list[CycleData]) -> dict:
    """Generate a data report.

    Args:
        cycles: All loaded cycles.

    Returns:
        Dictionary with:
            - n_batteries: number of unique batteries
            - n_cycles_total: total cycles
            - n_discharge_cycles: discharge cycles only
            - n_issues: number of validation issues found
            - issue_breakdown: dict of issue_type -> count
            - capacity_summary: per-battery capacity stats (first, last, min, max)
            - temperature_summary: overall temperature stats
    """
    df = pd.DataFrame([
        {
            "battery_id": c.battery_id,
            "cycle_number": c.cycle_number,
            "cycle_type": c.cycle_type,
            "capacity": c.capacity,
            "temperature_max": float(max(c.temperature)) if len(c.temperature) > 0 else None,
        }
        for c in cycles
    ])

    issues_df = validate_dataset(cycles)

    # Capacity summary per battery (discharge cycles only)
    discharge = df[df["cycle_type"] == "discharge"].copy()
    capacity_summary = {}
    for bat_id, group in discharge.groupby("battery_id"):
        group = group.sort_values("cycle_number")
        caps = group["capacity"].dropna()
        if len(caps) > 0:
            capacity_summary[bat_id] = {
                "first_capacity": float(caps.iloc[0]),
                "last_capacity": float(caps.iloc[-1]),
                "min_capacity": float(caps.min()),
                "max_capacity": float(caps.max()),
                "n_cycles": int(len(caps)),
            }

    # Temperature summary
    temp_vals = df["temperature_max"].dropna()
    temperature_summary = {
        "overall_max": float(temp_vals.max()) if len(temp_vals) > 0 else None,
        "overall_mean": float(temp_vals.mean()) if len(temp_vals) > 0 else None,
    }

    # Issue breakdown
    issue_breakdown = {}
    if len(issues_df) > 0:
        issue_breakdown = issues_df["issue_type"].value_counts().to_dict()

    return {
        "n_batteries": int(df["battery_id"].nunique()),
        "n_cycles_total": int(len(df)),
        "n_discharge_cycles": int(len(discharge)),
        "n_issues": int(len(issues_df)),
        "issue_breakdown": issue_breakdown,
        "capacity_summary": capacity_summary,
        "temperature_summary": temperature_summary,
    }
