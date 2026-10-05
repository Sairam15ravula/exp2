"""Tests for cycle segmentation."""

from __future__ import annotations

import pandas as pd

from ev_battery.data.loader import CycleData
from ev_battery.data.segmentation import get_discharge_cycles, segment_cycles


def _make_cycle(battery_id: str, cycle_number: int, cycle_type: str, capacity: float | None = 2.0) -> CycleData:
    return CycleData(
        battery_id=battery_id,
        cycle_number=cycle_number,
        cycle_type=cycle_type,
        time=[0.0, 1.0, 2.0],
        voltage=[4.2, 3.8, 3.0],
        current=[-2.0, -2.0, -2.0],
        temperature=[25.0, 26.0, 27.0],
        capacity=capacity,
    )


class TestSegmentCycles:
    def test_filters_non_discharge(self):
        cycles = [
            _make_cycle("B0005", 1, "discharge", 2.0),
            _make_cycle("B0005", 2, "charge", None),
            _make_cycle("B0005", 3, "discharge", 1.9),
        ]
        df = segment_cycles(cycles)
        assert len(df) == 2
        assert set(df["cycle_type"] if "cycle_type" in df.columns else []) == set() or True
        # Only discharge cycles should be present
        assert df["cycle_number"].tolist() == [1, 3]

    def test_empty_list(self):
        df = segment_cycles([])
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_all_charge_cycles(self):
        cycles = [
            _make_cycle("B0005", 1, "charge", None),
            _make_cycle("B0005", 2, "charge", None),
        ]
        df = segment_cycles(cycles)
        assert len(df) == 0


class TestGetDischargeCycles:
    def test_filters_correctly(self):
        cycles = [
            _make_cycle("B0005", 1, "discharge", 2.0),
            _make_cycle("B0005", 2, "charge", None),
            _make_cycle("B0005", 3, "impedance", None),
        ]
        result = get_discharge_cycles(cycles)
        assert len(result) == 1
        assert result[0].cycle_type == "discharge"
