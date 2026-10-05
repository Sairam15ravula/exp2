"""Tests for data validation."""

from __future__ import annotations

import numpy as np

from ev_battery.data.loader import CycleData
from ev_battery.data.validation import validate_cycle, validate_dataset


def _make_valid_cycle(battery_id: str = "B0005", cycle_number: int = 1) -> CycleData:
    return CycleData(
        battery_id=battery_id,
        cycle_number=cycle_number,
        cycle_type="discharge",
        time=np.linspace(0, 3600, 10),
        voltage=np.linspace(4.2, 3.0, 10),
        current=np.full(10, -2.0),
        temperature=np.full(10, 25.0),
        capacity=2.0,
    )


class TestValidateCycle:
    def test_valid_cycle_no_issues(self):
        cycle = _make_valid_cycle()
        issues = validate_cycle(cycle)
        assert len(issues) == 0

    def test_empty_time_array(self):
        cycle = _make_valid_cycle()
        cycle.time = np.array([])
        issues = validate_cycle(cycle)
        assert any(i["issue_type"] == "empty_time" for i in issues)

    def test_voltage_out_of_range(self):
        cycle = _make_valid_cycle()
        cycle.voltage = np.array([5.0] * 10)
        issues = validate_cycle(cycle)
        assert any(i["issue_type"] == "voltage_out_of_range" for i in issues)

    def test_temperature_out_of_range(self):
        cycle = _make_valid_cycle()
        cycle.temperature = np.array([150.0] * 10)
        issues = validate_cycle(cycle)
        assert any(i["issue_type"] == "temperature_out_of_range" for i in issues)

    def test_negative_capacity(self):
        cycle = _make_valid_cycle()
        cycle.capacity = -1.0
        issues = validate_cycle(cycle)
        assert any(i["issue_type"] == "negative_capacity" for i in issues)

    def test_array_length_mismatch(self):
        cycle = _make_valid_cycle()
        cycle.voltage = np.array([4.2, 3.8])  # shorter than time
        issues = validate_cycle(cycle)
        assert any(i["issue_type"] == "array_length_mismatch" for i in issues)

    def test_non_finite_voltage(self):
        cycle = _make_valid_cycle()
        cycle.voltage = np.array([np.nan] * 10)
        issues = validate_cycle(cycle)
        assert any(i["issue_type"] == "non_finite_voltage" for i in issues)


class TestValidateDataset:
    def test_clean_dataset(self):
        cycles = [_make_valid_cycle("B0005", i) for i in range(1, 4)]
        df = validate_dataset(cycles)
        assert len(df) == 0

    def test_issues_found(self):
        bad_cycle = _make_valid_cycle("B0005", 1)
        bad_cycle.voltage = np.array([5.0] * 10)
        good_cycle = _make_valid_cycle("B0005", 2)
        df = validate_dataset([bad_cycle, good_cycle])
        assert len(df) == 1
        assert df.iloc[0]["battery_id"] == "B0005"
        assert df.iloc[0]["issue_type"] == "voltage_out_of_range"
