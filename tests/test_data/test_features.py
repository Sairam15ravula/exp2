"""Tests for feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.data.features import extract_all_features, extract_features
from ev_battery.data.loader import CycleData


def _make_discharge_cycle(
    voltage_start: float = 4.2,
    voltage_end: float = 3.0,
    n_points: int = 100,
    temp_start: float = 25.0,
    temp_end: float = 35.0,
) -> CycleData:
    t = np.linspace(0, 3600, n_points)
    v = np.linspace(voltage_start, voltage_end, n_points)
    curr = np.full(n_points, -2.0)
    temp = np.linspace(temp_start, temp_end, n_points)
    return CycleData(
        battery_id="B0005",
        cycle_number=1,
        cycle_type="discharge",
        time=t,
        voltage=v,
        current=curr,
        temperature=temp,
        capacity=2.0,
    )


class TestExtractFeatures:
    def test_voltage_features(self):
        cycle = _make_discharge_cycle(voltage_start=4.2, voltage_end=3.0)
        feats = extract_features(cycle)
        assert feats["voltage_max"] == pytest.approx(4.2, abs=0.01)
        assert feats["voltage_min"] == pytest.approx(3.0, abs=0.01)
        assert feats["voltage_mean"] == pytest.approx(3.6, abs=0.1)

    def test_temperature_features(self):
        cycle = _make_discharge_cycle(temp_start=25.0, temp_end=35.0)
        feats = extract_features(cycle)
        assert feats["temp_max"] == pytest.approx(35.0, abs=0.1)
        assert feats["temp_min"] == pytest.approx(25.0, abs=0.1)
        assert feats["temp_rise"] == pytest.approx(10.0, abs=0.2)

    def test_duration(self):
        cycle = _make_discharge_cycle()
        feats = extract_features(cycle)
        assert feats["duration_s"] == pytest.approx(3600.0)

    def test_internal_resistance_proxy_positive(self):
        cycle = _make_discharge_cycle(voltage_start=4.2, voltage_end=3.0)
        feats = extract_features(cycle)
        # Voltage drops from 4.2 to ~3.6 at midpoint, current is 2A
        # IR proxy = (4.2 - 3.6) / 2.0 = 0.3
        assert feats["internal_resistance_proxy"] > 0

    def test_time_in_voltage_window(self):
        cycle = _make_discharge_cycle(voltage_start=4.2, voltage_end=3.0)
        feats = extract_features(cycle)
        # Most of the cycle is between 3.0V and 4.2V
        assert feats["time_in_voltage_window"] > 0

    def test_no_capacity_leakage(self):
        """Rule 2: Features must NOT include capacity or initial_capacity."""
        cycle = _make_discharge_cycle()
        feats = extract_features(cycle)
        assert "capacity" not in feats
        assert "initial_capacity" not in feats


class TestExtractAllFeatures:
    def test_returns_dataframe(self):
        from ev_battery.data.synthetic import generate_synthetic_battery
        cycles = generate_synthetic_battery(n_cycles=5)
        df = extract_all_features(cycles)
        assert len(df) == 5
        assert "battery_id" in df.columns
        assert "cycle_number" in df.columns
        assert "voltage_max" in df.columns

    def test_skips_non_discharge(self):
        cycles = [
            CycleData("B0005", 1, "discharge", [0, 1], [4.2, 3.0], [-2, -2], [25, 26], 2.0),
            CycleData("B0005", 2, "charge", [0, 1], [3.0, 4.2], [2, 2], [25, 26], None),
        ]
        df = extract_all_features(cycles)
        assert len(df) == 1
