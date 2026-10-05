"""Tests for synthetic data generator."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.data.synthetic import generate_synthetic_battery, generate_synthetic_with_fault


class TestGenerateSyntheticBattery:
    def test_correct_number_of_cycles(self):
        cycles = generate_synthetic_battery(n_cycles=50)
        assert len(cycles) == 50

    def test_all_discharge(self):
        cycles = generate_synthetic_battery(n_cycles=10)
        assert all(c.cycle_type == "discharge" for c in cycles)

    def test_capacity_degrades(self):
        cycles = generate_synthetic_battery(n_cycles=100, degradation_rate=0.001)
        first_cap = cycles[0].capacity
        last_cap = cycles[-1].capacity
        assert last_cap < first_cap

    def test_reproducible_with_seed(self):
        c1 = generate_synthetic_battery(n_cycles=5, seed=42)
        c2 = generate_synthetic_battery(n_cycles=5, seed=42)
        assert c1[0].capacity == pytest.approx(c2[0].capacity)
        np.testing.assert_array_almost_equal(c1[0].voltage, c2[0].voltage)

    def test_voltage_in_safe_range(self):
        cycles = generate_synthetic_battery(n_cycles=10)
        for c in cycles:
            assert np.all(c.voltage >= 2.5)
            assert np.all(c.voltage <= 4.2)

    def test_synthetic_prefix_in_id(self):
        """Rule 8: Synthetic data must be clearly labelled."""
        cycles = generate_synthetic_battery(battery_id="SYNTH_001", n_cycles=5)
        assert cycles[0].battery_id.startswith("SYNTH_")


class TestGenerateSyntheticWithFault:
    def test_fault_injected(self):
        cycles = generate_synthetic_with_fault(n_cycles=100, fault_cycle=50)
        # Before fault: normal degradation
        # After fault: sudden drop
        cap_49 = cycles[48].capacity
        cap_50 = cycles[49].capacity
        assert cap_50 < cap_49 * 0.9  # At least 10% drop

    def test_temperature_elevated_after_fault(self):
        cycles = generate_synthetic_with_fault(n_cycles=100, fault_cycle=50)
        temp_before = np.mean(cycles[48].temperature)
        temp_after = np.mean(cycles[49].temperature)
        assert temp_after > temp_before + 10
