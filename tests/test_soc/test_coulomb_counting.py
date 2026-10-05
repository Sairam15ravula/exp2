"""Tests for Coulomb counting."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.soc.coulomb_counting import coulomb_counting


class TestCoulombCounting:
    def test_constant_discharge(self):
        # 2A discharge for 3600s from 2Ah battery -> SOC drops by 1.0
        current = np.full(100, -2.0)
        soc = coulomb_counting(current, dt=36.0, initial_soc=1.0, capacity=7200.0)
        assert soc[0] == 1.0
        assert soc[-1] == pytest.approx(0.0, abs=0.01)

    def test_no_current_no_change(self):
        current = np.zeros(100)
        soc = coulomb_counting(current, dt=1.0, initial_soc=0.5, capacity=7200.0)
        assert np.allclose(soc, 0.5)

    def test_charge_increases_soc(self):
        current = np.full(100, 2.0)  # charging
        soc = coulomb_counting(current, dt=36.0, initial_soc=0.5, capacity=7200.0)
        assert soc[-1] > 0.5

    def test_soc_clipped(self):
        # Over-discharge should clip at 0
        current = np.full(100, -10.0)
        soc = coulomb_counting(current, dt=36.0, initial_soc=0.5, capacity=7200.0)
        assert np.all(soc >= 0.0)
        assert np.all(soc <= 1.0)

    def test_accuracy_with_small_dt(self):
        # With small time steps, Coulomb counting should be accurate
        current = np.full(1000, -2.0)
        soc = coulomb_counting(current, dt=3.6, initial_soc=1.0, capacity=7200.0)
        # Total charge removed: 2.0 * 3600 = 7200 C = full capacity
        assert soc[-1] == pytest.approx(0.0, abs=0.001)
