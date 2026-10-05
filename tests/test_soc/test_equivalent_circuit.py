"""Tests for the equivalent-circuit model."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.soc.equivalent_circuit import (
    ECMParameters,
    d_ocv_d_soc,
    ecm_voltage,
    ocv,
    simulate_ecm,
)


class TestOCV:
    def test_ocv_at_zero(self):
        assert ocv(0.0) == pytest.approx(3.0, abs=0.01)

    def test_ocv_at_full(self):
        assert ocv(1.0) == pytest.approx(4.25, abs=0.1)

    def test_ocv_monotonically_increasing(self):
        socs = np.linspace(0, 1, 100)
        voltages = [ocv(s) for s in socs]
        assert all(voltages[i] <= voltages[i + 1] for i in range(len(voltages) - 1))

    def test_ocv_clips_out_of_range(self):
        assert ocv(-0.5) == ocv(0.0)
        assert ocv(1.5) == ocv(1.0)


class TestDerivative:
    def test_derivative_positive(self):
        # OCV increases with SOC, so derivative should be positive
        for soc in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert d_ocv_d_soc(soc) > 0

    def test_derivative_at_midpoint(self):
        # dOCV/dSOC at SOC=0.5: 1.2 + 0.2*0.5 - 0.15*0.25 = 1.2 + 0.1 - 0.0375 = 1.2625
        assert d_ocv_d_soc(0.5) == pytest.approx(1.2625)


class TestECMVoltage:
    def test_voltage_less_than_ocv_during_discharge(self):
        params = ECMParameters()
        # During discharge (negative current), terminal voltage < OCV
        v_term = ecm_voltage(0.5, -2.0, 0.0, params)
        v_ocv = ocv(0.5)
        assert v_term < v_ocv

    def test_voltage_greater_than_ocv_during_charge(self):
        params = ECMParameters()
        # During charge (positive current), terminal voltage > OCV
        v_term = ecm_voltage(0.5, 2.0, 0.0, params)
        v_ocv = ocv(0.5)
        assert v_term > v_ocv

    def test_v1_reduces_voltage_during_discharge(self):
        params = ECMParameters()
        v_no_v1 = ecm_voltage(0.5, -2.0, 0.0, params)
        v_with_v1 = ecm_voltage(0.5, -2.0, 0.1, params)
        assert v_with_v1 < v_no_v1


class TestSimulateECM:
    def test_output_shapes(self):
        params = ECMParameters()
        soc, voltage, v1 = simulate_ecm(current=-2.0, dt=1.0, n_points=100, params=params)
        assert len(soc) == 100
        assert len(voltage) == 100
        assert len(v1) == 100

    def test_soc_decreases_during_discharge(self):
        params = ECMParameters()
        soc, _, _ = simulate_ecm(current=-2.0, dt=1.0, n_points=100, params=params)
        assert soc[0] > soc[-1]

    def test_soc_clipped_at_zero(self):
        params = ECMParameters()
        # Very long discharge should clip at 0
        soc, _, _ = simulate_ecm(current=-10.0, dt=1.0, n_points=10000, params=params)
        assert soc[-1] >= 0.0

    def test_v1_converges_steady_state(self):
        params = ECMParameters()
        # At steady state, V1 = I * R1
        _, _, v1 = simulate_ecm(current=-2.0, dt=1.0, n_points=10000, params=params)
        expected_v1 = 2.0 * params.R1
        assert v1[-1] == pytest.approx(expected_v1, rel=0.01)
