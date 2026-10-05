"""Tests for physics-based EKF innovation residuals."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.anomaly.detector import (
    EKFResidualMetrics,
    compute_cycle_ekf_residuals,
)
from ev_battery.data.loader import CycleData
from ev_battery.soc.equivalent_circuit import ECMParameters, simulate_ecm


class TestEKFResiduals:
    def test_normal_cycle_has_low_residuals(self):
        """Under normal simulated ECM physics with low noise, residuals are small."""
        params = ECMParameters()
        dt = 1.0
        n_points = 300
        true_soc, voltage, _ = simulate_ecm(
            current=-2.0, dt=dt, n_points=n_points, params=params, initial_soc=0.9
        )
        rng = np.random.default_rng(42)
        voltage_meas = voltage + rng.normal(0, 0.005, n_points)

        cycle = CycleData(
            battery_id="SYNTH_NORMAL",
            cycle_number=1,
            cycle_type="discharge",
            time=np.arange(n_points) * dt,
            voltage=voltage_meas,
            current=np.full(n_points, -2.0),
            temperature=np.full(n_points, 25.0),
            capacity=2.0,
        )

        metrics = compute_cycle_ekf_residuals(cycle, params=params, dt=dt)
        assert isinstance(metrics, EKFResidualMetrics)
        # Steady RMS residual should be below normal threshold
        assert metrics.rms_residual < 0.035
        assert not metrics.is_anomaly

    def test_faulted_voltage_sag_triggers_high_residual(self):
        """A sudden 0.15V voltage drop (internal short precursor) causes residual spike."""
        params = ECMParameters()
        dt = 1.0
        n_points = 300
        true_soc, voltage, _ = simulate_ecm(
            current=-2.0, dt=dt, n_points=n_points, params=params, initial_soc=0.9
        )
        # Inject unexpected voltage sag midway (internal short onset)
        faulted_voltage = voltage.copy()
        faulted_voltage[150:] -= 0.20

        cycle = CycleData(
            battery_id="SYNTH_FAULT",
            cycle_number=50,
            cycle_type="discharge",
            time=np.arange(n_points) * dt,
            voltage=faulted_voltage,
            current=np.full(n_points, -2.0),
            temperature=np.full(n_points, 45.0),
            capacity=1.7,
        )

        metrics = compute_cycle_ekf_residuals(cycle, params=params, dt=dt, threshold_max=0.15)
        assert metrics.max_residual >= 0.15
        assert metrics.is_anomaly

    def test_short_cycle_handled_gracefully(self):
        """Cycles with < 2 points return zero metrics without raising exception."""
        cycle = CycleData(
            battery_id="SYNTH_SHORT",
            cycle_number=1,
            cycle_type="discharge",
            time=np.array([0.0]),
            voltage=np.array([3.8]),
            current=np.array([-2.0]),
            temperature=np.array([25.0]),
            capacity=2.0,
        )
        metrics = compute_cycle_ekf_residuals(cycle)
        assert metrics.rms_residual == 0.0
        assert not metrics.is_anomaly
