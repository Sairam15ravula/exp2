"""Tests for the Extended Kalman Filter."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.soc.ekf import EKF, run_ekf
from ev_battery.soc.equivalent_circuit import ECMParameters, simulate_ecm
from ev_battery.soc.evaluation import convergence_time, evaluate_soc


class TestEKFConvergence:
    """EKF should converge from a wrong initial SOC."""

    def test_converges_from_wrong_initial(self):
        params = ECMParameters()
        dt = 1.0
        true_soc_initial = 0.8
        wrong_initial = 0.3

        # Simulate ground truth
        true_soc, voltage, _ = simulate_ecm(
            current=-2.0, dt=dt, n_points=500, params=params, initial_soc=true_soc_initial
        )

        # Add noise to voltage measurements
        rng = np.random.default_rng(42)
        voltage_meas = voltage + rng.normal(0, 0.01, len(voltage))

        # Run EKF with wrong initial SOC
        estimates = run_ekf(
            current=np.full(500, -2.0),
            voltage_meas=voltage_meas,
            dt=dt,
            params=params,
            initial_soc=wrong_initial,
        )

        # Should converge to true SOC
        final_error = abs(estimates[-1] - true_soc[-1])
        assert final_error < 0.02

    def test_convergence_time_reasonable(self):
        params = ECMParameters()
        dt = 1.0
        true_soc, voltage, _ = simulate_ecm(
            current=-2.0, dt=dt, n_points=500, params=params, initial_soc=0.8
        )
        rng = np.random.default_rng(42)
        voltage_meas = voltage + rng.normal(0, 0.01, len(voltage))

        estimates = run_ekf(
            current=np.full(500, -2.0),
            voltage_meas=voltage_meas,
            dt=dt,
            params=params,
            initial_soc=0.3,
        )

        conv_time = convergence_time(estimates, true_soc, dt, threshold=0.02)
        assert conv_time > 0
        assert conv_time < 200  # Should converge within 200 seconds


class TestEKFVsCoulombCounting:
    """EKF should outperform Coulomb counting with noisy measurements."""

    def test_ekf_beats_coulomb_counting(self):
        from ev_battery.soc.coulomb_counting import coulomb_counting

        params = ECMParameters()
        dt = 1.0
        n_points = 500
        true_soc, voltage, _ = simulate_ecm(
            current=-2.0, dt=dt, n_points=n_points, params=params, initial_soc=0.8
        )

        # Add noise to current (both methods use this)
        rng = np.random.default_rng(42)
        noisy_current = np.full(n_points, -2.0) + rng.normal(0, 0.5, n_points)

        # Add noise to voltage (EKF uses this)
        voltage_meas = voltage + rng.normal(0, 0.01, n_points)

        # Coulomb counting with noisy current and WRONG initial SOC
        cc_soc = coulomb_counting(noisy_current, dt, 0.5, params.capacity)

        # EKF with same noisy current and correct initial SOC
        ekf_soc = run_ekf(
            current=noisy_current,
            voltage_meas=voltage_meas,
            dt=dt,
            params=params,
            initial_soc=0.8,
        )

        cc_metrics = evaluate_soc(cc_soc, true_soc)
        ekf_metrics = evaluate_soc(ekf_soc, true_soc)

        # EKF should have lower RMSE
        assert ekf_metrics["rmse"] < cc_metrics["rmse"]


class TestEKFState:
    """EKF state should behave correctly."""

    def test_initial_state(self):
        params = ECMParameters()
        ekf = EKF(params, dt=1.0, initial_soc=0.6)
        assert ekf.get_soc() == pytest.approx(0.6)
        assert ekf.get_v1() == pytest.approx(0.0)

    def test_predict_changes_state(self):
        params = ECMParameters()
        ekf = EKF(params, dt=1.0, initial_soc=0.5)
        soc_before = ekf.get_soc()
        ekf.predict(current=-2.0)
        soc_after = ekf.get_soc()
        assert soc_after < soc_before  # Discharge reduces SOC

    def test_update_corrects_state(self):
        params = ECMParameters()
        ekf = EKF(params, dt=1.0, initial_soc=0.5)
        # If measured voltage is lower than predicted, SOC should decrease
        ekf.predict(current=-2.0)
        soc_before = ekf.get_soc()
        # Artificially low voltage -> EKF thinks battery is more discharged
        ekf.update(voltage_meas=2.5, current=-2.0)
        soc_after = ekf.get_soc()
        assert soc_after < soc_before
