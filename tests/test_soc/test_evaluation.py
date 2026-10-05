"""Tests for SOC evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.soc.evaluation import convergence_time, evaluate_soc, max_error, mae, rmse


class TestMetrics:
    def test_rmse_perfect(self):
        truth = np.array([0.5, 0.6, 0.7])
        est = np.array([0.5, 0.6, 0.7])
        assert rmse(est, truth) == pytest.approx(0.0)

    def test_rmse_known_error(self):
        truth = np.array([0.5, 0.5, 0.5])
        est = np.array([0.6, 0.6, 0.6])
        # RMSE = sqrt(mean(0.01)) = 0.1
        assert rmse(est, truth) == pytest.approx(0.1)

    def test_mae_known_error(self):
        truth = np.array([0.5, 0.5, 0.5])
        est = np.array([0.6, 0.4, 0.5])
        # MAE = mean(0.1, 0.1, 0.0) = 0.0667
        assert mae(est, truth) == pytest.approx(0.0667, rel=0.01)

    def test_max_error(self):
        truth = np.array([0.5, 0.5, 0.5])
        est = np.array([0.5, 0.7, 0.5])
        assert max_error(est, truth) == pytest.approx(0.2)

    def test_evaluate_soc_returns_all_metrics(self):
        truth = np.array([0.5, 0.6, 0.7])
        est = np.array([0.55, 0.65, 0.75])
        metrics = evaluate_soc(est, truth)
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "max_error" in metrics


class TestConvergenceTime:
    def test_converges_immediately(self):
        truth = np.full(100, 0.5)
        est = np.full(100, 0.5)
        assert convergence_time(est, truth, dt=1.0) == 0.0

    def test_converges_after_delay(self):
        truth = np.full(100, 0.5)
        est = np.full(100, 0.8)
        est[50:] = 0.5  # Converges at t=50
        assert convergence_time(est, truth, dt=1.0) == 50.0

    def test_never_converges(self):
        truth = np.full(100, 0.5)
        est = np.full(100, 0.8)
        assert convergence_time(est, truth, dt=1.0) == -1.0
