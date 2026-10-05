"""SOC estimation evaluation metrics."""

from __future__ import annotations

import numpy as np


def rmse(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((estimates - ground_truth) ** 2)))


def mae(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(estimates - ground_truth)))


def max_error(estimates: np.ndarray, ground_truth: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(estimates - ground_truth)))


def evaluate_soc(estimates: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """Compute all SOC evaluation metrics.

    Returns:
        Dictionary with rmse, mae, max_error (all in SOC fraction, 0-1).
    """
    return {
        "rmse": rmse(estimates, ground_truth),
        "mae": mae(estimates, ground_truth),
        "max_error": max_error(estimates, ground_truth),
    }


def convergence_time(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    dt: float,
    threshold: float = 0.02,
) -> float:
    """Time for SOC error to drop below threshold and stay there.

    Args:
        estimates: SOC estimates.
        ground_truth: True SOC values.
        dt: Time step in seconds.
        threshold: Error threshold in SOC fraction.

    Returns:
        Convergence time in seconds. Returns -1 if never converges.
    """
    errors = np.abs(estimates - ground_truth)
    below = errors < threshold

    # Find first index where error drops below threshold and stays there
    for i in range(len(below)):
        if below[i] and np.all(below[i:]):
            return float(i * dt)

    return -1.0
