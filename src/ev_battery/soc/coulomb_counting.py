"""Coulomb counting SOC estimation.

Simple integration of current over time. Drifts due to measurement noise
and unknown initial SOC — this is why we need the EKF.
"""

from __future__ import annotations

import numpy as np


def coulomb_counting(
    current: np.ndarray,
    dt: float,
    initial_soc: float,
    capacity: float,
) -> np.ndarray:
    """Estimate SOC by Coulomb counting.

    Args:
        current: Current array in amps (negative = discharge).
        dt: Time step in seconds.
        initial_soc: Starting SOC [0, 1].
        capacity: Battery capacity in Coulombs.

    Returns:
        SOC array of same length as current.
    """
    n = len(current)
    soc = np.zeros(n)
    soc[0] = initial_soc

    for k in range(1, n):
        soc[k] = soc[k - 1] + (current[k] * dt) / capacity
        soc[k] = np.clip(soc[k], 0.0, 1.0)

    return soc
