"""Equivalent-circuit model (ECM) for Li-ion cells.

Uses a 1RC model: OCV(SOC) - I*R0 - V1
where V1 is the voltage across an RC parallel branch (R1, C1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ECMParameters:
    """Parameters for the 1RC equivalent-circuit model.

    All values are cell-level (rule 5). Capacity in Coulombs.
    """
    capacity: float = 7200.0  # 2 Ah in Coulombs
    R0: float = 0.05  # Ohmic resistance (ohm)
    R1: float = 0.02  # Polarization resistance (ohm)
    C1: float = 2000.0  # Polarization capacitance (F)


def ocv(soc: float) -> float:
    """Open-circuit voltage as a function of SOC.

    Polynomial approximation for a typical Li-ion cell.
    SOC in [0, 1], returns voltage in volts.
    """
    soc = np.clip(soc, 0.0, 1.0)
    return 3.0 + 1.2 * soc + 0.1 * soc**2 - 0.05 * soc**3


def d_ocv_d_soc(soc: float) -> float:
    """Derivative of OCV with respect to SOC (for EKF Jacobian)."""
    soc = np.clip(soc, 0.0, 1.0)
    return 1.2 + 0.2 * soc - 0.15 * soc**2


def ecm_voltage(soc: float, current: float, v1: float, params: ECMParameters) -> float:
    """Terminal voltage from the 1RC ECM.

    Args:
        soc: State of charge [0, 1].
        current: Current in amps (positive = charge, negative = discharge).
        v1: RC branch voltage in volts.
        params: ECM parameters.

    Returns:
        Terminal voltage in volts.
    """
    return ocv(soc) + current * params.R0 - v1


def simulate_ecm(
    current: float,
    dt: float,
    n_points: int,
    params: ECMParameters,
    initial_soc: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a discharge cycle using the ECM model.

    Args:
        current: Constant current in amps (negative for discharge).
        dt: Time step in seconds.
        n_points: Number of time points.
        params: ECM parameters.
        initial_soc: Starting SOC.

    Returns:
        (soc, voltage, v1) arrays of length n_points.
    """
    soc = np.zeros(n_points)
    voltage = np.zeros(n_points)
    v1_arr = np.zeros(n_points)

    soc[0] = initial_soc
    v1_arr[0] = 0.0
    voltage[0] = ecm_voltage(soc[0], current, 0.0, params)

    alpha = np.exp(-dt / (params.R1 * params.C1))

    for k in range(1, n_points):
        soc[k] = soc[k - 1] + (current * dt) / params.capacity
        soc[k] = np.clip(soc[k], 0.0, 1.0)
        v1_arr[k] = alpha * v1_arr[k - 1] - params.R1 * (1 - alpha) * current
        voltage[k] = ecm_voltage(soc[k], current, v1_arr[k], params)

    return soc, voltage, v1_arr
