"""Extended Kalman Filter for SOC estimation.

State: [SOC, V1] where V1 is the RC branch voltage.
Measurement: terminal voltage from the ECM.
"""

from __future__ import annotations

import numpy as np

from ev_battery.soc.equivalent_circuit import (
    ECMParameters,
    d_ocv_d_soc,
    ecm_voltage,
    ocv,
)


class EKF:
    """Extended Kalman Filter for SOC estimation.

    Combines the ECM (prediction) with voltage measurements (correction)
    to produce a stable SOC estimate that converges from wrong initial values.
    """

    def __init__(
        self,
        params: ECMParameters,
        dt: float,
        initial_soc: float = 0.5,
        process_noise: np.ndarray | None = None,
        measurement_noise: float = 1e-4,
    ):
        """Initialize EKF.

        Args:
            params: ECM parameters.
            dt: Time step in seconds.
            initial_soc: Initial SOC guess (can be wrong — EKF will converge).
            process_noise: 2x2 process noise covariance matrix.
            measurement_noise: Measurement noise variance (volts^2).
        """
        self.params = params
        self.dt = dt
        self.Q = process_noise if process_noise is not None else np.diag([1e-6, 1e-6])
        self.R = measurement_noise

        # State: [SOC, V1]
        self.x = np.array([initial_soc, 0.0])

        # State covariance
        self.P = np.diag([0.01, 0.001])

        # State transition matrix (constant for fixed dt)
        alpha = np.exp(-dt / (params.R1 * params.C1))
        self.F = np.array([[1.0, 0.0], [0.0, alpha]])

        # Input matrix for current
        self.B = np.array([dt / params.capacity, -params.R1 * (1 - alpha)])

        # Voltage innovation residual
        self.last_innovation: float = 0.0

    def predict(self, current: float) -> None:
        """Prediction step: advance state using current measurement."""
        self.x = self.F @ self.x + self.B * current
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, voltage_meas: float, current: float) -> None:
        """Update step: correct state using voltage measurement."""
        # Measurement Jacobian
        d_ocv = d_ocv_d_soc(self.x[0])
        H = np.array([d_ocv, -1.0])

        # Predicted measurement
        v_pred = ecm_voltage(self.x[0], current, self.x[1], self.params)

        # Innovation
        y = voltage_meas - v_pred
        self.last_innovation = float(y)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T / S

        # State update
        self.x = self.x + K * y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.R

    def get_soc(self) -> float:
        """Return current SOC estimate."""
        return float(self.x[0])

    def get_v1(self) -> float:
        """Return current RC branch voltage estimate."""
        return float(self.x[1])

    def get_innovation(self) -> float:
        """Return the latest voltage innovation residual (y = V_meas - V_pred)."""
        return self.last_innovation


def run_ekf(
    current: np.ndarray,
    voltage_meas: np.ndarray,
    dt: float,
    params: ECMParameters,
    initial_soc: float = 0.5,
    measurement_noise: float = 1e-4,
) -> np.ndarray:
    """Run EKF on a full cycle and return SOC estimates.

    Args:
        current: Current array in amps.
        voltage_meas: Measured terminal voltage array in volts.
        dt: Time step in seconds.
        params: ECM parameters.
        initial_soc: Initial SOC guess.
        measurement_noise: Measurement noise variance.

    Returns:
        SOC estimate array of same length as input.
    """
    ekf = EKF(params, dt, initial_soc=initial_soc, measurement_noise=measurement_noise)
    n = len(current)
    soc_estimates = np.zeros(n)

    for k in range(n):
        ekf.predict(current[k])
        ekf.update(voltage_meas[k], current[k])
        soc_estimates[k] = ekf.get_soc()

    return soc_estimates


def run_ekf_residuals(
    current: np.ndarray,
    voltage_meas: np.ndarray,
    dt: float,
    params: ECMParameters,
    initial_soc: float = 0.5,
    measurement_noise: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run EKF on a full cycle and return both SOC estimates and voltage innovation residuals.

    Args:
        current: Current array in amps.
        voltage_meas: Measured terminal voltage array in volts.
        dt: Time step in seconds.
        params: ECM parameters.
        initial_soc: Initial SOC guess.
        measurement_noise: Measurement noise variance.

    Returns:
        (soc_estimates, innovation_residuals)
    """
    ekf = EKF(params, dt, initial_soc=initial_soc, measurement_noise=measurement_noise)
    n = len(current)
    soc_estimates = np.zeros(n)
    residuals = np.zeros(n)

    for k in range(n):
        ekf.predict(current[k])
        ekf.update(voltage_meas[k], current[k])
        soc_estimates[k] = ekf.get_soc()
        residuals[k] = ekf.get_innovation()

    return soc_estimates, residuals
