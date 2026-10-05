"""Hybrid anomaly detection combining physics EKF residuals and Isolation Forest.

Rule 2: Features must never contain capacity or initial_capacity.
Rule 6: Fixed random seeds, store training metadata next to model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ev_battery.data.features import extract_features
from ev_battery.data.loader import CycleData
from ev_battery.soc.ekf import run_ekf_residuals
from ev_battery.soc.equivalent_circuit import ECMParameters
from ev_battery.soh.dataset import validate_no_target_leakage

# Safe non-leaky features for multivariate anomaly detection
ANOMALY_FEATURE_COLUMNS = [
    "voltage_drop_rate",
    "voltage_min",
    "temp_max",
    "temp_rise",
    "internal_resistance_proxy",
    "duration_s",
]


from ev_battery.soc.coulomb_counting import coulomb_counting


@dataclass
class EKFResidualMetrics:
    """Physics-based innovation residual metrics from Extended Kalman Filter."""

    mean_residual: float
    rms_residual: float
    max_residual: float
    is_anomaly: bool


def compute_cycle_ekf_residuals(
    cycle: CycleData,
    params: ECMParameters | None = None,
    dt: float | None = None,
    threshold_rms: float = 0.075,
    threshold_max: float = 0.15,
) -> EKFResidualMetrics:
    """Evaluate physics-based EKF innovation residuals for a discharge cycle.

    In a healthy cell, terminal voltage tracks ECM prediction within nominal model/sensor error.
    Sudden voltage drop, contact impedance surge, or cell degradation anomalies cause
    persistent innovations and peak residuals exceeding threshold.

    Args:
        cycle: Discharge CycleData record.
        params: ECMParameters (defaults to standard cell parameters).
        dt: Sample time step in seconds (computed from time array if None).
        threshold_rms: RMS residual threshold above which cycle is flagged anomalous.
        threshold_max: Maximum absolute residual threshold for sudden voltage spikes/drops.

    Returns:
        EKFResidualMetrics with summary statistics and anomaly flag.
    """
    if params is None:
        params = ECMParameters()

    t = np.asarray(cycle.time, dtype=float)
    v = np.asarray(cycle.voltage, dtype=float)
    i = np.asarray(cycle.current, dtype=float)

    if len(t) < 2:
        return EKFResidualMetrics(0.0, 0.0, 0.0, False)

    dt_step = float(t[1] - t[0]) if dt is None else dt
    if dt_step <= 0:
        dt_step = 1.0

    _, residuals = run_ekf_residuals(
        current=i,
        voltage_meas=v,
        dt=dt_step,
        params=params,
        initial_soc=0.9,
    )

    # Discard initial convergence transient (first 5% of cycle points)
    trim_idx = max(1, int(len(residuals) * 0.05))
    steady_residuals = residuals[trim_idx:]

    mean_res = float(np.mean(steady_residuals))
    rms_res = float(np.sqrt(np.mean(steady_residuals**2)))
    max_res = float(np.max(np.abs(steady_residuals)))

    is_anom = bool(rms_res > threshold_rms or max_res > threshold_max)

    return EKFResidualMetrics(
        mean_residual=mean_res,
        rms_residual=rms_res,
        max_residual=max_res,
        is_anomaly=is_anom,
    )


def compute_dataset_sha256(X: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 hash of training data (Rule 6)."""
    hasher = hashlib.sha256()
    hasher.update(",".join(X.columns).encode("utf-8"))
    hasher.update(X.to_numpy().tobytes())
    return hasher.hexdigest()


class IsolationForestDetector:
    """Unsupervised multivariate anomaly detector using Isolation Forest.

    Learns the normal distribution of cycle telemetry features (thermal rise,
    internal resistance proxy, duration, voltage curve shape) and isolates outliers.
    """

    def __init__(
        self,
        contamination: float | str = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names = (
            feature_names if feature_names is not None else list(ANOMALY_FEATURE_COLUMNS)
        )

        validate_no_target_leakage(self.feature_names)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.is_fitted = False
        self.training_metadata: dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        dataset_version: str = "v1.0-normal-cycles",
    ) -> IsolationForestDetector:
        """Fit detector on normal operating cycle features."""
        validate_no_target_leakage(list(X.columns))
        X_df = X[self.feature_names].copy()

        self.model.fit(X_df)
        self.is_fitted = True

        import sklearn

        self.training_metadata = {
            "dataset_version": dataset_version,
            "data_sha256": compute_dataset_sha256(X_df),
            "feature_list": self.feature_names,
            "random_seed": self.random_state,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
            },
            "library_versions": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict whether cycles are anomalous.

        Returns:
            Boolean array: True if anomalous, False if normal.
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before predict().")
        validate_no_target_leakage(list(X.columns))
        X_df = X[self.feature_names].copy()
        raw_preds = self.model.predict(X_df)
        # IsolationForest returns -1 for outliers/anomalies, 1 for inliers/normal
        return raw_preds == -1

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw anomaly score (lower values indicate higher abnormality)."""
        if not self.is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before decision_function().")
        validate_no_target_leakage(list(X.columns))
        X_df = X[self.feature_names].copy()
        return self.model.decision_function(X_df)


class HybridAnomalyDetector:
    """Unified hybrid detector fusing physics residuals and ML isolation forest."""

    def __init__(
        self,
        ml_detector: IsolationForestDetector,
        residual_threshold_rms: float = 0.075,
        ecm_params: ECMParameters | None = None,
    ) -> None:
        self.ml_detector = ml_detector
        self.residual_threshold_rms = residual_threshold_rms
        self.ecm_params = ecm_params if ecm_params is not None else ECMParameters()

    def diagnose_cycle(self, cycle: CycleData) -> dict[str, Any]:
        """Diagnose a single cycle using both physics and ML criteria.

        Returns:
            Dictionary containing is_anomaly, anomaly_type, physics_metrics,
            and ml_decision_score.
        """
        # 1. Physics: EKF residuals
        physics_metrics = compute_cycle_ekf_residuals(
            cycle=cycle,
            params=self.ecm_params,
            threshold_rms=self.residual_threshold_rms,
        )
        physics_flag = physics_metrics.is_anomaly

        # 2. ML: Isolation Forest
        feats = extract_features(cycle)
        feats_df = pd.DataFrame([feats])
        ml_flag = bool(self.ml_detector.predict(feats_df)[0])
        ml_score = float(self.ml_detector.decision_function(feats_df)[0])

        # 3. Hybrid fusion
        is_anom = physics_flag or ml_flag

        if physics_flag and ml_flag:
            anomaly_type = "CRITICAL_HYBRID_ANOMALY"
        elif physics_flag:
            anomaly_type = "PHYSICS_RESIDUAL_SURGE"
        elif ml_flag:
            anomaly_type = "MULTIVARIATE_FEATURE_DRIFT"
        else:
            anomaly_type = "NORMAL"

        return {
            "cycle_number": cycle.cycle_number,
            "battery_id": cycle.battery_id,
            "is_anomaly": is_anom,
            "anomaly_type": anomaly_type,
            "physics_flag": physics_flag,
            "ml_flag": ml_flag,
            "rms_residual_v": physics_metrics.rms_residual,
            "max_residual_v": physics_metrics.max_residual,
            "ml_anomaly_score": ml_score,
        }
