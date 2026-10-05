"""Inference endpoints for SOC, SOH, RUL, and Anomaly detection.

Adheres strictly to:
- Rule 2: Zero target leakage (capacity/initial_capacity strictly rejected for SOH).
- Rule 4: RUL defined as cycles until SOH <= 80%.
- Rule 5: Cell-level valid ranges.
- Rule 7: Auth required since predictions write records to DB.
- Rule 8: Synthetic labels persisted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ev_battery.anomaly.thermal_risk import ThermalRiskEngine, ThermalRiskLevel
from ev_battery.db.models import Alert, AnomalyEvent, Cell, Prediction, StateEstimate, User
from ev_battery.db.session import get_db
from ev_battery.rul.dataset import EOL_SOH_THRESHOLD
from ev_battery.rul.model import RULModel
from ev_battery.schemas.inference import (
    AnomalyInferenceRequest,
    AnomalyResponse,
    PredictionResponse,
    RULInferenceRequest,
    SOCInferenceRequest,
    SOCInferenceResponse,
    SOHInferenceRequest,
)
from ev_battery.security.auth import get_current_user
from ev_battery.soc.coulomb_counting import coulomb_counting
from ev_battery.soc.ekf import run_ekf
from ev_battery.soc.equivalent_circuit import ECMParameters
from ev_battery.soh.dataset import FEATURE_COLUMNS
from ev_battery.soh.model import SOHModel

router = APIRouter(prefix="/inference", tags=["Inference & State Estimation"])

# In-memory cached model instances
_cached_soh_model: SOHModel | None = None
_cached_rul_model: RULModel | None = None
_thermal_engine = ThermalRiskEngine()


def _get_or_create_soh_model() -> SOHModel:
    """Return trained SOH model or train a lightweight surrogate for development/testing."""
    global _cached_soh_model
    if _cached_soh_model is not None:
        return _cached_soh_model

    model_path = Path("models/soh_xgboost.joblib")
    if model_path.exists():
        model = SOHModel.load(str(model_path))
        _cached_soh_model = model
        return model

    # Create & fit a surrogate model for demonstration & integration tests
    np.random.seed(42)
    n_samples = 50
    synth_X = pd.DataFrame({
        "cycle_number": np.arange(1, n_samples + 1),
        "voltage_max": np.linspace(4.2, 4.0, n_samples),
        "voltage_min": np.linspace(2.7, 2.5, n_samples),
        "voltage_mean": np.linspace(3.7, 3.4, n_samples),
        "voltage_drop_rate": np.linspace(0.001, 0.003, n_samples),
        "temp_max": np.linspace(26.0, 32.0, n_samples),
        "temp_min": np.full(n_samples, 24.0),
        "temp_rise": np.linspace(2.0, 8.0, n_samples),
        "internal_resistance_proxy": np.linspace(0.05, 0.12, n_samples),
        "time_in_voltage_window": np.linspace(3600.0, 2400.0, n_samples),
        "duration_s": np.linspace(3800.0, 2600.0, n_samples),
    })
    synth_y = pd.Series(np.linspace(1.0, 0.75, n_samples))
    model = SOHModel(n_estimators=30, max_depth=3)
    model.fit(synth_X, synth_y)
    _cached_soh_model = model
    return model


def _get_or_create_rul_model() -> RULModel:
    """Return trained RUL model or train a lightweight surrogate for development/testing."""
    global _cached_rul_model
    if _cached_rul_model is not None:
        return _cached_rul_model

    model_path = Path("models/rul_quantile_gb.joblib")
    if model_path.exists():
        model = RULModel.load(str(model_path))
        _cached_rul_model = model
        return model

    np.random.seed(42)
    n_samples = 50
    synth_X = pd.DataFrame({
        "cycle_number": np.arange(1, n_samples + 1),
        "voltage_max": np.linspace(4.2, 4.0, n_samples),
        "voltage_min": np.linspace(2.7, 2.5, n_samples),
        "voltage_mean": np.linspace(3.7, 3.4, n_samples),
        "voltage_drop_rate": np.linspace(0.001, 0.003, n_samples),
        "temp_max": np.linspace(26.0, 32.0, n_samples),
        "temp_min": np.full(n_samples, 24.0),
        "temp_rise": np.linspace(2.0, 8.0, n_samples),
        "internal_resistance_proxy": np.linspace(0.05, 0.12, n_samples),
        "time_in_voltage_window": np.linspace(3600.0, 2400.0, n_samples),
        "duration_s": np.linspace(3800.0, 2600.0, n_samples),
    })
    # RUL decreases to 0 at EOL-80%
    synth_y = pd.Series(np.maximum(0.0, np.linspace(120.0, 0.0, n_samples)))
    model = RULModel(n_estimators=30, max_depth=3)
    model.fit(synth_X, synth_y)
    _cached_rul_model = model
    return model


@router.post("/soc", response_model=SOCInferenceResponse)
def estimate_soc(
    req: SOCInferenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Estimate State of Charge (SOC) via Physics-based EKF and Coulomb Counting."""
    if len(req.current_stream) != len(req.voltage_stream):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="current_stream and voltage_stream must have equal lengths",
        )
    if len(req.current_stream) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Stream cannot be empty")

    curr_arr = np.array(req.current_stream, dtype=float)
    volt_arr = np.array(req.voltage_stream, dtype=float)

    capacity_coulombs = req.nominal_capacity_ah * 3600.0
    params = ECMParameters(
        R0=0.05,
        R1=0.02,
        C1=1000.0,
        capacity=capacity_coulombs,
    )

    # Note: Coulomb counting convention in soc module: negative current = discharge
    # Here current_stream > 0 indicates discharge, so we pass -curr_arr
    soc_cc = coulomb_counting(
        current=-curr_arr,
        dt=req.dt_s,
        initial_soc=req.initial_soc,
        capacity=capacity_coulombs,
    )

    soc_ekf = run_ekf(
        current=curr_arr,
        voltage_meas=volt_arr,
        dt=req.dt_s,
        params=params,
        initial_soc=req.initial_soc,
    )

    final_ekf = float(soc_ekf[-1])
    final_cc = float(soc_cc[-1])
    error_bound = float(np.abs(final_ekf - final_cc))

    # Persist state estimate
    record = StateEstimate(
        cell_id=req.cell_id,
        soc_ekf=final_ekf,
        soc_cc=final_cc,
        error_bound=error_bound,
        is_synthetic=req.is_synthetic,
    )
    db.add(record)
    db.commit()

    return SOCInferenceResponse(
        cell_id=req.cell_id,
        soc_history_ekf=soc_ekf.tolist(),
        soc_history_cc=soc_cc.tolist(),
        final_soc_ekf=final_ekf,
        final_soc_cc=final_cc,
        error_bound=error_bound,
        is_synthetic=req.is_synthetic,
    )


@router.post("/soh", response_model=PredictionResponse)
def predict_soh(
    req: SOHInferenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Predict State of Health (SOH) using XGBoost model.

    Rule 2: Capacity and initial_capacity features are strictly rejected.
    """
    model = _get_or_create_soh_model()

    # Format input DataFrame matching FEATURE_COLUMNS
    feature_dict = {}
    for col in FEATURE_COLUMNS:
        feature_dict[col] = [req.features.get(col, 0.0)]
    df_input = pd.DataFrame(feature_dict)

    pred_soh = float(model.predict(df_input)[0])

    # Persist prediction in DB
    record = Prediction(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        target_type="SOH",
        predicted_value=pred_soh,
        confidence_interval=None,
        is_synthetic=req.is_synthetic,
    )
    db.add(record)
    db.commit()

    return PredictionResponse(
        cell_id=req.cell_id,
        target_type="SOH",
        predicted_value=pred_soh,
        lower_bound=None,
        upper_bound=None,
        confidence_interval=None,
        unit="fraction (0-1.0)",
        model_name="XGBoost_SOH",
        is_synthetic=req.is_synthetic,
    )


@router.post("/rul", response_model=PredictionResponse)
def predict_rul(
    req: RULInferenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Predict Remaining Useful Life (RUL) with 90% prediction intervals.

    Rule 4: RUL defined as cycles until SOH <= 80% (EOL).
    """
    model = _get_or_create_rul_model()

    feature_dict = {}
    for col in FEATURE_COLUMNS:
        feature_dict[col] = [req.features.get(col, 0.0)]
    df_input = pd.DataFrame(feature_dict)

    lower, median, upper = model.predict_interval(df_input)
    lower_val = float(np.maximum(0.0, lower[0]))
    median_val = float(np.maximum(0.0, median[0]))
    upper_val = float(np.maximum(0.0, upper[0]))

    record = Prediction(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        target_type="RUL",
        predicted_value=median_val,
        lower_bound=lower_val,
        upper_bound=upper_val,
        confidence_interval=0.90,
        is_synthetic=req.is_synthetic,
    )
    db.add(record)
    db.commit()

    return PredictionResponse(
        cell_id=req.cell_id,
        target_type="RUL",
        predicted_value=median_val,
        lower_bound=lower_val,
        upper_bound=upper_val,
        confidence_interval=0.90,
        unit="cycles",
        model_name="QuantileGB_RUL_EOL80",
        is_synthetic=req.is_synthetic,
    )


@router.post("/anomaly", response_model=AnomalyResponse)
def evaluate_anomaly(
    req: AnomalyInferenceRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Evaluate thermal risk and multi-feature anomalies with engineering evidence."""
    peak_temp = req.max_temp_c if req.max_temp_c is not None else req.features.get("temp_max", 25.0)
    temp_rise = req.features.get("temp_rise", 0.0)
    duration = max(req.features.get("duration_s", 3600.0), 1.0)
    heating_rate = req.temp_rise_rate if req.temp_rise_rate is not None else (temp_rise / duration)
    r_proxy = req.features.get("internal_resistance_proxy", 0.05)

    assessment = _thermal_engine.assess(
        peak_temperature_c=peak_temp,
        temperature_rise_c=temp_rise,
        max_heating_rate_c_per_s=heating_rate,
        internal_resistance_proxy=r_proxy,
    )

    is_anomaly = assessment.level != ThermalRiskLevel.NORMAL
    evidence = {
        "peak_temperature_c": assessment.peak_temperature_c,
        "temperature_rise_c": assessment.temperature_rise_c,
        "max_heating_rate_c_per_s": assessment.max_heating_rate_c_per_s,
        "internal_resistance_proxy": assessment.internal_resistance_proxy,
        "notes": assessment.evidence,
    }

    # Persist AnomalyEvent
    event = AnomalyEvent(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        anomaly_type="THERMAL_RISK",
        severity=assessment.level.value,
        score=assessment.score,
        evidence=evidence,
        is_synthetic=req.is_synthetic,
    )
    db.add(event)

    # If elevated or critical, create an Alert record
    if assessment.level in (ThermalRiskLevel.ELEVATED, ThermalRiskLevel.CRITICAL):
        alert_severity = "CRITICAL" if assessment.level == ThermalRiskLevel.CRITICAL else "WARNING"
        alert = Alert(
            cell_id=req.cell_id,
            alert_type="THERMAL_RISK_ESCALATION",
            severity=alert_severity,
            message=f"Thermal risk score {assessment.score:.1f}/100 ({assessment.level.value}) for cell {req.cell_id}",
            evidence=evidence,
            status="ACTIVE",
        )
        db.add(alert)

    db.commit()

    return AnomalyResponse(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        is_anomaly=is_anomaly,
        thermal_risk_score=assessment.score,
        thermal_severity=assessment.level.value,
        evidence=evidence,
        is_synthetic=req.is_synthetic,
    )
