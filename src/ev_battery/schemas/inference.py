"""Inference request and response schemas (Rule 2 & Rule 4).
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class SOCInferenceRequest(BaseModel):
    """Request for physics-based EKF and Coulomb Counting SOC estimation."""
    cell_id: str
    initial_soc: float = Field(default=1.0, ge=0.0, le=1.0)
    current_stream: list[float] = Field(description="Array of current readings in Amperes (>0 discharge)")
    voltage_stream: list[float] = Field(description="Array of terminal voltage readings in Volts")
    dt_s: float = Field(default=1.0, gt=0, description="Sampling interval in seconds")
    nominal_capacity_ah: float = Field(default=2.0, gt=0)
    is_synthetic: bool = False


class SOCInferenceResponse(BaseModel):
    """Result of SOC estimation."""
    cell_id: str
    soc_history_ekf: list[float]
    soc_history_cc: list[float]
    final_soc_ekf: float
    final_soc_cc: float
    error_bound: float
    is_synthetic: bool


class SOHInferenceRequest(BaseModel):
    """Request for ML SOH estimation.

    Rule 2: capacity and initial_capacity are strictly forbidden features!
    """
    cell_id: str
    cycle_number: int = Field(ge=1)
    features: dict[str, float]
    is_synthetic: bool = False

    @field_validator("features")
    @classmethod
    def reject_target_leakage_features(cls, v: dict[str, float]) -> dict[str, float]:
        forbidden = {"capacity", "initial_capacity", "soh", "target", "capacity_ah"}
        detected = forbidden.intersection(v.keys())
        if detected:
            raise ValueError(
                f"Target leakage violation (Rule 2): forbidden features present: {detected}. "
                "SOH is the ratio of capacity / initial_capacity and must NOT take them as inputs."
            )
        return v


class RULInferenceRequest(BaseModel):
    """Request for Quantile ML RUL prediction (Rule 4: EOL-80%)."""
    cell_id: str
    cycle_number: int = Field(ge=1)
    features: dict[str, float]
    is_synthetic: bool = False


class AnomalyInferenceRequest(BaseModel):
    """Request for Hybrid Anomaly and Thermal Risk scoring."""
    cell_id: str
    cycle_number: int = Field(ge=1)
    features: dict[str, float]
    voltage_series: Optional[list[float]] = None
    current_series: Optional[list[float]] = None
    time_series: Optional[list[float]] = None
    max_temp_c: Optional[float] = None
    temp_rise_rate: Optional[float] = None
    is_synthetic: bool = False


class PredictionResponse(BaseModel):
    """Standardized prediction output with uncertainty bounds."""
    cell_id: str
    target_type: str  # SOH or RUL
    predicted_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    confidence_interval: Optional[float] = None
    unit: str
    model_name: str
    is_synthetic: bool


class AnomalyResponse(BaseModel):
    """Standardized anomaly and thermal risk output with evidence."""
    cell_id: str
    cycle_number: int
    is_anomaly: bool
    thermal_risk_score: float
    thermal_severity: str
    evidence: dict[str, Any]
    is_synthetic: bool
