"""Pydantic schemas for explainability endpoints (Rule 2 & Rule 4).
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


class ExplainSOHRequest(BaseModel):
    """Request for SOH explainability.

    Rule 2: capacity and initial_capacity are strictly rejected!
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
                f"Target leakage violation (Rule 2): forbidden features present: {detected}."
            )
        return v


class ExplainRULRequest(BaseModel):
    """Request for RUL explainability (Rule 4: EOL-80%)."""
    cell_id: str
    cycle_number: int = Field(ge=1)
    features: dict[str, float]
    is_synthetic: bool = False


class FeatureImpactDTO(BaseModel):
    """Serialized feature attribution."""
    feature_name: str
    feature_value: float
    shap_value: float
    direction: str


class RuleFindingDTO(BaseModel):
    """Serialized diagnostic rule evaluation."""
    rule_id: str
    category: str
    severity: str
    metric_name: str
    measured_value: float
    threshold_value: float
    explanation: str


class ExplainResponse(BaseModel):
    """Unified explainability response combining TreeSHAP and grounded rules."""
    cell_id: str
    cycle_number: int
    target_type: str
    predicted_value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    base_value: float
    feature_contributions: list[FeatureImpactDTO]
    top_positive_drivers: list[FeatureImpactDTO]
    top_negative_drivers: list[FeatureImpactDTO]
    rule_findings: list[RuleFindingDTO]
    narrative_explanation: str
    is_grounded: bool
    grounding_audit: list[str] = []
    is_synthetic: bool = False
