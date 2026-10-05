"""REST API endpoints for TreeSHAP and Rule-Grounded Explainability.

Rules enforced:
- Rule 2: Zero target leakage.
- Rule 4: RUL strictly defined as cycles until SOH <= 80%.
- Anti-Hallucination: Verifies that narrative text contains only factual computed values.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
import pandas as pd

from ev_battery.api.inference import _get_or_create_rul_model, _get_or_create_soh_model
from ev_battery.explain.narrative import (
    synthesize_rul_narrative,
    synthesize_soh_narrative,
    verify_grounded_text,
)
from ev_battery.explain.rule_explainer import BatteryRuleExplainer
from ev_battery.explain.shap_explainer import BatterySHAPExplainer
from ev_battery.schemas.explain import (
    ExplainResponse,
    ExplainRULRequest,
    ExplainSOHRequest,
    FeatureImpactDTO,
    RuleFindingDTO,
)
from ev_battery.soh.dataset import FEATURE_COLUMNS

router = APIRouter(prefix="/explain", tags=["Explainability"])

_rule_explainer = BatteryRuleExplainer()


@router.post("/soh", response_model=ExplainResponse)
def explain_soh(req: ExplainSOHRequest):
    """Generate exact TreeSHAP attributions and grounded engineering explanations for SOH."""
    model = _get_or_create_soh_model()
    shap_explainer = BatterySHAPExplainer(model, target_type="SOH")

    # 1. Compute TreeSHAP attributions
    shap_res = shap_explainer.explain_instance(req.features)

    # 2. Evaluate domain physical rules
    findings = _rule_explainer.evaluate_all(
        features=req.features,
        predicted_soh=shap_res.predicted_value,
    )

    # 3. Synthesize grounded narrative
    narrative = synthesize_soh_narrative(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        soh_prediction=shap_res.predicted_value,
        shap_explanation=shap_res,
        rule_findings=findings,
        features=req.features,
    )

    # 4. Anti-hallucination verification
    evidence_bundle = {
        "cell_id": req.cell_id,
        "cycle_number": req.cycle_number,
        "predicted_value": shap_res.predicted_value,
        "base_value": shap_res.base_value,
        "features": req.features,
        "findings": [f.measured_value for f in findings],
    }
    is_grounded, audit_notes = verify_grounded_text(narrative, evidence_bundle)

    # Map DTOs
    feat_dtos = [
        FeatureImpactDTO(
            feature_name=c.feature_name,
            feature_value=c.feature_value,
            shap_value=c.shap_value,
            direction=c.direction,
        )
        for c in shap_res.contributions
    ]
    pos_dtos = [
        FeatureImpactDTO(
            feature_name=c.feature_name,
            feature_value=c.feature_value,
            shap_value=c.shap_value,
            direction=c.direction,
        )
        for c in shap_res.top_positive_drivers
    ]
    neg_dtos = [
        FeatureImpactDTO(
            feature_name=c.feature_name,
            feature_value=c.feature_value,
            shap_value=c.shap_value,
            direction=c.direction,
        )
        for c in shap_res.top_negative_drivers
    ]
    rule_dtos = [
        RuleFindingDTO(
            rule_id=f.rule_id,
            category=f.category,
            severity=f.severity,
            metric_name=f.metric_name,
            measured_value=f.measured_value,
            threshold_value=f.threshold_value,
            explanation=f.explanation,
        )
        for f in findings
    ]

    return ExplainResponse(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        target_type="SOH",
        predicted_value=shap_res.predicted_value,
        lower_bound=None,
        upper_bound=None,
        base_value=shap_res.base_value,
        feature_contributions=feat_dtos,
        top_positive_drivers=pos_dtos,
        top_negative_drivers=neg_dtos,
        rule_findings=rule_dtos,
        narrative_explanation=narrative,
        is_grounded=is_grounded,
        grounding_audit=audit_notes,
        is_synthetic=req.is_synthetic,
    )


@router.post("/rul", response_model=ExplainResponse)
def explain_rul(req: ExplainRULRequest):
    """Generate TreeSHAP attributions and grounded engineering explanations for RUL."""
    model = _get_or_create_rul_model()
    shap_explainer = BatterySHAPExplainer(model, target_type="RUL")

    # Compute prediction interval
    feature_dict = {col: [req.features.get(col, 0.0)] for col in FEATURE_COLUMNS}
    df_input = pd.DataFrame(feature_dict)
    lower_arr, median_arr, upper_arr = model.predict_interval(df_input)
    lower = float(np.maximum(0.0, lower_arr[0]))
    median = float(np.maximum(0.0, median_arr[0]))
    upper = float(np.maximum(0.0, upper_arr[0]))

    # Compute SHAP
    shap_res = shap_explainer.explain_instance(req.features)

    # Physical rules evaluation
    findings = _rule_explainer.evaluate_all(
        features=req.features,
        predicted_rul=median,
        rul_lower=lower,
        rul_upper=upper,
    )

    # Grounded narrative
    soh_est = req.features.get("soh_estimate", None)
    narrative = synthesize_rul_narrative(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        rul_median=median,
        rul_lower=lower,
        rul_upper=upper,
        soh_current=soh_est,
        rule_findings=findings,
    )

    evidence_bundle = {
        "cell_id": req.cell_id,
        "cycle_number": req.cycle_number,
        "rul_median": median,
        "rul_lower": lower,
        "rul_upper": upper,
        "soh_current": soh_est,
        "projected_eol": req.cycle_number + int(median),
        "features": req.features,
        "findings": [f.measured_value for f in findings],
    }
    is_grounded, audit_notes = verify_grounded_text(narrative, evidence_bundle)

    feat_dtos = [
        FeatureImpactDTO(
            feature_name=c.feature_name,
            feature_value=c.feature_value,
            shap_value=c.shap_value,
            direction=c.direction,
        )
        for c in shap_res.contributions
    ]
    pos_dtos = [
        FeatureImpactDTO(
            feature_name=c.feature_name,
            feature_value=c.feature_value,
            shap_value=c.shap_value,
            direction=c.direction,
        )
        for c in shap_res.top_positive_drivers
    ]
    neg_dtos = [
        FeatureImpactDTO(
            feature_name=c.feature_name,
            feature_value=c.feature_value,
            shap_value=c.shap_value,
            direction=c.direction,
        )
        for c in shap_res.top_negative_drivers
    ]
    rule_dtos = [
        RuleFindingDTO(
            rule_id=f.rule_id,
            category=f.category,
            severity=f.severity,
            metric_name=f.metric_name,
            measured_value=f.measured_value,
            threshold_value=f.threshold_value,
            explanation=f.explanation,
        )
        for f in findings
    ]

    return ExplainResponse(
        cell_id=req.cell_id,
        cycle_number=req.cycle_number,
        target_type="RUL",
        predicted_value=median,
        lower_bound=lower,
        upper_bound=upper,
        base_value=shap_res.base_value,
        feature_contributions=feat_dtos,
        top_positive_drivers=pos_dtos,
        top_negative_drivers=neg_dtos,
        rule_findings=rule_dtos,
        narrative_explanation=narrative,
        is_grounded=is_grounded,
        grounding_audit=audit_notes,
        is_synthetic=req.is_synthetic,
    )
