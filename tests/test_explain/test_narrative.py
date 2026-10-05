"""Tests for narrative synthesis and anti-hallucination verification.

Requirement: "Any LLM text must only rephrase computed values, never invent."
"""

from __future__ import annotations

import pytest

from ev_battery.explain.narrative import (
    extract_numbers_from_text,
    synthesize_rul_narrative,
    synthesize_soh_narrative,
    verify_grounded_text,
)
from ev_battery.explain.rule_explainer import BatteryRuleExplainer
from ev_battery.explain.shap_explainer import FeatureContribution, SHAPExplanation


def test_extract_numbers_from_text():
    """Verify number extractor parses integers, floats, and decimals."""
    sample = "Cell B0005 at cycle 42 has SOH 86.4% and RUL 35 cycles (90% CI: [28, 42]). Temp is 31.5 °C."
    numbers = extract_numbers_from_text(sample)
    expected_subset = {42.0, 86.4, 35.0, 90.0, 28.0, 31.5}
    assert expected_subset.issubset(set(numbers))


def test_synthesize_soh_narrative_and_grounding():
    """Verify that synthesized SOH narrative passes strict anti-hallucination verification."""
    cell_id = "B0005"
    cycle_number = 30
    soh_prediction = 0.885
    features = {
        "internal_resistance_proxy": 0.072,
        "temp_max": 32.5,
        "temp_rise": 5.2,
        "duration_s": 3500.0,
    }

    dummy_shap = SHAPExplanation(
        target_type="SOH",
        predicted_value=0.885,
        base_value=0.950,
        contributions=[],
        top_positive_drivers=[],
        top_negative_drivers=[
            FeatureContribution(
                feature_name="internal_resistance_proxy",
                feature_value=0.072,
                shap_value=-0.045,
                abs_impact=0.045,
                direction="decreases_prediction",
            )
        ],
        is_additive=True,
    )

    explainer = BatteryRuleExplainer()
    rule_findings = explainer.evaluate_all(features, predicted_soh=soh_prediction)

    narrative = synthesize_soh_narrative(
        cell_id=cell_id,
        cycle_number=cycle_number,
        soh_prediction=soh_prediction,
        shap_explanation=dummy_shap,
        rule_findings=rule_findings,
        features=features,
    )

    # Narrative must mention essential facts
    assert "B0005" in narrative
    assert "cycle 30" in narrative
    assert "88.5%" in narrative
    assert "0.072" in narrative

    # Verify anti-hallucination grounding
    evidence_bundle = {
        "cell_id": cell_id,
        "cycle_number": cycle_number,
        "soh_prediction": soh_prediction,
        "features": features,
        "findings": [f.measured_value for f in rule_findings],
    }

    is_grounded, audit_notes = verify_grounded_text(narrative, evidence_bundle)
    assert is_grounded is True


def test_anti_hallucination_detects_invented_numbers():
    """Verify that ungrounded, invented numbers in text are caught by the auditor."""
    evidence_bundle = {
        "cycle_number": 20,
        "soh": 0.90,
        "temp_max": 30.0,
    }

    # Text containing invented numbers not in evidence: 777.7 and 9999
    hallucinated_text = (
        "Cell at cycle 20 has SOH 90.0% with maximum temperature 30.0 °C. "
        "Estimated energy capacity is 777.7 Wh with 9999 microcracks in anode."
    )

    is_grounded, audit_notes = verify_grounded_text(hallucinated_text, evidence_bundle)
    assert is_grounded is False
    assert any("777.7" in note for note in audit_notes)
    assert any("9999" in note for note in audit_notes)


def test_synthesize_rul_narrative_and_grounding():
    """Verify RUL narrative mentions confidence intervals and passes grounding check."""
    cell_id = "B0006"
    cycle_number = 50
    rul_median = 40.0
    rul_lower = 32.0
    rul_upper = 48.0
    soh_current = 0.85

    explainer = BatteryRuleExplainer()
    rule_findings = explainer.evaluate_degradation_trajectory(
        features={"cycle_number": cycle_number},
        soh=soh_current,
        rul=rul_median,
        rul_lower=rul_lower,
        rul_upper=rul_upper,
    )

    narrative = synthesize_rul_narrative(
        cell_id=cell_id,
        cycle_number=cycle_number,
        rul_median=rul_median,
        rul_lower=rul_lower,
        rul_upper=rul_upper,
        soh_current=soh_current,
        rule_findings=rule_findings,
    )

    assert "40 cycles" in narrative
    assert "[32, 48]" in narrative
    assert "cycle 90" in narrative  # 50 + 40 = 90
    assert "80.0%" in narrative

    evidence_bundle = {
        "cell_id": cell_id,
        "cycle_number": cycle_number,
        "rul_median": rul_median,
        "rul_lower": rul_lower,
        "rul_upper": rul_upper,
        "soh_current": soh_current,
        "projected_eol": 90,
    }

    is_grounded, _ = verify_grounded_text(narrative, evidence_bundle)
    assert is_grounded is True
