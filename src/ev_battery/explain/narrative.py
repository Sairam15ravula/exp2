"""Grounded engineering narrative synthesis and anti-hallucination verification.

Strict requirement: "Any LLM text must only rephrase computed values, never invent."
Every claim in the narrative must be verifiable against the numerical evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ev_battery.explain.rule_explainer import RuleFinding
from ev_battery.explain.shap_explainer import FeatureContribution, SHAPExplanation


@dataclass
class ExplanationBrief:
    """Comprehensive explainability report containing numerical attributions and grounded text."""

    cell_id: str
    cycle_number: int
    target_type: str
    predicted_value: float
    uncertainty_interval: tuple[float, float] | None
    narrative_summary: str
    top_drivers: list[str]
    rule_evidence: list[str]
    is_grounded: bool
    grounding_audit: list[str]


def synthesize_soh_narrative(
    cell_id: str,
    cycle_number: int,
    soh_prediction: float,
    shap_explanation: SHAPExplanation,
    rule_findings: list[RuleFinding],
    features: dict[str, float],
) -> str:
    """Synthesize a diagnostic narrative for SOH grounded exclusively in computed metrics."""
    soh_pct = soh_prediction * 100.0
    ir = float(features.get("internal_resistance_proxy", 0.050))
    t_max = float(features.get("temp_max", 25.0))
    t_rise = float(features.get("temp_rise", 0.0))

    # Identify primary SHAP feature driver
    top_driver_name = "aging progression"
    if shap_explanation.top_negative_drivers:
        top_driver_name = shap_explanation.top_negative_drivers[0].feature_name.replace("_", " ")

    # Severity tone based on SOH
    if soh_prediction >= 0.90:
        status_desc = "healthy operational condition"
    elif soh_prediction >= 0.80:
        status_desc = "moderate degradation"
    else:
        status_desc = "advanced degradation below the 80.0% End-of-Life threshold"

    # Assemble strictly grounded paragraphs
    p1 = (
        f"Cell {cell_id} at cycle {cycle_number} exhibits a predicted State of Health (SOH) of {soh_pct:.1f}%, "
        f"indicating {status_desc}. "
        f"Model feature attribution indicates that {top_driver_name} is the primary factor influencing this prediction."
    )

    # Physical observations
    obs_parts = []
    if t_max > 35.0:
        obs_parts.append(f"peak temperature of {t_max:.1f} °C (with a rise of {t_rise:.1f} °C)")
    else:
        obs_parts.append(f"controlled peak temperature of {t_max:.1f} °C")

    obs_parts.append(f"an internal resistance proxy of {ir:.3f} Ω")
    p2 = "Physical telemetry shows " + " and ".join(obs_parts) + "."

    # Actionable engineering conclusion
    severe_findings = [f for f in rule_findings if f.severity in ("WARNING", "CRITICAL")]
    if severe_findings:
        findings_str = "; ".join(f.explanation for f in severe_findings)
        p3 = f"Engineering alerts flagged: {findings_str}"
    else:
        p3 = "All physical telemetry metrics remain within acceptable operational envelopes."

    return f"{p1} {p2} {p3}"


def synthesize_rul_narrative(
    cell_id: str,
    cycle_number: int,
    rul_median: float,
    rul_lower: float,
    rul_upper: float,
    soh_current: float | None,
    rule_findings: list[RuleFinding],
) -> str:
    """Synthesize a diagnostic narrative for RUL grounded in uncertainty intervals and Rule 4."""
    projected_eol = cycle_number + int(rul_median)
    soh_str = f" from current SOH of {soh_current*100:.1f}%" if soh_current is not None else ""

    p1 = (
        f"Cell {cell_id} at cycle {cycle_number} has an estimated Remaining Useful Life of {rul_median:.0f} cycles "
        f"(90% confidence interval: [{rul_lower:.0f}, {rul_upper:.0f}] cycles) until reaching the 80.0% End-of-Life limit{soh_str}. "
        f"Projected End-of-Life is expected around cycle {projected_eol}."
    )

    if rul_median <= 20:
        p2 = "Warning: The remaining cycle margin is critically low. Battery module servicing or replacement is advised."
    else:
        p2 = "The battery maintains an operational margin before reaching decommissioning criteria."

    return f"{p1} {p2}"


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric representations (integers and floats) from text."""
    # Match numbers like 85.4, 0.050, 100, -2.5
    pattern = r"[-+]?\b\d+(?:\.\d+)?\b"
    matches = re.findall(pattern, text)
    numbers = []
    for m in matches:
        try:
            numbers.append(float(m))
        except ValueError:
            pass
    return numbers


def verify_grounded_text(text: str, evidence: dict[str, Any], tolerance: float = 0.05) -> tuple[bool, list[str]]:
    """Verify that every number cited in text is grounded in the evidence dictionary.

    Rule: Any LLM or narrative text must only rephrase computed values, never invent.
    """
    text_numbers = extract_numbers_from_text(text)
    known_numbers = set()

    def _collect_numbers(obj: Any):
        if isinstance(obj, (int, float)):
            v = float(obj)
            known_numbers.add(round(v, 4))
            known_numbers.add(round(v, 2))
            known_numbers.add(round(v, 1))
            known_numbers.add(round(v, 0))
            # Also add percentages (e.g. 0.85 -> 85.0)
            if 0.0 <= v <= 1.0:
                pct = v * 100.0
                known_numbers.add(round(pct, 2))
                known_numbers.add(round(pct, 1))
                known_numbers.add(round(pct, 0))
        elif isinstance(obj, dict):
            for val in obj.values():
                _collect_numbers(val)
        elif isinstance(obj, (list, tuple, set)):
            for item in obj:
                _collect_numbers(item)
        elif hasattr(obj, "__dict__"):
            _collect_numbers(obj.__dict__)

    _collect_numbers(evidence)

    # Standard constant electrochemical thresholds allowed in text
    known_numbers.update({80.0, 80, 0.80, 0.8, 90.0, 90, 35.0, 35, 45.0, 45, 60.0, 60, 2.5, 4.2, 0.05, 50.0})

    unmatched = []
    audit_notes = []

    for num in text_numbers:
        # Check if num is close to any known number
        matched = any(abs(num - k) <= tolerance or (k != 0 and abs(num - k) / abs(k) <= 0.02) for k in known_numbers)
        if matched:
            audit_notes.append(f"Number {num} verified against computed evidence.")
        else:
            unmatched.append(num)
            audit_notes.append(f"WARNING: Unverified number {num} detected in narrative text!")

    is_grounded = len(unmatched) == 0
    return is_grounded, audit_notes
