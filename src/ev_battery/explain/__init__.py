"""Explainability package exports.
"""

from ev_battery.explain.narrative import (
    ExplanationBrief,
    extract_numbers_from_text,
    synthesize_rul_narrative,
    synthesize_soh_narrative,
    verify_grounded_text,
)
from ev_battery.explain.rule_explainer import BatteryRuleExplainer, RuleFinding
from ev_battery.explain.shap_explainer import (
    BatterySHAPExplainer,
    FeatureContribution,
    SHAPExplanation,
)

__all__ = [
    "BatterySHAPExplainer",
    "FeatureContribution",
    "SHAPExplanation",
    "BatteryRuleExplainer",
    "RuleFinding",
    "ExplanationBrief",
    "synthesize_soh_narrative",
    "synthesize_rul_narrative",
    "extract_numbers_from_text",
    "verify_grounded_text",
]
