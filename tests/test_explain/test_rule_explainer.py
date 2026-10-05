"""Tests for BatteryRuleExplainer (electrochemical and physical diagnostic rules).
"""

from __future__ import annotations

import pytest

from ev_battery.explain.rule_explainer import BatteryRuleExplainer


def test_rule_explainer_healthy_cell():
    """Verify that a healthy cell generates purely INFO findings."""
    explainer = BatteryRuleExplainer()
    healthy_features = {
        "cycle_number": 5,
        "voltage_max": 4.19,
        "voltage_min": 2.75,
        "voltage_mean": 3.75,
        "voltage_drop_rate": 0.0010,
        "temp_max": 28.0,
        "temp_min": 24.0,
        "temp_rise": 4.0,
        "internal_resistance_proxy": 0.052,
        "duration_s": 3600.0,
    }

    findings = explainer.evaluate_all(healthy_features, predicted_soh=0.98, predicted_rul=120.0)

    severities = {f.severity for f in findings}
    assert "CRITICAL" not in severities
    assert "WARNING" not in severities
    assert any(f.rule_id == "RULE_IR_NORMAL" for f in findings)
    assert any(f.rule_id == "RULE_TEMP_SAFE" for f in findings)
    assert any(f.rule_id == "RULE_SOH_HEALTHY" for f in findings)


def test_rule_explainer_degraded_impedance():
    """Verify resistance growth triggers impedance warning/critical findings."""
    explainer = BatteryRuleExplainer()

    # Elevated IR
    elevated_features = {"internal_resistance_proxy": 0.095}
    findings_elevated = explainer.evaluate_internal_resistance(elevated_features)
    assert any(f.rule_id == "RULE_IR_ELEVATED" and f.severity == "WARNING" for f in findings_elevated)

    # Critical IR
    critical_features = {"internal_resistance_proxy": 0.165}
    findings_crit = explainer.evaluate_internal_resistance(critical_features)
    assert any(f.rule_id == "RULE_IR_CRITICAL" and f.severity == "CRITICAL" for f in findings_crit)


def test_rule_explainer_thermal_hazards():
    """Verify high temperatures and rises trigger thermal warnings."""
    explainer = BatteryRuleExplainer()

    # Breaching 60°C critical runaway threshold
    crit_temp_features = {"temp_max": 62.5, "temp_rise": 15.0}
    findings = explainer.evaluate_thermal_stress(crit_temp_features)

    assert any(f.rule_id == "RULE_TEMP_CRITICAL" and f.severity == "CRITICAL" for f in findings)
    assert any(f.rule_id == "RULE_TEMP_RISE_HIGH" and f.severity == "WARNING" for f in findings)


def test_rule_explainer_eol_and_rul_rule_4():
    """Rule 4: Verify EOL-80% threshold rule and RUL confidence interval interpretation."""
    explainer = BatteryRuleExplainer()
    features = {"cycle_number": 80}

    # Case 1: SOH has reached EOL (<= 80%)
    findings_eol = explainer.evaluate_degradation_trajectory(
        features=features,
        soh=0.785,
        rul=0.0,
    )
    assert any(f.rule_id == "RULE_EOL_REACHED" and f.severity == "CRITICAL" for f in findings_eol)

    # Case 2: SOH is above EOL with RUL prediction interval
    findings_rul = explainer.evaluate_degradation_trajectory(
        features=features,
        soh=0.865,
        rul=45.0,
        rul_lower=38.0,
        rul_upper=52.0,
    )
    rul_finding = next(f for f in findings_rul if f.rule_id == "RULE_RUL_PREDICTION")
    assert "[38, 52]" in rul_finding.explanation
    assert "80.0% SOH" in rul_finding.explanation
    assert "cycle 125" in rul_finding.explanation  # 80 + 45 = 125
