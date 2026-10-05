"""Domain-grounded physical diagnostic rule engine for battery health.

Translates physical telemetry metrics into engineering diagnostic findings
strictly backed by numerical thresholds and electrochemistry principles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ev_battery.rul.dataset import EOL_SOH_THRESHOLD
from ev_battery.units import MAX_CELL_VOLTAGE, MIN_CELL_VOLTAGE


@dataclass
class RuleFinding:
    """Individual diagnostic finding grounded in a specific physical rule."""

    rule_id: str
    category: str  # 'IMPEDANCE', 'THERMAL', 'VOLTAGE', 'DEGRADATION'
    severity: str  # 'INFO', 'WARNING', 'CRITICAL'
    metric_name: str
    measured_value: float
    threshold_value: float
    explanation: str


class BatteryRuleExplainer:
    """Evaluates battery operational features against electrochemical thresholds."""

    # Baseline constants for healthy cell (~2Ah 18650 NMC)
    R_HEALTHY_BASELINE_OHM = 0.050
    R_WARNING_OHM = 0.080
    R_CRITICAL_OHM = 0.150

    TEMP_SAFE_LIMIT_C = 35.0
    TEMP_WARNING_LIMIT_C = 45.0
    TEMP_CRITICAL_LIMIT_C = 60.0
    TEMP_RISE_LIMIT_C = 12.0

    VOLTAGE_DROP_WARNING_RATE = 0.0020  # 2.0 mV/s
    VOLTAGE_DROP_CRITICAL_RATE = 0.0035  # 3.5 mV/s

    def evaluate_all(
        self,
        features: dict[str, float],
        predicted_soh: float | None = None,
        predicted_rul: float | None = None,
        rul_lower: float | None = None,
        rul_upper: float | None = None,
    ) -> list[RuleFinding]:
        """Evaluate all physical diagnostic rules on cycle features."""
        findings: list[RuleFinding] = []

        # 1. Internal Resistance Rule
        findings.extend(self.evaluate_internal_resistance(features))

        # 2. Thermal Stress Rule
        findings.extend(self.evaluate_thermal_stress(features))

        # 3. Voltage Dynamics Rule
        findings.extend(self.evaluate_voltage_dynamics(features))

        # 4. SOH & RUL Trajectory Rule
        if predicted_soh is not None or predicted_rul is not None:
            findings.extend(
                self.evaluate_degradation_trajectory(
                    features=features,
                    soh=predicted_soh,
                    rul=predicted_rul,
                    rul_lower=rul_lower,
                    rul_upper=rul_upper,
                )
            )

        return findings

    def evaluate_internal_resistance(self, features: dict[str, float]) -> list[RuleFinding]:
        """Evaluate internal resistance proxy relative to fresh cell baseline."""
        r_proxy = float(features.get("internal_resistance_proxy", self.R_HEALTHY_BASELINE_OHM))
        pct_increase = ((r_proxy - self.R_HEALTHY_BASELINE_OHM) / self.R_HEALTHY_BASELINE_OHM) * 100.0

        findings = []
        if r_proxy >= self.R_CRITICAL_OHM:
            findings.append(
                RuleFinding(
                    rule_id="RULE_IR_CRITICAL",
                    category="IMPEDANCE",
                    severity="CRITICAL",
                    metric_name="internal_resistance_proxy",
                    measured_value=round(r_proxy, 4),
                    threshold_value=self.R_CRITICAL_OHM,
                    explanation=(
                        f"Internal resistance proxy is {r_proxy:.3f} Ω ({pct_increase:+.1f}% vs baseline {self.R_HEALTHY_BASELINE_OHM:.3f} Ω). "
                        "Severe ohmic degradation causes excessive Joule heating (I²R) and extreme terminal voltage sag."
                    ),
                )
            )
        elif r_proxy >= self.R_WARNING_OHM:
            findings.append(
                RuleFinding(
                    rule_id="RULE_IR_ELEVATED",
                    category="IMPEDANCE",
                    severity="WARNING",
                    metric_name="internal_resistance_proxy",
                    measured_value=round(r_proxy, 4),
                    threshold_value=self.R_WARNING_OHM,
                    explanation=(
                        f"Internal resistance proxy reached {r_proxy:.3f} Ω ({pct_increase:+.1f}% vs baseline). "
                        "Moderate impedance growth indicates solid electrolyte interphase (SEI) thickening or current collector oxidation."
                    ),
                )
            )
        else:
            findings.append(
                RuleFinding(
                    rule_id="RULE_IR_NORMAL",
                    category="IMPEDANCE",
                    severity="INFO",
                    metric_name="internal_resistance_proxy",
                    measured_value=round(r_proxy, 4),
                    threshold_value=self.R_HEALTHY_BASELINE_OHM,
                    explanation=f"Internal resistance proxy is normal at {r_proxy:.3f} Ω (healthy baseline: {self.R_HEALTHY_BASELINE_OHM:.3f} Ω).",
                )
            )
        return findings

    def evaluate_thermal_stress(self, features: dict[str, float]) -> list[RuleFinding]:
        """Evaluate peak temperature and temperature rise against thermal limits."""
        findings = []
        t_max = float(features.get("temp_max", 25.0))
        t_rise = float(features.get("temp_rise", 0.0))

        if t_max >= self.TEMP_CRITICAL_LIMIT_C:
            findings.append(
                RuleFinding(
                    rule_id="RULE_TEMP_CRITICAL",
                    category="THERMAL",
                    severity="CRITICAL",
                    metric_name="temp_max",
                    measured_value=round(t_max, 1),
                    threshold_value=self.TEMP_CRITICAL_LIMIT_C,
                    explanation=(
                        f"Peak temperature reached {t_max:.1f} °C, breaching the critical 60 °C safety limit. "
                        "Metastable SEI decomposition and electrolyte breakdown risks are severe."
                    ),
                )
            )
        elif t_max >= self.TEMP_WARNING_LIMIT_C:
            findings.append(
                RuleFinding(
                    rule_id="RULE_TEMP_WARNING",
                    category="THERMAL",
                    severity="WARNING",
                    metric_name="temp_max",
                    measured_value=round(t_max, 1),
                    threshold_value=self.TEMP_WARNING_LIMIT_C,
                    explanation=(
                        f"Peak temperature reached {t_max:.1f} °C (above 45.0 °C threshold). "
                        "Elevated thermal stress accelerates parasitic chemical reactions and calendar ageing."
                    ),
                )
            )
        else:
            findings.append(
                RuleFinding(
                    rule_id="RULE_TEMP_SAFE",
                    category="THERMAL",
                    severity="INFO",
                    metric_name="temp_max",
                    measured_value=round(t_max, 1),
                    threshold_value=self.TEMP_SAFE_LIMIT_C,
                    explanation=f"Peak temperature is well-controlled at {t_max:.1f} °C (within safe zone <= 35.0 °C).",
                )
            )

        if t_rise >= self.TEMP_RISE_LIMIT_C:
            findings.append(
                RuleFinding(
                    rule_id="RULE_TEMP_RISE_HIGH",
                    category="THERMAL",
                    severity="WARNING",
                    metric_name="temp_rise",
                    measured_value=round(t_rise, 1),
                    threshold_value=self.TEMP_RISE_LIMIT_C,
                    explanation=(
                        f"Large temperature rise of {t_rise:.1f} °C observed during discharge (threshold: {self.TEMP_RISE_LIMIT_C:.1f} °C), "
                        "indicating substantial internal heat generation."
                    ),
                )
            )

        return findings

    def evaluate_voltage_dynamics(self, features: dict[str, float]) -> list[RuleFinding]:
        """Evaluate discharge voltage drop rate and voltage boundaries."""
        findings = []
        drop_rate = float(features.get("voltage_drop_rate", 0.001))
        v_min = float(features.get("voltage_min", 2.7))

        if drop_rate >= self.VOLTAGE_DROP_CRITICAL_RATE:
            findings.append(
                RuleFinding(
                    rule_id="RULE_VOLT_DROP_FAST",
                    category="VOLTAGE",
                    severity="WARNING",
                    metric_name="voltage_drop_rate",
                    measured_value=round(drop_rate, 4),
                    threshold_value=self.VOLTAGE_DROP_CRITICAL_RATE,
                    explanation=(
                        f"Discharge voltage drop rate is high at {drop_rate*1000.0:.1f} mV/s (threshold: {self.VOLTAGE_DROP_CRITICAL_RATE*1000.0:.1f} mV/s), "
                        "showing rapid polarization and early depletion."
                    ),
                )
            )

        if v_min < MIN_CELL_VOLTAGE:
            findings.append(
                RuleFinding(
                    rule_id="RULE_VOLT_UNDERVOLT",
                    category="VOLTAGE",
                    severity="CRITICAL",
                    metric_name="voltage_min",
                    measured_value=round(v_min, 3),
                    threshold_value=MIN_CELL_VOLTAGE,
                    explanation=f"Minimum cell voltage {v_min:.3f} V breached lower cut-off limit ({MIN_CELL_VOLTAGE} V). Risk of copper dissolution.",
                )
            )

        return findings

    def evaluate_degradation_trajectory(
        self,
        features: dict[str, float],
        soh: float | None = None,
        rul: float | None = None,
        rul_lower: float | None = None,
        rul_upper: float | None = None,
    ) -> list[RuleFinding]:
        """Rule 4: Evaluate remaining useful life until SOH <= 80% EOL boundary."""
        findings = []
        cycle_num = int(features.get("cycle_number", 1))

        if soh is not None:
            if soh <= EOL_SOH_THRESHOLD:
                findings.append(
                    RuleFinding(
                        rule_id="RULE_EOL_REACHED",
                        category="DEGRADATION",
                        severity="CRITICAL",
                        metric_name="soh",
                        measured_value=round(soh, 3),
                        threshold_value=EOL_SOH_THRESHOLD,
                        explanation=(
                            f"Cell SOH has reached {soh*100:.1f}%, which is at or below the 80.0% End-of-Life (EOL) standard. "
                            "Pack retirement or second-life recycling is recommended."
                        ),
                    )
                )
            else:
                remaining_margin = (soh - EOL_SOH_THRESHOLD) * 100.0
                findings.append(
                    RuleFinding(
                        rule_id="RULE_SOH_HEALTHY",
                        category="DEGRADATION",
                        severity="INFO",
                        metric_name="soh",
                        measured_value=round(soh, 3),
                        threshold_value=EOL_SOH_THRESHOLD,
                        explanation=f"Current SOH is {soh*100:.1f}% with a {remaining_margin:.1f}% health buffer before reaching the 80.0% EOL threshold.",
                    )
                )

        if rul is not None:
            ci_str = f" (90% CI: [{rul_lower:.0f}, {rul_upper:.0f}] cycles)" if (rul_lower is not None and rul_upper is not None) else ""
            findings.append(
                RuleFinding(
                    rule_id="RULE_RUL_PREDICTION",
                    category="DEGRADATION",
                    severity="INFO" if rul > 20 else "WARNING",
                    metric_name="rul",
                    measured_value=round(rul, 1),
                    threshold_value=20.0,
                    explanation=(
                        f"Predicted Remaining Useful Life is {rul:.0f} cycles{ci_str} until 80.0% SOH is reached (Rule 4). "
                        f"Projected EOL will occur around cycle {cycle_num + int(rul)}."
                    ),
                )
            )

        return findings
