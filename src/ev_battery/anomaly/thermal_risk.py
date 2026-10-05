"""Thermal risk forecasting and escalation scoring engine.

Rule 5: Cell-level temperature and voltage validation.
Rule 10: Simple, clear, evidence-grounded risk scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ev_battery.data.loader import CycleData
from ev_battery.units import validate_temperature


class ThermalRiskLevel(str, Enum):
    """Categorical thermal risk levels for BMS alerts."""

    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


@dataclass
class ThermalRiskAssessment:
    """Comprehensive thermal risk evaluation with audit evidence."""

    score: float
    level: ThermalRiskLevel
    peak_temperature_c: float
    temperature_rise_c: float
    max_heating_rate_c_per_s: float
    internal_resistance_proxy: float
    evidence: list[str]


class ThermalRiskEngine:
    """Calculates continuous 0-100 thermal risk score and engineering evidence.

    Thresholds:
    - Safe Operating Limit: <= 35.0 °C
    - Elevated Warning Limit: 45.0 °C
    - Critical Danger Limit: 60.0 °C (SEI layer decomposition onset)
    - High delta-T threshold: 12.0 °C
    """

    SAFE_TEMP_LIMIT = 35.0
    WARNING_TEMP_LIMIT = 45.0
    CRITICAL_TEMP_LIMIT = 60.0
    HIGH_DELTA_T_LIMIT = 12.0

    def evaluate_cycle(self, cycle: CycleData) -> ThermalRiskAssessment:
        """Evaluate thermal risk for a completed discharge cycle."""
        temp = np.asarray(cycle.temperature, dtype=float)
        time = np.asarray(cycle.time, dtype=float)
        curr = np.asarray(cycle.current, dtype=float)
        volt = np.asarray(cycle.voltage, dtype=float)

        if len(temp) == 0:
            return ThermalRiskAssessment(
                score=0.0,
                level=ThermalRiskLevel.NORMAL,
                peak_temperature_c=25.0,
                temperature_rise_c=0.0,
                max_heating_rate_c_per_s=0.0,
                internal_resistance_proxy=0.0,
                evidence=["No temperature telemetry provided."],
            )

        # Rule 5: Validate safe temperature bounds
        for t_val in [float(np.min(temp)), float(np.max(temp))]:
            if not validate_temperature(t_val):
                raise ValueError(f"Temperature reading {t_val}°C is out of physical cell bounds.")

        t_max = float(np.max(temp))
        t_min = float(np.min(temp))
        delta_t = t_max - t_min

        # Heating rate calculation (dT / dt)
        if len(temp) > 1 and len(time) > 1:
            dt_diff = np.diff(time)
            dt_diff[dt_diff == 0] = 1.0
            dt_rates = np.diff(temp) / dt_diff
            max_rate = float(np.max(dt_rates))
        else:
            max_rate = 0.0

        # Internal resistance proxy
        mid_idx = len(volt) // 2
        if mid_idx > 0 and len(curr) > 0:
            avg_i = float(np.mean(np.abs(curr)))
            ir_proxy = float(volt[0] - volt[mid_idx]) / avg_i if avg_i > 0 else 0.0
        else:
            ir_proxy = 0.0

        return self.assess(
            peak_temperature_c=t_max,
            temperature_rise_c=delta_t,
            max_heating_rate_c_per_s=max_rate,
            internal_resistance_proxy=ir_proxy,
        )

    def assess(
        self,
        peak_temperature_c: float,
        temperature_rise_c: float,
        max_heating_rate_c_per_s: float = 0.0,
        internal_resistance_proxy: float = 0.0,
    ) -> ThermalRiskAssessment:
        """Calculate thermal risk score and audit evidence from scalar engineering metrics."""
        t_max = peak_temperature_c
        delta_t = temperature_rise_c
        max_rate = max_heating_rate_c_per_s
        ir_proxy = internal_resistance_proxy

        score = 0.0
        evidence: list[str] = []

        # 1. Peak temperature contribution (up to 55 points)
        if t_max <= self.SAFE_TEMP_LIMIT:
            score += 5.0 * (t_max / self.SAFE_TEMP_LIMIT)
            evidence.append(f"Peak temperature {t_max:.1f}°C is within optimal zone (<= {self.SAFE_TEMP_LIMIT}°C).")
        elif t_max <= self.WARNING_TEMP_LIMIT:
            # 35°C to 45°C -> 15 to 40 points
            frac = (t_max - self.SAFE_TEMP_LIMIT) / (self.WARNING_TEMP_LIMIT - self.SAFE_TEMP_LIMIT)
            pts = 15.0 + 25.0 * frac
            score += pts
            evidence.append(f"Peak temperature {t_max:.1f}°C entered elevated range (35°C-45°C).")
        elif t_max <= self.CRITICAL_TEMP_LIMIT:
            # 45°C to 60°C -> 45 to 75 points
            frac = (t_max - self.WARNING_TEMP_LIMIT) / (self.CRITICAL_TEMP_LIMIT - self.WARNING_TEMP_LIMIT)
            pts = 45.0 + 30.0 * frac
            score += pts
            evidence.append(f"WARNING: Peak temperature {t_max:.1f}°C breached 45°C limit.")
        else:
            # > 60°C -> 80 to 95 points
            excess = min(t_max - self.CRITICAL_TEMP_LIMIT, 20.0)
            score += 80.0 + (excess / 20.0) * 15.0
            evidence.append(f"CRITICAL HAZARD: Peak temperature {t_max:.1f}°C exceeds 60°C runaway risk threshold!")

        # 2. Temperature rise contribution (up to 25 points)
        if delta_t > self.HIGH_DELTA_T_LIMIT:
            excess_rise = min(delta_t - self.HIGH_DELTA_T_LIMIT, 15.0)
            rise_pts = 10.0 + (excess_rise / 15.0) * 15.0
            score += rise_pts
            evidence.append(f"High temperature rise of {delta_t:.1f}°C during discharge (+{rise_pts:.1f} risk pts).")
        else:
            evidence.append(f"Normal temperature rise of {delta_t:.1f}°C.")

        # 3. Heating rate contribution (up to 10 points)
        if max_rate > 0.05:  # faster than 0.05 °C/s (3 °C/min)
            score += 10.0
            evidence.append(f"Rapid transient heating rate detected: {max_rate*60:.1f}°C/min (+10 risk pts).")

        # 4. Internal resistance contribution (Joule heating potential I^2*R) (up to 10 points)
        if ir_proxy > 0.35:
            score += 10.0
            evidence.append(f"High internal resistance proxy ({ir_proxy:.3f} Ω) escalates thermal dissipation (+10 risk pts).")

        score = float(np.clip(score, 0.0, 100.0))

        # Severity level assignment
        if score >= 70.0:
            level = ThermalRiskLevel.CRITICAL
        elif score >= 40.0:
            level = ThermalRiskLevel.ELEVATED
        else:
            level = ThermalRiskLevel.NORMAL

        return ThermalRiskAssessment(
            score=round(score, 1),
            level=level,
            peak_temperature_c=round(t_max, 2),
            temperature_rise_c=round(delta_t, 2),
            max_heating_rate_c_per_s=round(max_rate, 4),
            internal_resistance_proxy=round(ir_proxy, 4),
            evidence=evidence,
        )
