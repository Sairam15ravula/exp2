"""Tests for thermal risk forecasting and score calibration."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.anomaly.thermal_risk import (
    ThermalRiskAssessment,
    ThermalRiskEngine,
    ThermalRiskLevel,
)
from ev_battery.data.loader import CycleData


class TestThermalRiskEngine:
    @pytest.fixture
    def engine(self):
        return ThermalRiskEngine()

    def _make_cycle(self, t_start: float, t_end: float, duration: float = 3600.0, v_sag: float = 0.6) -> CycleData:
        n = 100
        time = np.linspace(0, duration, n)
        temp = np.linspace(t_start, t_end, n)
        volt = np.linspace(4.2, 4.2 - v_sag, n)
        curr = np.full(n, -2.0)
        return CycleData("SYNTH_TEST", 1, "discharge", time, volt, curr, temp, 2.0)

    def test_normal_operating_cycle_is_low_risk(self, engine):
        """Under normal temperatures (25°C to 30°C), thermal risk score is low and classified NORMAL."""
        cycle = self._make_cycle(t_start=25.0, t_end=30.0)
        result = engine.evaluate_cycle(cycle)

        assert isinstance(result, ThermalRiskAssessment)
        assert result.score < 40.0
        assert result.level == ThermalRiskLevel.NORMAL
        assert result.peak_temperature_c == pytest.approx(30.0)
        assert len(result.evidence) > 0

    def test_elevated_warning_temperature(self, engine):
        """Temperatures breaching 45°C trigger ELEVATED risk level."""
        cycle = self._make_cycle(t_start=35.0, t_end=48.0)
        result = engine.evaluate_cycle(cycle)

        assert 40.0 <= result.score < 70.0
        assert result.level == ThermalRiskLevel.ELEVATED
        assert any("WARNING" in e or "45°C" in e for e in result.evidence)

    def test_critical_danger_temperature(self, engine):
        """Temperatures exceeding 60°C trigger CRITICAL risk level."""
        cycle = self._make_cycle(t_start=40.0, t_end=68.0)
        result = engine.evaluate_cycle(cycle)

        assert result.score >= 70.0
        assert result.level == ThermalRiskLevel.CRITICAL
        assert any("CRITICAL" in e for e in result.evidence)

    def test_score_strictly_bounded_0_to_100(self, engine):
        """Extreme temperatures must not exceed 100 or drop below 0."""
        cold_cycle = self._make_cycle(t_start=0.0, t_end=5.0)
        hot_cycle = self._make_cycle(t_start=70.0, t_end=95.0)

        assert 0.0 <= engine.evaluate_cycle(cold_cycle).score <= 100.0
        assert 0.0 <= engine.evaluate_cycle(hot_cycle).score <= 100.0

    def test_rule_5_out_of_bounds_temperature_rejected(self, engine):
        """Rule 5: Out-of-physical-range temperatures (> 100°C or < -40°C) must raise ValueError."""
        invalid_cycle = self._make_cycle(t_start=25.0, t_end=150.0)
        with pytest.raises(ValueError, match="out of physical cell bounds"):
            engine.evaluate_cycle(invalid_cycle)
