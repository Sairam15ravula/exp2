"""Injected-fault integration tests using synthetic generator (Rule 8)."""

from __future__ import annotations

import pytest

from ev_battery.anomaly.detector import (
    ANOMALY_FEATURE_COLUMNS,
    HybridAnomalyDetector,
    IsolationForestDetector,
)
from ev_battery.anomaly.thermal_risk import (
    ThermalRiskEngine,
    ThermalRiskLevel,
)
from ev_battery.data.features import extract_all_features
from ev_battery.data.synthetic import (
    generate_synthetic_battery,
    generate_synthetic_with_fault,
)


class TestInjectedFaultDetection:
    """Integration test verifying that sudden faults injected into synthetic telemetry

    are caught immediately by both the anomaly detector and thermal risk engine.
    Rule 8: Synthetic data must be clearly labelled ('SYNTH_FAULT').
    """

    @pytest.fixture
    def setup_detectors(self):
        # Train ML detector on healthy baseline cycles
        normal_cycles = generate_synthetic_battery("SYNTH_NORMAL_BASE", n_cycles=60, seed=42)
        normal_feats = extract_all_features(normal_cycles)[ANOMALY_FEATURE_COLUMNS]

        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(normal_feats)

        hybrid = HybridAnomalyDetector(detector, residual_threshold_rms=0.075)
        thermal_engine = ThermalRiskEngine()

        return hybrid, thermal_engine

    def test_fault_detected_at_injected_cycle(self, setup_detectors):
        hybrid, thermal_engine = setup_detectors

        # Fault is injected at cycle 35 (capacity drop, +15°C temperature rise)
        fault_cycle_idx = 35
        cycles = generate_synthetic_with_fault(
            battery_id="SYNTH_FAULT_001",
            n_cycles=60,
            fault_cycle=fault_cycle_idx,
            seed=42,
        )

        # 1. Inspect cycle immediately before fault (Cycle 34)
        pre_fault_cycle = cycles[fault_cycle_idx - 2]  # 0-indexed: 33 is cycle 34
        pre_risk = thermal_engine.evaluate_cycle(pre_fault_cycle)
        assert pre_risk.level == ThermalRiskLevel.NORMAL
        assert pre_risk.peak_temperature_c < 42.0

        # 2. Inspect cycle at fault injection (Cycle 35)
        fault_cycle = cycles[fault_cycle_idx - 1]  # 0-indexed: 34 is cycle 35
        fault_diag = hybrid.diagnose_cycle(fault_cycle)
        fault_risk = thermal_engine.evaluate_cycle(fault_cycle)

        # The fault MUST be flagged as an anomaly
        assert fault_diag["is_anomaly"] == True  # noqa: E712
        assert fault_diag["anomaly_type"] in [
            "CRITICAL_HYBRID_ANOMALY",
            "MULTIVARIATE_FEATURE_DRIFT",
            "PHYSICS_RESIDUAL_SURGE",
        ]

        # Thermal risk MUST escalate to ELEVATED or CRITICAL
        assert fault_risk.level in [ThermalRiskLevel.ELEVATED, ThermalRiskLevel.CRITICAL]
        assert fault_risk.score >= 40.0
        assert fault_risk.peak_temperature_c >= 45.0  # Due to +15°C injection

    def test_detection_statistics_pre_vs_post_fault(self, setup_detectors):
        hybrid, thermal_engine = setup_detectors
        fault_cycle_idx = 30
        cycles = generate_synthetic_with_fault(
            battery_id="SYNTH_FAULT_STATS",
            n_cycles=60,
            fault_cycle=fault_cycle_idx,
            seed=10,
        )

        pre_fault_anomalies = 0
        post_fault_anomalies = 0
        post_fault_elevated_thermal = 0

        for c in cycles:
            diag = hybrid.diagnose_cycle(c)
            risk = thermal_engine.evaluate_cycle(c)

            if c.cycle_number < fault_cycle_idx:
                if diag["is_anomaly"]:
                    pre_fault_anomalies += 1
            else:
                if diag["is_anomaly"]:
                    post_fault_anomalies += 1
                if risk.level != ThermalRiskLevel.NORMAL:
                    post_fault_elevated_thermal += 1

        n_pre = fault_cycle_idx - 1
        n_post = len(cycles) - n_pre

        false_alarm_rate = pre_fault_anomalies / n_pre
        detection_rate = post_fault_anomalies / n_post
        thermal_alarm_rate = post_fault_elevated_thermal / n_post

        # False alarm rate on healthy cycles should be low
        assert false_alarm_rate <= 0.15
        # Detection rate on faulted cycles should be very high
        assert detection_rate >= 0.80
        # Thermal engine should catch the elevated temperatures
        assert thermal_alarm_rate >= 0.90
