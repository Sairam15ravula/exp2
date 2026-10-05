"""Tests for Isolation Forest and Hybrid Anomaly Detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_battery.anomaly.detector import (
    ANOMALY_FEATURE_COLUMNS,
    HybridAnomalyDetector,
    IsolationForestDetector,
)
from ev_battery.data.features import extract_all_features
from ev_battery.data.synthetic import generate_synthetic_battery, generate_synthetic_with_fault


class TestIsolationForestDetector:
    @pytest.fixture
    def normal_feature_df(self):
        cycles = generate_synthetic_battery("SYNTH_TRAIN", n_cycles=60, seed=42)
        df = extract_all_features(cycles)
        return df[ANOMALY_FEATURE_COLUMNS]

    def test_fit_and_predict(self, normal_feature_df):
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(normal_feature_df)
        assert detector.is_fitted

        preds = detector.predict(normal_feature_df)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(normal_feature_df)
        # In training data, the majority should be classified as normal (False)
        assert np.mean(preds == False) > 0.85  # noqa: E712

    def test_detects_severe_outlier(self, normal_feature_df):
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(normal_feature_df)

        # Create extreme synthetic outlier (e.g. 70°C temp rise and huge IR)
        outlier_row = pd.DataFrame([{
            "voltage_drop_rate": 0.05,
            "voltage_min": 2.5,
            "temp_max": 75.0,
            "temp_rise": 45.0,
            "internal_resistance_proxy": 1.2,
            "duration_s": 500.0,
        }])

        is_anom = detector.predict(outlier_row)
        score = detector.decision_function(outlier_row)

        assert is_anom[0] == True  # noqa: E712
        assert score[0] < 0.0  # Strongly negative decision score

    def test_rule_2_target_leakage_prevented(self):
        """Rule 2: Anomaly features must not include target columns."""
        with pytest.raises(ValueError, match="Target leakage detected"):
            IsolationForestDetector(feature_names=["voltage_min", "capacity"])

    def test_rule_6_metadata_stored(self, normal_feature_df):
        """Rule 6: Training metadata must be tracked."""
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(normal_feature_df, dataset_version="synth-norm-v1")

        meta = detector.training_metadata
        assert meta["dataset_version"] == "synth-norm-v1"
        assert len(meta["data_sha256"]) == 64
        assert meta["random_seed"] == 42
        assert "scikit-learn" in meta["library_versions"]

    def test_unfitted_raises_runtime_error(self):
        detector = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="must be fitted"):
            detector.predict(pd.DataFrame())


class TestHybridAnomalyDetector:
    def test_hybrid_diagnosis_classifications(self):
        # Train ML detector on normal cycles
        normal_cycles = generate_synthetic_battery("SYNTH_NORM", n_cycles=40, seed=1)
        train_df = extract_all_features(normal_cycles)[ANOMALY_FEATURE_COLUMNS]
        ml_detector = IsolationForestDetector(contamination=0.05, random_state=42)
        ml_detector.fit(train_df)

        hybrid = HybridAnomalyDetector(ml_detector, residual_threshold_rms=0.075)

        # 1. Normal cycle diagnosis
        norm_diag = hybrid.diagnose_cycle(normal_cycles[0])
        assert norm_diag["anomaly_type"] in ["NORMAL", "MULTIVARIATE_FEATURE_DRIFT"]
        assert "rms_residual_v" in norm_diag
        assert "ml_anomaly_score" in norm_diag

        # 2. Faulted cycle diagnosis (with sudden capacity drop and heat spike)
        fault_cycles = generate_synthetic_with_fault("SYNTH_FAULT", n_cycles=60, fault_cycle=30, seed=1)
        fault_diag = hybrid.diagnose_cycle(fault_cycles[35])  # Cycle 36 is faulted

        assert fault_diag["is_anomaly"] == True  # noqa: E712
        assert fault_diag["anomaly_type"] in ["CRITICAL_HYBRID_ANOMALY", "PHYSICS_RESIDUAL_SURGE", "MULTIVARIATE_FEATURE_DRIFT"]
