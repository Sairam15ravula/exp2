"""Tests for SOH explainability (feature importance and TreeSHAP)."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.soh.dataset import get_feature_matrix_and_target, prepare_soh_dataset
from ev_battery.soh.explain import compute_shap_values, get_feature_importances
from ev_battery.soh.model import SOHModel


class TestSOHExplainability:
    @pytest.fixture
    def fitted_model_and_data(self):
        cycles = generate_synthetic_battery("SYNTH_001", n_cycles=30, seed=42)
        df = prepare_soh_dataset(cycles)
        X, y = get_feature_matrix_and_target(df)
        model = SOHModel(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X, y)
        return model, X

    def test_get_feature_importances(self, fitted_model_and_data):
        model, _ = fitted_model_and_data
        importances = get_feature_importances(model)

        assert isinstance(importances, dict)
        assert len(importances) == len(model.feature_names)
        # Importances should be non-negative
        for feat, score in importances.items():
            assert score >= 0.0

    def test_compute_shap_values(self, fitted_model_and_data):
        model, X = fitted_model_and_data
        shap_vals, shap_summary = compute_shap_values(model, X)

        assert isinstance(shap_vals, np.ndarray)
        assert shap_vals.shape == X.shape
        assert len(shap_summary) == X.shape[1]
        for feat, imp in shap_summary.items():
            assert imp >= 0.0

    def test_unfitted_model_raises_error(self):
        model = SOHModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            get_feature_importances(model)
