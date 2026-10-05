"""Tests for BatterySHAPExplainer (TreeSHAP exact attributions and local efficiency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_battery.explain.shap_explainer import BatterySHAPExplainer
from ev_battery.rul.model import RULModel
from ev_battery.soh.dataset import FEATURE_COLUMNS
from ev_battery.soh.model import SOHModel


@pytest.fixture
def fitted_soh_model():
    """Create and fit a deterministic SOH model."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "cycle_number": np.arange(1, n + 1),
        "voltage_max": np.linspace(4.2, 4.0, n),
        "voltage_min": np.linspace(2.7, 2.5, n),
        "voltage_mean": np.linspace(3.7, 3.4, n),
        "voltage_drop_rate": np.linspace(0.001, 0.003, n),
        "temp_max": np.linspace(26.0, 34.0, n),
        "temp_min": np.full(n, 24.0),
        "temp_rise": np.linspace(2.0, 10.0, n),
        "internal_resistance_proxy": np.linspace(0.05, 0.14, n),
        "time_in_voltage_window": np.linspace(3600.0, 2200.0, n),
        "duration_s": np.linspace(3800.0, 2400.0, n),
    })
    y = pd.Series(np.linspace(1.0, 0.72, n))
    model = SOHModel(n_estimators=25, max_depth=3)
    model.fit(X, y)
    return model, X


@pytest.fixture
def fitted_rul_model():
    """Create and fit a deterministic RUL quantile model."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "cycle_number": np.arange(1, n + 1),
        "voltage_max": np.linspace(4.2, 4.0, n),
        "voltage_min": np.linspace(2.7, 2.5, n),
        "voltage_mean": np.linspace(3.7, 3.4, n),
        "voltage_drop_rate": np.linspace(0.001, 0.003, n),
        "temp_max": np.linspace(26.0, 34.0, n),
        "temp_min": np.full(n, 24.0),
        "temp_rise": np.linspace(2.0, 10.0, n),
        "internal_resistance_proxy": np.linspace(0.05, 0.14, n),
        "time_in_voltage_window": np.linspace(3600.0, 2200.0, n),
        "duration_s": np.linspace(3800.0, 2400.0, n),
    })
    y = pd.Series(np.maximum(0.0, np.linspace(100.0, 0.0, n)))
    model = RULModel(n_estimators=25, max_depth=3)
    model.fit(X, y)
    return model, X


def test_soh_shap_explainer_local_additivity(fitted_soh_model):
    """Verify local efficiency: base_value + sum(shap_values) == predicted_value."""
    model, X = fitted_soh_model
    explainer = BatterySHAPExplainer(model, target_type="SOH")

    sample_features = X.iloc[15].to_dict()
    explanation = explainer.explain_instance(sample_features)

    assert explanation.target_type == "SOH"
    assert explanation.is_additive is True
    assert len(explanation.contributions) == len(FEATURE_COLUMNS)

    # Check manual sum
    total_shap = sum(c.shap_value for c in explanation.contributions)
    reconstructed = explainer.base_value + total_shap
    assert pytest.approx(reconstructed, 0.001) == explanation.predicted_value


def test_soh_shap_top_drivers_ordering(fitted_soh_model):
    """Verify top drivers are sorted by impact."""
    model, X = fitted_soh_model
    explainer = BatterySHAPExplainer(model, target_type="SOH")

    sample_features = X.iloc[45].to_dict()
    explanation = explainer.explain_instance(sample_features)

    # Verify contributions are descending by absolute impact
    for i in range(len(explanation.contributions) - 1):
        assert explanation.contributions[i].abs_impact >= explanation.contributions[i + 1].abs_impact

    # Check top negative drivers (pushing SOH down)
    if explanation.top_negative_drivers:
        assert all(c.shap_value < 0 for c in explanation.top_negative_drivers)


def test_global_shap_importance(fitted_soh_model):
    """Verify global feature importance summary is sorted."""
    model, X = fitted_soh_model
    explainer = BatterySHAPExplainer(model, target_type="SOH")

    global_imp = explainer.explain_global(X)
    assert len(global_imp) == len(FEATURE_COLUMNS)
    values = list(global_imp.values())
    assert values == sorted(values, reverse=True)


def test_rul_shap_explainer(fitted_rul_model):
    """Verify SHAP explanation works for RUL model."""
    model, X = fitted_rul_model
    explainer = BatterySHAPExplainer(model, target_type="RUL")

    sample = X.iloc[10].to_dict()
    explanation = explainer.explain_instance(sample)

    assert explanation.target_type == "RUL"
    assert explanation.is_additive is True
    assert explanation.predicted_value >= 0.0
