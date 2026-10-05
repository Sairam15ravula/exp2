"""Tests for explainability REST API endpoints (/api/v1/explain).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ev_battery.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_explain_soh_endpoint_valid(client):
    """Test SOH explain endpoint returning SHAP values, rules, and narrative."""
    payload = {
        "cell_id": "CELL_EXP_01",
        "cycle_number": 25,
        "features": {
            "voltage_max": 4.18,
            "voltage_min": 2.65,
            "voltage_mean": 3.65,
            "voltage_drop_rate": 0.0016,
            "temp_max": 29.5,
            "temp_min": 24.0,
            "temp_rise": 5.5,
            "internal_resistance_proxy": 0.068,
            "time_in_voltage_window": 3100.0,
            "duration_s": 3300.0,
        },
        "is_synthetic": False,
    }
    res = client.post("/api/v1/explain/soh", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["cell_id"] == "CELL_EXP_01"
    assert data["cycle_number"] == 25
    assert data["target_type"] == "SOH"
    assert 0.5 <= data["predicted_value"] <= 1.2
    assert len(data["feature_contributions"]) > 0
    assert len(data["rule_findings"]) > 0
    assert len(data["narrative_explanation"]) > 20
    assert data["is_grounded"] is True


def test_api_explain_soh_rejects_target_leakage(client):
    """Rule 2: Explain SOH endpoint must fail with 422 if capacity or initial_capacity is supplied."""
    leaky_payload = {
        "cell_id": "CELL_LEAK_EXP",
        "cycle_number": 10,
        "features": {
            "capacity": 1.75,  # TARGET LEAKAGE!
            "voltage_mean": 3.7,
            "temp_rise": 4.0,
        },
    }
    res = client.post("/api/v1/explain/soh", json=leaky_payload)
    assert res.status_code == 422
    assert "target leakage" in res.text.lower()


def test_api_explain_rul_endpoint(client):
    """Rule 4: Explain RUL endpoint returns prediction interval and EOL trajectory explanation."""
    payload = {
        "cell_id": "CELL_RUL_EXP",
        "cycle_number": 40,
        "features": {
            "voltage_max": 4.16,
            "voltage_min": 2.62,
            "voltage_mean": 3.62,
            "voltage_drop_rate": 0.0018,
            "temp_max": 31.0,
            "temp_min": 24.0,
            "temp_rise": 7.0,
            "internal_resistance_proxy": 0.075,
            "duration_s": 3200.0,
        },
        "is_synthetic": True,
    }
    res = client.post("/api/v1/explain/rul", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["target_type"] == "RUL"
    assert data["lower_bound"] is not None
    assert data["upper_bound"] is not None
    assert data["lower_bound"] <= data["predicted_value"] <= data["upper_bound"]
    assert len(data["feature_contributions"]) > 0
    assert "80.0%" in data["narrative_explanation"]  # Rule 4: EOL-80%
    assert data["is_grounded"] is True
    assert data["is_synthetic"] is True
