"""Integration tests for all REST API endpoints.
"""

from __future__ import annotations

import pytest


def test_health_check_endpoints(client):
    """Test health check route."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["database_connected"] is True

    res2 = client.get("/api/v1/health")
    assert res2.status_code == 200


def test_pack_and_cell_queries(client, engineer_headers):
    """Test querying battery packs and child cells."""
    pack_payload = {
        "id": "PACK_QUERY_01",
        "name": "Query Pack",
        "series_cells": 3,
        "parallel_strings": 1,
        "nominal_cell_voltage_v": 3.6,
        "nominal_cell_capacity_ah": 2.0,
        "auto_generate_cells": True,
    }
    client.post("/api/v1/packs", json=pack_payload, headers=engineer_headers)

    # 1. List packs
    res_list = client.get("/api/v1/packs")
    assert res_list.status_code == 200
    packs = res_list.json()
    assert any(p["id"] == "PACK_QUERY_01" for p in packs)

    # 2. Get specific pack
    res_single = client.get("/api/v1/packs/PACK_QUERY_01")
    assert res_single.status_code == 200
    assert res_single.json()["total_voltage_v"] == pytest.approx(10.8)

    # 3. Get pack cells
    res_cells = client.get("/api/v1/packs/PACK_QUERY_01/cells")
    assert res_cells.status_code == 200
    assert len(res_cells.json()) == 3


def test_soc_inference_endpoint(client, engineer_headers):
    """Test SOC estimation endpoint with EKF and Coulomb counting."""
    stream_len = 20
    payload = {
        "cell_id": "CELL_SOC_TEST",
        "initial_soc": 1.0,
        "current_stream": [2.0] * stream_len,  # 2A discharge
        "voltage_stream": [4.15 - 0.02 * i for i in range(stream_len)],
        "dt_s": 1.0,
        "nominal_capacity_ah": 2.0,
    }
    res = client.post("/api/v1/inference/soc", json=payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["cell_id"] == "CELL_SOC_TEST"
    assert len(data["soc_history_ekf"]) == stream_len
    assert len(data["soc_history_cc"]) == stream_len
    assert 0.0 <= data["final_soc_ekf"] <= 1.0
    assert 0.0 <= data["final_soc_cc"] <= 1.0


def test_soh_inference_rejects_target_leakage(client, engineer_headers):
    """Rule 2: SOH inference MUST fail validation if capacity or initial_capacity is passed."""
    # Attempt with forbidden target feature 'capacity'
    leaky_payload_1 = {
        "cell_id": "CELL_LEAK_01",
        "cycle_number": 10,
        "features": {
            "capacity": 1.85,  # TARGET LEAKAGE!
            "voltage_mean": 3.7,
            "temp_rise": 4.5,
        },
    }
    res1 = client.post("/api/v1/inference/soh", json=leaky_payload_1, headers=engineer_headers)
    assert res1.status_code == 422  # Pydantic validation error
    assert "target leakage" in res1.text.lower()

    # Attempt with forbidden target feature 'initial_capacity'
    leaky_payload_2 = {
        "cell_id": "CELL_LEAK_02",
        "cycle_number": 10,
        "features": {
            "initial_capacity": 2.0,  # TARGET LEAKAGE!
            "voltage_mean": 3.7,
        },
    }
    res2 = client.post("/api/v1/inference/soh", json=leaky_payload_2, headers=engineer_headers)
    assert res2.status_code == 422
    assert "target leakage" in res2.text.lower()


def test_soh_inference_valid_features(client, engineer_headers):
    """Test valid SOH inference with non-leaky features."""
    valid_payload = {
        "cell_id": "CELL_SOH_VALID",
        "cycle_number": 20,
        "features": {
            "voltage_max": 4.18,
            "voltage_min": 2.65,
            "voltage_mean": 3.65,
            "voltage_drop_rate": 0.0015,
            "temp_max": 28.5,
            "temp_min": 24.0,
            "temp_rise": 4.5,
            "internal_resistance_proxy": 0.065,
            "time_in_voltage_window": 3200.0,
            "duration_s": 3400.0,
        },
    }
    res = client.post("/api/v1/inference/soh", json=valid_payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["target_type"] == "SOH"
    assert 0.5 <= data["predicted_value"] <= 1.2


def test_rul_inference_endpoint(client, engineer_headers):
    """Rule 4: RUL inference returns median and 90% prediction interval."""
    payload = {
        "cell_id": "CELL_RUL_TEST",
        "cycle_number": 15,
        "features": {
            "voltage_max": 4.18,
            "voltage_min": 2.65,
            "voltage_mean": 3.65,
            "temp_max": 28.0,
            "temp_rise": 4.0,
            "internal_resistance_proxy": 0.06,
            "duration_s": 3400.0,
        },
    }
    res = client.post("/api/v1/inference/rul", json=payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["target_type"] == "RUL"
    assert data["confidence_interval"] == 0.90
    assert data["lower_bound"] is not None
    assert data["upper_bound"] is not None
    assert data["lower_bound"] <= data["predicted_value"] <= data["upper_bound"]


def test_anomaly_inference_and_alert_generation(client, engineer_headers):
    """Phase 6 integration: Thermal risk anomaly detection triggers alert generation."""
    # 1. Normal cycle
    normal_payload = {
        "cell_id": "CELL_ANOM_TEST",
        "cycle_number": 5,
        "features": {
            "temp_max": 26.0,
            "temp_rise": 2.0,
            "duration_s": 3600.0,
            "internal_resistance_proxy": 0.05,
        },
    }
    res_normal = client.post("/api/v1/inference/anomaly", json=normal_payload, headers=engineer_headers)
    assert res_normal.status_code == 200
    data_normal = res_normal.json()
    assert data_normal["is_anomaly"] is False
    assert data_normal["thermal_severity"] == "NORMAL"

    # 2. Critical thermal runaway risk scenario
    critical_payload = {
        "cell_id": "CELL_ANOM_TEST",
        "cycle_number": 30,
        "max_temp_c": 58.5,
        "temp_rise_rate": 0.15,
        "features": {
            "temp_max": 58.5,
            "temp_rise": 33.5,
            "duration_s": 1200.0,
            "internal_resistance_proxy": 0.22,
        },
    }
    res_crit = client.post("/api/v1/inference/anomaly", json=critical_payload, headers=engineer_headers)
    assert res_crit.status_code == 200
    data_crit = res_crit.json()
    assert data_crit["is_anomaly"] is True
    assert data_crit["thermal_severity"] == "CRITICAL"
    assert data_crit["thermal_risk_score"] > 60.0

    # 3. Verify that an alert was generated in /api/v1/alerts
    res_alerts = client.get("/api/v1/alerts?severity=CRITICAL")
    assert res_alerts.status_code == 200
    alerts = res_alerts.json()
    assert any(a["cell_id"] == "CELL_ANOM_TEST" for a in alerts)


def test_alert_acknowledgment_and_resolution(client, engineer_headers):
    """Test resolving and acknowledging alerts."""
    # Create manual alert
    create_res = client.post(
        "/api/v1/alerts",
        json={
            "cell_id": "CELL_ALERT_MGMT",
            "alert_type": "VOLTAGE_DEVIATION",
            "severity": "WARNING",
            "message": "Cell voltage dropped below threshold",
        },
        headers=engineer_headers,
    )
    assert create_res.status_code == 201
    alert_id = create_res.json()["id"]

    # Acknowledge
    ack_res = client.patch(
        f"/api/v1/alerts/{alert_id}/status",
        json={"status": "ACKNOWLEDGED"},
        headers=engineer_headers,
    )
    assert ack_res.status_code == 200
    assert ack_res.json()["status"] == "ACKNOWLEDGED"

    # Resolve
    resolve_res = client.patch(
        f"/api/v1/alerts/{alert_id}/status",
        json={"status": "RESOLVED"},
        headers=engineer_headers,
    )
    assert resolve_res.status_code == 200
    assert resolve_res.json()["status"] == "RESOLVED"
    assert resolve_res.json()["resolved_at"] is not None


def test_model_registry_endpoints(client, engineer_headers):
    """Rule 6: Register model with reproducibility metadata and query."""
    model_payload = {
        "model_name": "XGBoost_Monotonic_SOH",
        "target_type": "SOH",
        "version": "1.0.0",
        "file_path": "models/soh_xgboost.joblib",
        "dataset_version": "nasa_v1",
        "data_sha256": "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
        "feature_list": ["voltage_mean", "temp_rise", "internal_resistance_proxy"],
        "metrics": {"lobo_rmse_mean": 0.0142, "lobo_rmse_std": 0.0085},
        "library_versions": {"xgboost": "2.1.3", "scikit-learn": "1.6.0"},
        "is_active": True,
    }
    reg_res = client.post("/api/v1/models", json=model_payload, headers=engineer_headers)
    assert reg_res.status_code == 201
    model_id = reg_res.json()["id"]

    # List models
    list_res = client.get("/api/v1/models?target_type=SOH")
    assert list_res.status_code == 200
    assert any(m["id"] == model_id for m in list_res.json())

    # Get single model
    get_res = client.get(f"/api/v1/models/{model_id}")
    assert get_res.status_code == 200
    assert get_res.json()["data_sha256"] == model_payload["data_sha256"]
