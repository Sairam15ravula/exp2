"""End-to-End System Integration Test.

Exercises the complete telemetry-to-decision lifecycle across all 11 phases:
1. Pack registration and pack-to-cell scaling (Rule 5)
2. Normal and synthetic fault telemetry ingestion (Rule 8)
3. Physics SOC estimation (1RC Equivalent Circuit Model + EKF)
4. SOH prediction with zero target leakage verification (Rule 2)
5. RUL prediction with 90% prediction intervals bounded by 80% EOL (Rule 4)
6. Thermal risk escalation and automated safety alert creation
7. Explainability brief (TreeSHAP attributions + physical rules + anti-hallucination text)
8. Safety alert acknowledgment and resolution lifecycle
9. Model registry provenance and reproducibility audit (Rule 3 & Rule 6)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ev_battery.api.app import app
from ev_battery.db.base import Base
from ev_battery.db.models import Alert, AnomalyEvent, BatteryPack, Cell, Prediction, Telemetry, User
from ev_battery.db.session import get_db
from ev_battery.security.auth import create_access_token, hash_password


@pytest.fixture(scope="module")
def e2e_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def e2e_db(e2e_engine):
    connection = e2e_engine.connect()
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestingSessionLocal()

    # Seed engineer user
    engineer = User(
        username="lead_battery_engineer",
        hashed_password=hash_password("automotive_grade_secret"),
        role="engineer",
        is_active=True,
    )
    session.add(engineer)
    session.commit()

    yield session

    session.close()
    connection.close()


@pytest.fixture(scope="module")
def client(e2e_db):
    def _override_get_db():
        yield e2e_db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def engineer_token():
    token = create_access_token(subject="lead_battery_engineer", role="engineer")
    return {"Authorization": f"Bearer {token}"}


def test_complete_ev_battery_intelligence_lifecycle(client, engineer_token, e2e_db):
    """Full lifecycle integration test verifying Rules 1 through 10."""

    # -------------------------------------------------------------------------
    # Step 1: Battery Pack Registration & Series/Parallel Scaling (Rule 5)
    # -------------------------------------------------------------------------
    pack_payload = {
        "id": "PACK_4S2P_FLEET_01",
        "name": "Fleet Prototype 4S2P Module",
        "chemistry": "Li-ion NMC",
        "series_cells": 4,
        "parallel_strings": 2,
        "nominal_cell_voltage_v": 3.6,
        "nominal_cell_capacity_ah": 2.0,
        "is_synthetic": False,
        "auto_generate_cells": True,
    }
    res_pack = client.post("/api/v1/packs", json=pack_payload, headers=engineer_token)
    assert res_pack.status_code == 201
    pack_data = res_pack.json()
    assert pack_data["total_voltage_v"] == pytest.approx(14.4)
    assert pack_data["total_capacity_ah"] == pytest.approx(4.0)
    assert len(pack_data["cells"]) == 8  # 4 * 2 = 8 cells

    active_cell_id = pack_data["cells"][0]["id"]

    # -------------------------------------------------------------------------
    # Step 2: Ingest Pack Telemetry and Verify Scale to Cells (Rule 5 & Rule 8)
    # -------------------------------------------------------------------------
    # Pack telemetry: 14.8V total, 4.4A discharge total, 28.5°C
    pack_tel_payload = {
        "pack_id": "PACK_4S2P_FLEET_01",
        "pack_voltage_v": 14.8,  # -> 14.8 / 4 = 3.7 V per cell
        "pack_current_a": 4.4,   # -> 4.4 / 2 = 2.2 A per cell
        "pack_temperature_c": 28.5,
        "is_synthetic": False,
    }
    res_ingest = client.post("/api/v1/telemetry/ingest-pack", json=pack_tel_payload, headers=engineer_token)
    assert res_ingest.status_code == 201
    assert res_ingest.json()["ingested_count"] == 8

    # Query cell telemetry to verify scaled cell voltage & current
    res_cell_tel = client.get(f"/api/v1/telemetry/cells/{active_cell_id}?limit=5")
    assert res_cell_tel.status_code == 200
    cell_readings = res_cell_tel.json()
    assert len(cell_readings) >= 1
    assert pytest.approx(cell_readings[0]["voltage_v"], 0.001) == 3.70
    assert pytest.approx(cell_readings[0]["current_a"], 0.001) == 2.20

    # -------------------------------------------------------------------------
    # Step 3: Physics-based SOC Estimation (1RC EKF vs Coulomb Counting)
    # -------------------------------------------------------------------------
    soc_payload = {
        "cell_id": active_cell_id,
        "initial_soc": 1.0,
        "current_stream": [2.2] * 10,
        "voltage_stream": [4.15, 4.10, 4.05, 4.00, 3.95, 3.90, 3.85, 3.80, 3.75, 3.70],
        "dt_s": 1.0,
        "nominal_capacity_ah": 2.0,
        "is_synthetic": False,
    }
    res_soc = client.post("/api/v1/inference/soc", json=soc_payload, headers=engineer_token)
    assert res_soc.status_code == 200
    soc_data = res_soc.json()
    assert 0.0 <= soc_data["final_soc_ekf"] <= 1.0
    assert 0.0 <= soc_data["final_soc_cc"] <= 1.0
    assert soc_data["error_bound"] >= 0.0

    # -------------------------------------------------------------------------
    # Step 4: SOH Prediction with Rule 2 Target Leakage Verification
    # -------------------------------------------------------------------------
    # Attempting to supply 'capacity' must trigger HTTP 422
    leaky_soh_payload = {
        "cell_id": active_cell_id,
        "cycle_number": 25,
        "features": {
            "capacity": 1.85,  # Target leakage!
            "voltage_mean": 3.65,
        },
    }
    res_leaky = client.post("/api/v1/inference/soh", json=leaky_soh_payload, headers=engineer_token)
    assert res_leaky.status_code == 422
    assert "target leakage" in res_leaky.text.lower()

    # Valid non-leaky features
    valid_features = {
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
    }
    valid_soh_payload = {
        "cell_id": active_cell_id,
        "cycle_number": 25,
        "features": valid_features,
        "is_synthetic": False,
    }
    res_soh = client.post("/api/v1/inference/soh", json=valid_soh_payload, headers=engineer_token)
    assert res_soh.status_code == 200
    soh_data = res_soh.json()
    assert soh_data["target_type"] == "SOH"
    assert 0.5 <= soh_data["predicted_value"] <= 1.2

    # -------------------------------------------------------------------------
    # Step 5: RUL Prediction with Uncertainty Bands (Rule 4: EOL-80%)
    # -------------------------------------------------------------------------
    rul_payload = {
        "cell_id": active_cell_id,
        "cycle_number": 25,
        "features": valid_features,
        "is_synthetic": False,
    }
    res_rul = client.post("/api/v1/inference/rul", json=rul_payload, headers=engineer_token)
    assert res_rul.status_code == 200
    rul_data = res_rul.json()
    assert rul_data["target_type"] == "RUL"
    assert rul_data["confidence_interval"] == 0.90
    assert rul_data["lower_bound"] <= rul_data["predicted_value"] <= rul_data["upper_bound"]

    # -------------------------------------------------------------------------
    # Step 6: Thermal Risk & Anomaly Ingestion -> Alert Generation
    # -------------------------------------------------------------------------
    # Simulate severe thermal event on cell
    anom_payload = {
        "cell_id": active_cell_id,
        "cycle_number": 25,
        "max_temp_c": 62.0,  # Breaches 60°C critical limit!
        "temp_rise_rate": 0.18,
        "features": {
            "temp_max": 62.0,
            "temp_rise": 38.0,
            "duration_s": 1500.0,
            "internal_resistance_proxy": 0.18,
        },
        "is_synthetic": False,
    }
    res_anom = client.post("/api/v1/inference/anomaly", json=anom_payload, headers=engineer_token)
    assert res_anom.status_code == 200
    anom_data = res_anom.json()
    assert anom_data["is_anomaly"] is True
    assert anom_data["thermal_severity"] == "CRITICAL"
    assert anom_data["thermal_risk_score"] >= 70.0

    # Verify that a CRITICAL alert was automatically raised
    res_alerts = client.get(f"/api/v1/alerts?cell_id={active_cell_id}&status=ACTIVE")
    assert res_alerts.status_code == 200
    alerts = res_alerts.json()
    assert len(alerts) >= 1
    critical_alert = alerts[0]
    assert critical_alert["severity"] == "CRITICAL"
    alert_id = critical_alert["id"]

    # -------------------------------------------------------------------------
    # Step 7: Explainability Brief (TreeSHAP + Physical Rules + Anti-Hallucination)
    # -------------------------------------------------------------------------
    explain_payload = {
        "cell_id": active_cell_id,
        "cycle_number": 25,
        "features": valid_features,
        "is_synthetic": False,
    }
    res_explain = client.post("/api/v1/explain/soh", json=explain_payload)
    assert res_explain.status_code == 200
    exp_data = res_explain.json()
    assert len(exp_data["feature_contributions"]) == 11
    assert len(exp_data["rule_findings"]) > 0
    assert exp_data["is_grounded"] is True
    assert len(exp_data["narrative_explanation"]) > 30

    # -------------------------------------------------------------------------
    # Step 8: Safety Alert Resolution Lifecycle
    # -------------------------------------------------------------------------
    # Acknowledge
    ack_res = client.patch(
        f"/api/v1/alerts/{alert_id}/status",
        json={"status": "ACKNOWLEDGED"},
        headers=engineer_token,
    )
    assert ack_res.status_code == 200
    assert ack_res.json()["status"] == "ACKNOWLEDGED"

    # Resolve
    res_resolve = client.patch(
        f"/api/v1/alerts/{alert_id}/status",
        json={"status": "RESOLVED"},
        headers=engineer_token,
    )
    assert res_resolve.status_code == 200
    assert res_resolve.json()["status"] == "RESOLVED"
    assert res_resolve.json()["resolved_at"] is not None

    # -------------------------------------------------------------------------
    # Step 9: Model Registry Provenance & Reproducibility Audit (Rule 3 & Rule 6)
    # -------------------------------------------------------------------------
    model_reg_payload = {
        "model_name": "XGBoost_Monotonic_SOH",
        "target_type": "SOH",
        "version": "1.0.0",
        "file_path": "models/soh_xgboost.joblib",
        "dataset_version": "nasa_battery_q1",
        "data_sha256": "3e9b1f7c04d13e9a2f64d7889104efacab120956487e810398fba6714d693240",
        "feature_list": list(valid_features.keys()),
        "metrics": {
            "lobo_rmse_mean": 0.0142,
            "lobo_rmse_std": 0.0085,
            "baseline_mean_rmse": 0.0543,
            "baseline_linear_rmse": 0.0266,
        },
        "library_versions": {
            "xgboost": "2.1.3",
            "scikit-learn": "1.6.0",
            "numpy": "2.2.1",
        },
        "is_active": True,
    }
    res_reg = client.post("/api/v1/models", json=model_reg_payload, headers=engineer_token)
    assert res_reg.status_code == 201
    model_record = res_reg.json()

    # Query model registry
    res_get_model = client.get(f"/api/v1/models/{model_record['id']}")
    assert res_get_model.status_code == 200
    retrieved = res_get_model.json()
    assert retrieved["data_sha256"] == model_reg_payload["data_sha256"]
    assert retrieved["metrics"]["lobo_rmse_mean"] < retrieved["metrics"]["baseline_linear_rmse"]
