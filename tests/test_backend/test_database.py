"""Tests for SQLAlchemy database schema and relationships.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from sqlalchemy import select

from ev_battery.db.models import (
    Alert,
    AnomalyEvent,
    BatteryPack,
    Cell,
    Cycle,
    ModelRegistry,
    Prediction,
    StateEstimate,
    Telemetry,
    User,
)


def test_battery_pack_and_cell_relationship(test_db):
    """Test creation and cascade deletion of BatteryPack and Cell."""
    pack = BatteryPack(
        id="PACK_TEST_01",
        name="Test 4S2P Module",
        chemistry="Li-ion NMC",
        series_cells=4,
        parallel_strings=2,
        nominal_cell_voltage_v=3.6,
        nominal_cell_capacity_ah=2.0,
        total_voltage_v=14.4,
        total_capacity_ah=4.0,
        is_synthetic=False,
    )
    cell1 = Cell(
        id="CELL_TEST_01",
        pack_id="PACK_TEST_01",
        cell_index=0,
        cell_serial="SN001",
        initial_capacity_ah=2.0,
        nominal_voltage_v=3.6,
        is_synthetic=False,
    )
    cell2 = Cell(
        id="CELL_TEST_02",
        pack_id="PACK_TEST_01",
        cell_index=1,
        cell_serial="SN002",
        initial_capacity_ah=2.0,
        nominal_voltage_v=3.6,
        is_synthetic=False,
    )
    test_db.add_all([pack, cell1, cell2])
    test_db.commit()

    retrieved_pack = test_db.get(BatteryPack, "PACK_TEST_01")
    assert retrieved_pack is not None
    assert len(retrieved_pack.cells) == 2
    assert retrieved_pack.total_voltage_v == 14.4
    assert retrieved_pack.total_capacity_ah == 4.0


def test_telemetry_and_cycle_records(test_db):
    """Test persisting time-series telemetry and cycle records."""
    cell = Cell(
        id="CELL_B0005",
        cell_index=0,
        cell_serial="NASA-B0005",
        initial_capacity_ah=1.85,
        nominal_voltage_v=3.6,
        is_synthetic=False,
    )
    test_db.add(cell)
    test_db.commit()

    tel = Telemetry(
        cell_id="CELL_B0005",
        voltage_v=3.82,
        current_a=2.0,
        temperature_c=24.5,
        soc_reported=0.75,
        cycle_count=1,
        is_synthetic=False,
    )
    cycle = Cycle(
        cell_id="CELL_B0005",
        cycle_number=1,
        capacity_ah=1.85,
        duration_s=3650.0,
        avg_temp_c=25.2,
        max_temp_c=32.0,
        is_synthetic=False,
    )
    test_db.add_all([tel, cycle])
    test_db.commit()

    assert tel.id is not None
    assert cycle.id is not None
    assert tel.cell.id == "CELL_B0005"
    assert cycle.cell.id == "CELL_B0005"


def test_state_estimate_and_prediction_records(test_db):
    """Test SOC state estimates and SOH/RUL predictions."""
    cell = Cell(
        id="CELL_PRED_01",
        cell_index=0,
        cell_serial="SN-PRED",
        initial_capacity_ah=2.0,
        nominal_voltage_v=3.6,
        is_synthetic=False,
    )
    test_db.add(cell)
    test_db.commit()

    state = StateEstimate(
        cell_id="CELL_PRED_01",
        soc_ekf=0.82,
        soc_cc=0.81,
        error_bound=0.01,
        is_synthetic=False,
    )
    pred_soh = Prediction(
        cell_id="CELL_PRED_01",
        cycle_number=25,
        target_type="SOH",
        predicted_value=0.915,
        is_synthetic=False,
    )
    pred_rul = Prediction(
        cell_id="CELL_PRED_01",
        cycle_number=25,
        target_type="RUL",
        predicted_value=75.0,
        lower_bound=68.0,
        upper_bound=84.0,
        confidence_interval=0.90,
        is_synthetic=False,
    )
    test_db.add_all([state, pred_soh, pred_rul])
    test_db.commit()

    assert state.id is not None
    assert pred_soh.id is not None
    assert pred_rul.lower_bound == 68.0
    assert pred_rul.upper_bound == 84.0


def test_anomaly_event_and_alert(test_db):
    """Test anomaly logging and alert resolution workflow."""
    cell = Cell(
        id="CELL_ANOM_01",
        cell_index=0,
        cell_serial="SN-ANOM",
        initial_capacity_ah=2.0,
        is_synthetic=True,
    )
    test_db.add(cell)
    test_db.commit()

    event = AnomalyEvent(
        cell_id="CELL_ANOM_01",
        cycle_number=40,
        anomaly_type="THERMAL_SPIKE",
        severity="CRITICAL",
        score=92.5,
        evidence={"peak_temp_c": 56.4, "heating_rate": 0.12},
        is_synthetic=True,
    )
    alert = Alert(
        cell_id="CELL_ANOM_01",
        alert_type="THERMAL_RUNAWAY_RISK",
        severity="CRITICAL",
        message="Critical thermal spike detected on cell CELL_ANOM_01",
        evidence={"score": 92.5},
        status="ACTIVE",
    )
    test_db.add_all([event, alert])
    test_db.commit()

    assert event.id is not None
    assert alert.id is not None
    assert alert.status == "ACTIVE"
    assert alert.resolved_at is None

    # Resolve alert
    alert.status = "RESOLVED"
    alert.resolved_at = datetime.now(timezone.utc)
    test_db.commit()

    retrieved = test_db.get(Alert, alert.id)
    assert retrieved.status == "RESOLVED"
    assert retrieved.resolved_at is not None


def test_synthetic_labelling_across_all_models(test_db):
    """Rule 8: Verify is_synthetic boolean flag persists accurately on all relevant entities."""
    pack = BatteryPack(
        id="SYNTH_PACK_99",
        name="Synthetic Evaluation Pack",
        series_cells=2,
        parallel_strings=1,
        nominal_cell_voltage_v=3.6,
        nominal_cell_capacity_ah=2.0,
        total_voltage_v=7.2,
        total_capacity_ah=2.0,
        is_synthetic=True,
    )
    cell = Cell(
        id="SYNTH_CELL_99",
        pack_id="SYNTH_PACK_99",
        cell_index=0,
        cell_serial="SYNTH-99",
        initial_capacity_ah=2.0,
        nominal_voltage_v=3.6,
        is_synthetic=True,
    )
    tel = Telemetry(
        cell_id="SYNTH_CELL_99",
        voltage_v=3.6,
        current_a=1.0,
        temperature_c=25.0,
        is_synthetic=True,
    )
    test_db.add_all([pack, cell, tel])
    test_db.commit()

    assert pack.is_synthetic is True
    assert cell.is_synthetic is True
    assert tel.is_synthetic is True


def test_model_registry_provenance(test_db):
    """Rule 6: Verify ModelRegistry stores all required reproducibility fields."""
    entry = ModelRegistry(
        model_name="XGBoost_SOH_LOBO",
        target_type="SOH",
        version="v1.0.0",
        file_path="models/soh_xgboost.joblib",
        dataset_version="nasa_2026_q1",
        data_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        feature_list=["voltage_mean", "temp_rise", "internal_resistance_proxy"],
        metrics={"rmse": 0.0142, "baseline_linear_rmse": 0.0266},
        library_versions={"xgboost": "2.1.3", "scikit-learn": "1.6.0"},
        is_active=True,
    )
    test_db.add(entry)
    test_db.commit()

    retrieved = test_db.get(ModelRegistry, entry.id)
    assert retrieved.version == "v1.0.0"
    assert retrieved.metrics["rmse"] == 0.0142
    assert "xgboost" in retrieved.library_versions
