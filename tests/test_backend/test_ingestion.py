"""Tests for telemetry ingestion, Rule 5 pack-to-cell scaling, and Rule 8 synthetic labeling.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from ev_battery.db.models import Cell, Telemetry


def test_cell_telemetry_ingest_normal(client, engineer_headers, test_db):
    """Test normal cell telemetry ingestion."""
    payload = {
        "readings": [
            {
                "cell_id": "CELL_INGEST_01",
                "voltage_v": 3.75,
                "current_a": 1.5,
                "temperature_c": 26.5,
                "soc_reported": 0.65,
                "cycle_count": 5,
                "is_synthetic": False,
            }
        ]
    }
    res = client.post("/api/v1/telemetry/ingest", json=payload, headers=engineer_headers)
    assert res.status_code == 201
    data = res.json()
    assert data["ingested_count"] == 1
    assert data["cell_ids"] == ["CELL_INGEST_01"]
    assert len(data["warnings"]) == 0
    assert data["is_synthetic"] is False

    # Check database record
    stmt = select(Telemetry).where(Telemetry.cell_id == "CELL_INGEST_01")
    record = test_db.scalar(stmt)
    assert record is not None
    assert record.voltage_v == 3.75
    assert record.current_a == 1.5
    assert record.is_synthetic is False


def test_cell_telemetry_out_of_range_warnings(client, engineer_headers):
    """Rule 5: Out-of-range cell voltage (<2.5V or >4.2V) and temp generates warnings."""
    payload = {
        "readings": [
            {
                "cell_id": "CELL_WARN_01",
                "voltage_v": 1.80,  # Below 2.5V
                "current_a": 1.0,
                "temperature_c": 115.0,  # Above 100°C
                "is_synthetic": False,
            }
        ]
    }
    res = client.post("/api/v1/telemetry/ingest", json=payload, headers=engineer_headers)
    assert res.status_code == 201
    data = res.json()
    assert len(data["warnings"]) == 2
    assert any("voltage" in w.lower() for w in data["warnings"])
    assert any("temperature" in w.lower() for w in data["warnings"])


def test_pack_telemetry_scaling_to_cells(client, engineer_headers, test_db):
    """Rule 5: Pack telemetry must be scaled to cell-level via Ns and Np."""
    # 1. Create a 4S2P Pack
    pack_payload = {
        "id": "PACK_4S2P_SCALE",
        "name": "Scaling Verification Pack",
        "series_cells": 4,
        "parallel_strings": 2,
        "nominal_cell_voltage_v": 3.6,
        "nominal_cell_capacity_ah": 2.0,
        "is_synthetic": False,
        "auto_generate_cells": True,
    }
    create_res = client.post("/api/v1/packs", json=pack_payload, headers=engineer_headers)
    assert create_res.status_code == 201
    assert len(create_res.json()["cells"]) == 8  # 4 * 2 = 8 cells

    # 2. Ingest pack telemetry: 14.8V total, 4.4A total, 31.0°C
    pack_tel_payload = {
        "pack_id": "PACK_4S2P_SCALE",
        "pack_voltage_v": 14.8,  # Pack voltage -> 14.8 / 4 = 3.7V per cell
        "pack_current_a": 4.4,   # Pack current -> 4.4 / 2 = 2.2A per cell
        "pack_temperature_c": 31.0,
        "is_synthetic": False,
    }
    ingest_res = client.post("/api/v1/telemetry/ingest-pack", json=pack_tel_payload, headers=engineer_headers)
    assert ingest_res.status_code == 201
    data = ingest_res.json()
    assert data["ingested_count"] == 8
    assert len(data["warnings"]) == 0

    # 3. Verify in database that cell telemetry received scaled values
    stmt = select(Telemetry).join(Cell).where(Cell.pack_id == "PACK_4S2P_SCALE")
    telemetry_list = test_db.scalars(stmt).all()
    assert len(telemetry_list) == 8
    for t in telemetry_list:
        assert pytest.approx(t.voltage_v, 0.001) == 3.7
        assert pytest.approx(t.current_a, 0.001) == 2.2
        assert pytest.approx(t.temperature_c, 0.001) == 31.0


def test_synthetic_telemetry_label_persists(client, engineer_headers, test_db):
    """Rule 8: Ingesting synthetic data persists is_synthetic=True."""
    payload = {
        "readings": [
            {
                "cell_id": "SYNTH_TEST_CELL",
                "voltage_v": 3.65,
                "current_a": 2.0,
                "temperature_c": 25.0,
                "is_synthetic": True,
            }
        ]
    }
    res = client.post("/api/v1/telemetry/ingest", json=payload, headers=engineer_headers)
    assert res.status_code == 201
    assert res.json()["is_synthetic"] is True

    stmt = select(Telemetry).where(Telemetry.cell_id == "SYNTH_TEST_CELL")
    record = test_db.scalar(stmt)
    assert record.is_synthetic is True
