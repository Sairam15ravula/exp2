"""Telemetry ingestion endpoints with Rule 5 unit scaling and Rule 8 synthetic labeling.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ev_battery.db.models import BatteryPack, Cell, Telemetry, User
from ev_battery.db.session import get_db
from ev_battery.schemas.telemetry import (
    IngestSummaryResponse,
    PackTelemetryIngestRequest,
    TelemetryBatchIngestRequest,
    TelemetryReading,
    TelemetryResponse,
)
from ev_battery.security.auth import require_write_access
from ev_battery.units import (
    pack_to_cell_current,
    pack_to_cell_voltage,
    validate_temperature,
    validate_voltage,
)

router = APIRouter(prefix="/telemetry", tags=["Telemetry Ingestion"])


@router.post("/ingest", response_model=IngestSummaryResponse, status_code=status.HTTP_201_CREATED)
def ingest_cell_telemetry(
    batch: TelemetryBatchIngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access),
):
    """Ingest cell-level telemetry readings with range validation (Rule 5 & Rule 7)."""
    if not batch.readings:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No telemetry readings provided")

    ingested_cell_ids = set()
    warnings = []
    telemetry_records = []

    for reading in batch.readings:
        # Rule 5: Physical validity check on cell voltage & temperature
        if not validate_voltage(reading.voltage_v):
            warnings.append(
                f"Cell '{reading.cell_id}' voltage {reading.voltage_v:.3f}V is outside safe range [2.5V, 4.2V]"
            )
        if not validate_temperature(reading.temperature_c):
            warnings.append(
                f"Cell '{reading.cell_id}' temperature {reading.temperature_c:.1f}°C is outside safe range [-40°C, 100°C]"
            )

        # Ensure cell exists; create standalone if not exists
        cell = db.get(Cell, reading.cell_id)
        if not cell:
            cell = Cell(
                id=reading.cell_id,
                cell_serial=f"STANDALONE-{reading.cell_id}",
                initial_capacity_ah=2.0,
                nominal_voltage_v=3.6,
                is_synthetic=reading.is_synthetic,
            )
            db.add(cell)
            db.flush()

        # Rule 8: Persist synthetic label
        record = Telemetry(
            cell_id=reading.cell_id,
            timestamp=reading.timestamp,
            voltage_v=reading.voltage_v,
            current_a=reading.current_a,
            temperature_c=reading.temperature_c,
            soc_reported=reading.soc_reported,
            cycle_count=reading.cycle_count,
            is_synthetic=reading.is_synthetic or cell.is_synthetic,
        )
        telemetry_records.append(record)
        ingested_cell_ids.add(reading.cell_id)

    db.add_all(telemetry_records)
    db.commit()

    return IngestSummaryResponse(
        ingested_count=len(telemetry_records),
        cell_ids=list(ingested_cell_ids),
        warnings=warnings,
        is_synthetic=any(r.is_synthetic for r in telemetry_records),
    )


@router.post("/ingest-pack", response_model=IngestSummaryResponse, status_code=status.HTTP_201_CREATED)
def ingest_pack_telemetry(
    req: PackTelemetryIngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access),
):
    """Ingest pack-level telemetry and scale to cell values via series/parallel config (Rule 5)."""
    pack = db.get(BatteryPack, req.pack_id)
    if not pack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Battery pack '{req.pack_id}' not found",
        )

    # Rule 5: Convert pack measurements to cell-level quantities
    cell_voltage = pack_to_cell_voltage(req.pack_voltage_v, pack.series_cells)
    cell_current = pack_to_cell_current(req.pack_current_a, pack.parallel_strings)
    cell_temperature = req.pack_temperature_c

    warnings = []
    if not validate_voltage(cell_voltage):
        warnings.append(
            f"Derived cell voltage {cell_voltage:.3f}V is out of range [2.5V, 4.2V] (pack: {req.pack_voltage_v:.1f}V)"
        )
    if not validate_temperature(cell_temperature):
        warnings.append(
            f"Cell temperature {cell_temperature:.1f}°C is out of range [-40°C, 100°C]"
        )

    # Ingest for all cells in the pack
    stmt = select(Cell).where(Cell.pack_id == pack.id)
    cells = db.scalars(stmt).all()
    if not cells:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Pack '{pack.id}' has no registered cells to receive scaled telemetry",
        )

    records = []
    for cell in cells:
        record = Telemetry(
            cell_id=cell.id,
            timestamp=req.timestamp,
            voltage_v=cell_voltage,
            current_a=cell_current,
            temperature_c=cell_temperature,
            is_synthetic=req.is_synthetic or pack.is_synthetic,
        )
        records.append(record)

    db.add_all(records)
    db.commit()

    return IngestSummaryResponse(
        ingested_count=len(records),
        cell_ids=[c.id for c in cells],
        warnings=warnings,
        is_synthetic=req.is_synthetic or pack.is_synthetic,
    )


@router.get("/cells/{cell_id}", response_model=list[TelemetryResponse])
def get_cell_telemetry(
    cell_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """Retrieve time-series telemetry records for a given cell."""
    cell = db.get(Cell, cell_id)
    if not cell:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Cell '{cell_id}' not found")

    stmt = select(Telemetry).where(Telemetry.cell_id == cell_id).order_by(Telemetry.timestamp.desc()).limit(limit)
    return db.scalars(stmt).all()
