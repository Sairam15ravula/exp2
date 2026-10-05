"""Battery pack and cell management endpoints (Rule 5 & Rule 8).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ev_battery.db.models import BatteryPack, Cell, User
from ev_battery.db.session import get_db
from ev_battery.schemas.pack import CellCreate, CellResponse, PackCreate, PackResponse
from ev_battery.security.auth import require_write_access

router = APIRouter(prefix="/packs", tags=["Battery Packs & Cells"])


@router.get("", response_model=list[PackResponse])
def list_packs(db: Session = Depends(get_db)):
    """List all battery packs registered in the system."""
    stmt = select(BatteryPack).options(selectinload(BatteryPack.cells))
    packs = db.scalars(stmt).all()
    return packs


@router.post("", response_model=PackResponse, status_code=status.HTTP_201_CREATED)
def create_pack(
    pack_in: PackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access),
):
    """Register a new battery pack and scale pack-to-cell specs (Rule 5 & Rule 7)."""
    existing = db.get(BatteryPack, pack_in.id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Battery pack with id '{pack_in.id}' already exists",
        )

    # Scale total voltage and capacity from series and parallel cell specs
    total_voltage = pack_in.series_cells * pack_in.nominal_cell_voltage_v
    total_capacity = pack_in.parallel_strings * pack_in.nominal_cell_capacity_ah

    pack = BatteryPack(
        id=pack_in.id,
        name=pack_in.name,
        chemistry=pack_in.chemistry,
        series_cells=pack_in.series_cells,
        parallel_strings=pack_in.parallel_strings,
        nominal_cell_voltage_v=pack_in.nominal_cell_voltage_v,
        nominal_cell_capacity_ah=pack_in.nominal_cell_capacity_ah,
        total_voltage_v=total_voltage,
        total_capacity_ah=total_capacity,
        is_synthetic=pack_in.is_synthetic,
    )
    db.add(pack)

    # Auto-generate individual cells if requested
    if pack_in.auto_generate_cells:
        total_cells = pack_in.series_cells * pack_in.parallel_strings
        for idx in range(total_cells):
            cell_id = f"{pack_in.id}_C{idx+1:02d}"
            cell = Cell(
                id=cell_id,
                pack_id=pack_in.id,
                cell_index=idx,
                cell_serial=f"SER-{pack_in.id}-{idx+1:03d}",
                initial_capacity_ah=pack_in.nominal_cell_capacity_ah,
                nominal_voltage_v=pack_in.nominal_cell_voltage_v,
                is_synthetic=pack_in.is_synthetic,
            )
            db.add(cell)

    db.commit()
    db.refresh(pack)

    # Re-query with loaded cells
    stmt = select(BatteryPack).where(BatteryPack.id == pack.id).options(selectinload(BatteryPack.cells))
    return db.scalar(stmt)


@router.get("/{pack_id}", response_model=PackResponse)
def get_pack(pack_id: str, db: Session = Depends(get_db)):
    """Retrieve details for a specific battery pack."""
    stmt = select(BatteryPack).where(BatteryPack.id == pack_id).options(selectinload(BatteryPack.cells))
    pack = db.scalar(stmt)
    if not pack:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Battery pack '{pack_id}' not found",
        )
    return pack


@router.get("/{pack_id}/cells", response_model=list[CellResponse])
def get_pack_cells(pack_id: str, db: Session = Depends(get_db)):
    """Retrieve all cell units belonging to a pack."""
    pack = db.get(BatteryPack, pack_id)
    if not pack:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pack not found")

    stmt = select(Cell).where(Cell.pack_id == pack_id).order_by(Cell.cell_index)
    return db.scalars(stmt).all()


@router.post("/{pack_id}/cells", response_model=CellResponse, status_code=status.HTTP_201_CREATED)
def add_cell_to_pack(
    pack_id: str,
    cell_in: CellCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access),
):
    """Manually add an individual cell record to a pack."""
    pack = db.get(BatteryPack, pack_id)
    if not pack:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pack not found")

    existing_cell = db.get(Cell, cell_in.id)
    if existing_cell:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cell '{cell_in.id}' already exists",
        )

    cell = Cell(
        id=cell_in.id,
        pack_id=pack_id,
        cell_index=cell_in.cell_index,
        cell_serial=cell_in.cell_serial,
        initial_capacity_ah=cell_in.initial_capacity_ah,
        nominal_voltage_v=cell_in.nominal_voltage_v,
        is_synthetic=cell_in.is_synthetic or pack.is_synthetic,
    )
    db.add(cell)
    db.commit()
    db.refresh(cell)
    return cell
