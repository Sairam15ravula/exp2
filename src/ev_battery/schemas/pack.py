"""Battery pack and cell schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class CellCreate(BaseModel):
    """Schema for manual cell creation."""
    id: str
    cell_index: int = 0
    cell_serial: str
    initial_capacity_ah: float = Field(gt=0)
    nominal_voltage_v: float = Field(default=3.6, gt=0)
    is_synthetic: bool = False


class CellResponse(BaseModel):
    """Cell response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    pack_id: Optional[str] = None
    cell_index: int
    cell_serial: str
    initial_capacity_ah: float
    nominal_voltage_v: float
    is_synthetic: bool
    created_at: datetime


class PackCreate(BaseModel):
    """Battery pack registration schema (Rule 5: pack architecture)."""
    id: str
    name: str
    chemistry: str = "Li-ion NMC"
    series_cells: int = Field(gt=0, description="Number of series connected cells (Ns)")
    parallel_strings: int = Field(gt=0, description="Number of parallel strings (Np)")
    nominal_cell_voltage_v: float = Field(default=3.6, gt=0)
    nominal_cell_capacity_ah: float = Field(default=2.0, gt=0)
    is_synthetic: bool = False
    auto_generate_cells: bool = Field(default=True, description="Automatically instantiate cell records")


class PackResponse(BaseModel):
    """Battery pack response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    chemistry: str
    series_cells: int
    parallel_strings: int
    nominal_cell_voltage_v: float
    nominal_cell_capacity_ah: float
    total_voltage_v: float
    total_capacity_ah: float
    is_synthetic: bool
    created_at: datetime
    cells: list[CellResponse] = []
