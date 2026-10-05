"""Telemetry ingestion and validation schemas (Rule 5 & Rule 8).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TelemetryReading(BaseModel):
    """Raw cell-level telemetry data point."""
    cell_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    voltage_v: float = Field(description="Cell terminal voltage in Volts")
    current_a: float = Field(description="Cell current in Amperes (>0 discharge, <0 charge)")
    temperature_c: float = Field(description="Cell surface temperature in Celsius")
    soc_reported: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cycle_count: Optional[int] = Field(default=None, ge=0)
    is_synthetic: bool = Field(default=False, description="Rule 8: synthetic data label")


class PackTelemetryIngestRequest(BaseModel):
    """Pack-level telemetry ingestion request to be scaled into cell values (Rule 5)."""
    pack_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pack_voltage_v: float = Field(gt=0, description="Total pack voltage in Volts")
    pack_current_a: float = Field(description="Total pack current in Amperes")
    pack_temperature_c: float = Field(description="Average pack temperature in Celsius")
    is_synthetic: bool = False


class TelemetryBatchIngestRequest(BaseModel):
    """Batch ingestion of cell telemetry readings."""
    readings: list[TelemetryReading]


class TelemetryResponse(BaseModel):
    """Telemetry record response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    cell_id: str
    timestamp: datetime
    voltage_v: float
    current_a: float
    temperature_c: float
    soc_reported: Optional[float]
    cycle_count: Optional[int]
    is_synthetic: bool


class IngestSummaryResponse(BaseModel):
    """Result of a telemetry ingestion operation."""
    ingested_count: int
    cell_ids: list[str]
    warnings: list[str] = []
    is_synthetic: bool = False
