"""SQLAlchemy 2.0 ORM models for EV Battery Intelligence Platform.

Tables:
- user: Auth and server-side roles (Rule 7)
- battery_pack: Pack configuration (Rule 5) with synthetic flags (Rule 8)
- cell: Individual cell records
- telemetry: Time-series telemetry readings
- cycle: Aggregated cycle features
- state_estimate: SOC Kalman filter and Coulomb counting estimates
- prediction: SOH / RUL model predictions with uncertainty intervals (Rule 4)
- anomaly_event: Voltage residual and thermal risk events
- model_registry: Model provenance, metrics vs baselines, and metadata (Rule 3, 6)
- alerts: Safety alerts with evidence and resolution lifecycle
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ev_battery.db.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """User accounts for role-based access control (Rule 7)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="viewer", nullable=False)  # admin, engineer, viewer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class BatteryPack(Base, TimestampMixin):
    """Pack-level metadata and series/parallel architecture (Rule 5)."""

    __tablename__ = "battery_packs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    chemistry: Mapped[str] = mapped_column(String(64), default="Li-ion NMC", nullable=False)
    series_cells: Mapped[int] = mapped_column(Integer, nullable=False)
    parallel_strings: Mapped[int] = mapped_column(Integer, nullable=False)
    nominal_cell_voltage_v: Mapped[float] = mapped_column(Float, default=3.6, nullable=False)
    nominal_cell_capacity_ah: Mapped[float] = mapped_column(Float, default=2.0, nullable=False)
    total_voltage_v: Mapped[float] = mapped_column(Float, nullable=False)
    total_capacity_ah: Mapped[float] = mapped_column(Float, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Rule 8

    cells: Mapped[list["Cell"]] = relationship("Cell", back_populates="pack", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="pack")


class Cell(Base, TimestampMixin):
    """Individual cell records associated with pack or standalone testing."""

    __tablename__ = "cells"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pack_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("battery_packs.id", ondelete="CASCADE"), nullable=True)
    cell_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cell_serial: Mapped[str] = mapped_column(String(128), nullable=False)
    initial_capacity_ah: Mapped[float] = mapped_column(Float, nullable=False)
    nominal_voltage_v: Mapped[float] = mapped_column(Float, default=3.6, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Rule 8

    pack: Mapped[Optional["BatteryPack"]] = relationship("BatteryPack", back_populates="cells")
    telemetry: Mapped[list["Telemetry"]] = relationship("Telemetry", back_populates="cell", cascade="all, delete-orphan")
    cycles: Mapped[list["Cycle"]] = relationship("Cycle", back_populates="cell", cascade="all, delete-orphan")
    state_estimates: Mapped[list["StateEstimate"]] = relationship("StateEstimate", back_populates="cell", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship("Prediction", back_populates="cell", cascade="all, delete-orphan")
    anomaly_events: Mapped[list["AnomalyEvent"]] = relationship("AnomalyEvent", back_populates="cell", cascade="all, delete-orphan")
    alerts: Mapped[list["Alert"]] = relationship("Alert", back_populates="cell")


class Telemetry(Base):
    """High-frequency or periodic telemetry measurements (Rule 5)."""

    __tablename__ = "telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(64), ForeignKey("cells.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    voltage_v: Mapped[float] = mapped_column(Float, nullable=False)
    current_a: Mapped[float] = mapped_column(Float, nullable=False)
    temperature_c: Mapped[float] = mapped_column(Float, nullable=False)
    soc_reported: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cycle_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Rule 8

    cell: Mapped["Cell"] = relationship("Cell", back_populates="telemetry")


class Cycle(Base, TimestampMixin):
    """Segmented charge/discharge cycle features."""

    __tablename__ = "cycles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(64), ForeignKey("cells.id", ondelete="CASCADE"), nullable=False, index=True)
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    capacity_ah: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    avg_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    max_temp_c: Mapped[float] = mapped_column(Float, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Rule 8

    cell: Mapped["Cell"] = relationship("Cell", back_populates="cycles")


class StateEstimate(Base):
    """Real-time physics and observer state estimates (SOC)."""

    __tablename__ = "state_estimates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(64), ForeignKey("cells.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    soc_ekf: Mapped[float] = mapped_column(Float, nullable=False)
    soc_cc: Mapped[float] = mapped_column(Float, nullable=False)
    error_bound: Mapped[float] = mapped_column(Float, default=0.05, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    cell: Mapped["Cell"] = relationship("Cell", back_populates="state_estimates")


class Prediction(Base, TimestampMixin):
    """Machine learning predictions with uncertainty intervals (Rules 2, 4)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(64), ForeignKey("cells.id", ondelete="CASCADE"), nullable=False, index=True)
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    target_type: Mapped[str] = mapped_column(String(16), nullable=False)  # SOH or RUL
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_interval: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("model_registry.id"), nullable=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    cell: Mapped["Cell"] = relationship("Cell", back_populates="predictions")
    model: Mapped[Optional["ModelRegistry"]] = relationship("ModelRegistry")


class AnomalyEvent(Base, TimestampMixin):
    """Detected anomaly occurrences with evidence records (Phase 6)."""

    __tablename__ = "anomaly_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[str] = mapped_column(String(64), ForeignKey("cells.id", ondelete="CASCADE"), nullable=False, index=True)
    cycle_number: Mapped[int] = mapped_column(Integer, nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)  # NORMAL, ELEVATED, CRITICAL
    score: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    cell: Mapped["Cell"] = relationship("Cell", back_populates="anomaly_events")


class ModelRegistry(Base, TimestampMixin):
    """Model registry tracking reproducibility, dataset hashes, and metrics (Rule 6)."""

    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    target_type: Mapped[str] = mapped_column(String(32), nullable=False)  # SOH, RUL, ANOMALY
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    data_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_list: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    library_versions: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Alert(Base, TimestampMixin):
    """Actionable battery health alerts with evidence logs."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cell_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("cells.id", ondelete="CASCADE"), nullable=True)
    pack_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("battery_packs.id", ondelete="CASCADE"), nullable=True)
    alert_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)  # WARNING, CRITICAL
    message: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", nullable=False)  # ACTIVE, ACKNOWLEDGED, RESOLVED
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    cell: Mapped[Optional["Cell"]] = relationship("Cell", back_populates="alerts")
    pack: Mapped[Optional["BatteryPack"]] = relationship("BatteryPack", back_populates="alerts")
