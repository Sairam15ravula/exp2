"""Alert management endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ev_battery.db.models import Alert, User
from ev_battery.db.session import get_db
from ev_battery.schemas.alert import AlertCreate, AlertResponse, AlertStatusUpdate
from ev_battery.security.auth import require_write_access

router = APIRouter(prefix="/alerts", tags=["Safety Alerts"])


@router.get("", response_model=list[AlertResponse])
def list_alerts(
    alert_status: Optional[str] = Query(None, alias="status"),
    severity: Optional[str] = Query(None),
    cell_id: Optional[str] = Query(None),
    pack_id: Optional[str] = Query(None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """List system alerts with filtering capabilities."""
    stmt = select(Alert).order_by(Alert.created_at.desc())
    if alert_status:
        stmt = stmt.where(Alert.status == alert_status.upper())
    if severity:
        stmt = stmt.where(Alert.severity == severity.upper())
    if cell_id:
        stmt = stmt.where(Alert.cell_id == cell_id)
    if pack_id:
        stmt = stmt.where(Alert.pack_id == pack_id)
    stmt = stmt.limit(limit)
    return db.scalars(stmt).all()


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: int, db: Session = Depends(get_db)):
    """Retrieve details of a specific alert."""
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert {alert_id} not found")
    return alert


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
def create_alert(
    alert_in: AlertCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access),
):
    """Manually raise a safety alert (Rule 7: write route requires auth)."""
    alert = Alert(
        cell_id=alert_in.cell_id,
        pack_id=alert_in.pack_id,
        alert_type=alert_in.alert_type,
        severity=alert_in.severity,
        message=alert_in.message,
        evidence=alert_in.evidence,
        status="ACTIVE",
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


@router.patch("/{alert_id}/status", response_model=AlertResponse)
def update_alert_status(
    alert_id: int,
    status_update: AlertStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access),
):
    """Acknowledge or resolve an active alert (Rule 7)."""
    alert = db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Alert {alert_id} not found")

    new_status = status_update.status.upper()
    alert.status = new_status
    if new_status == "RESOLVED":
        alert.resolved_at = datetime.now(timezone.utc)
    else:
        alert.resolved_at = None

    db.commit()
    db.refresh(alert)
    return alert
