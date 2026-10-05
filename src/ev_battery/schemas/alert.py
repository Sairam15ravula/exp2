"""Alert schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class AlertCreate(BaseModel):
    """Schema for manual or automated alert creation."""
    cell_id: Optional[str] = None
    pack_id: Optional[str] = None
    alert_type: str = Field(description="Alert category")
    severity: str = Field(default="WARNING", pattern="^(WARNING|CRITICAL)$")
    message: str
    evidence: dict[str, Any] = {}


class AlertResponse(BaseModel):
    """Alert entity response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    cell_id: Optional[str]
    pack_id: Optional[str]
    alert_type: str
    severity: str
    message: str
    evidence: dict[str, Any]
    status: str
    created_at: datetime
    resolved_at: Optional[datetime]


class AlertStatusUpdate(BaseModel):
    """Schema to update an alert status."""
    status: str = Field(pattern="^(ACTIVE|ACKNOWLEDGED|RESOLVED)$")
