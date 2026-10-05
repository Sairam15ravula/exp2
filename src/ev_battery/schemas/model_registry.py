"""Model registry schemas (Rule 6).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ModelRegistryCreate(BaseModel):
    """Schema to register a trained model artifact."""
    model_name: str
    target_type: str = Field(pattern="^(SOH|RUL|ANOMALY)$")
    version: str
    file_path: str
    dataset_version: str
    data_sha256: str
    feature_list: list[str]
    metrics: dict[str, Any]
    library_versions: dict[str, str]
    is_active: bool = True


class ModelRegistryResponse(BaseModel):
    """Model registry response."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_name: str
    target_type: str
    version: str
    file_path: str
    dataset_version: str
    data_sha256: str
    feature_list: list[str]
    metrics: dict[str, Any]
    library_versions: dict[str, str]
    is_active: bool
    created_at: datetime
