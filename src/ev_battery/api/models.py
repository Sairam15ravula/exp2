"""Model Registry endpoints for auditing, provenance, and reproducibility (Rule 3 & Rule 6).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ev_battery.db.models import ModelRegistry, User
from ev_battery.db.session import get_db
from ev_battery.schemas.model_registry import ModelRegistryCreate, ModelRegistryResponse
from ev_battery.security.auth import require_write_access

router = APIRouter(prefix="/models", tags=["Model Registry"])


@router.get("", response_model=list[ModelRegistryResponse])
def list_registered_models(
    target_type: Optional[str] = Query(None, pattern="^(SOH|RUL|ANOMALY)$"),
    active_only: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    """List all registered models with their training metadata (Rule 6)."""
    stmt = select(ModelRegistry).order_by(ModelRegistry.created_at.desc())
    if target_type:
        stmt = stmt.where(ModelRegistry.target_type == target_type)
    if active_only:
        stmt = stmt.where(ModelRegistry.is_active.is_(True))
    return db.scalars(stmt).all()


@router.get("/{model_id}", response_model=ModelRegistryResponse)
def get_registered_model(model_id: int, db: Session = Depends(get_db)):
    """Retrieve full audit metadata for a specific model (Rule 6: SHA-256, metrics, versions)."""
    model = db.get(ModelRegistry, model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with id {model_id} not found in registry",
        )
    return model


@router.post("", response_model=ModelRegistryResponse, status_code=status.HTTP_201_CREATED)
def register_model(
    model_in: ModelRegistryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_write_access),
):
    """Register a trained model artifact with complete provenance metadata (Rule 6 & Rule 7)."""
    record = ModelRegistry(
        model_name=model_in.model_name,
        target_type=model_in.target_type,
        version=model_in.version,
        file_path=model_in.file_path,
        dataset_version=model_in.dataset_version,
        data_sha256=model_in.data_sha256,
        feature_list=model_in.feature_list,
        metrics=model_in.metrics,
        library_versions=model_in.library_versions,
        is_active=model_in.is_active,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
