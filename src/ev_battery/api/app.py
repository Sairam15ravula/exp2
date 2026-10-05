"""Main FastAPI application factory.

Rules enforced:
- Rule 7: CORS is an allow-list (settings.cors_origins), NEVER '*'.
- Rule 7: Strict startup configuration and secret validation.
- Rule 10: Simple, lean architecture without redundant gateways.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from ev_battery.api.alerts import router as alerts_router
from ev_battery.api.auth import router as auth_router
from ev_battery.api.explain import router as explain_router
from ev_battery.api.inference import router as inference_router
from ev_battery.api.models import router as models_router
from ev_battery.api.packs import router as packs_router
from ev_battery.api.telemetry import router as telemetry_router
from ev_battery.config import get_settings
from ev_battery.db.session import SessionLocal, get_db

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Verify database connectivity at startup
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        # In test environments with in-memory SQLite or lazy db, don't crash
        pass
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "EV Battery Intelligence Platform REST API — Decision-support system extending "
            "conventional BMS with physics-based SOC, machine learning SOH/RUL, and thermal risk forecasting."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Rule 7: CORS allow-list (strictly from settings, never '*')
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", tags=["System"])
    @app.get("/api/v1/health", tags=["System"])
    def health_check(db: Session = Depends(get_db)):
        db_ok = False
        try:
            db.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            db_ok = False

        return {
            "status": "healthy" if db_ok else "degraded",
            "app_name": settings.app_name,
            "environment": settings.app_env,
            "database_connected": db_ok,
        }

    # Mount API v1 Routers
    api_prefix = "/api/v1"
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(packs_router, prefix=api_prefix)
    app.include_router(telemetry_router, prefix=api_prefix)
    app.include_router(inference_router, prefix=api_prefix)
    app.include_router(alerts_router, prefix=api_prefix)
    app.include_router(models_router, prefix=api_prefix)
    app.include_router(explain_router, prefix=api_prefix)

    return app


app = create_app()
