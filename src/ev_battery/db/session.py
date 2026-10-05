"""Database engine and session management.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ev_battery.config import get_settings
from ev_battery.db.base import Base
# Import models to ensure they are registered with Base.metadata
import ev_battery.db.models  # noqa: F401

settings = get_settings()


def create_db_engine(database_url: str | None = None):
    """Create SQLAlchemy engine with appropriate dialect arguments."""
    url = database_url or settings.database_url
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, connect_args=connect_args)


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a transactional DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(target_engine=None) -> None:
    """Create all tables in the target database (primarily for test environments)."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)


def drop_db(target_engine=None) -> None:
    """Drop all tables in the target database (for test isolation)."""
    eng = target_engine or engine
    Base.metadata.drop_all(bind=eng)
