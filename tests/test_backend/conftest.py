"""Test fixtures for backend test suite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ev_battery.api.app import app
from ev_battery.db.base import Base
from ev_battery.db.models import BatteryPack, Cell, User
from ev_battery.db.session import get_db
from ev_battery.security.auth import create_access_token, hash_password


@pytest.fixture(scope="session")
def test_engine():
    """Create an isolated in-memory SQLite engine for backend tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_db(test_engine):
    """Provide a transactional database session for each test function."""
    connection = test_engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestingSessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(test_db):
    """FastAPI TestClient with overridden get_db dependency."""
    def _override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def auth_users(test_db):
    """Seed test users across distinct roles (admin, engineer, viewer)."""
    admin_user = User(
        username="admin_test",
        hashed_password=hash_password("admin_pass_123"),
        role="admin",
        is_active=True,
    )
    engineer_user = User(
        username="engineer_test",
        hashed_password=hash_password("engineer_pass_123"),
        role="engineer",
        is_active=True,
    )
    viewer_user = User(
        username="viewer_test",
        hashed_password=hash_password("viewer_pass_123"),
        role="viewer",
        is_active=True,
    )
    inactive_user = User(
        username="inactive_test",
        hashed_password=hash_password("inactive_pass_123"),
        role="viewer",
        is_active=False,
    )
    test_db.add_all([admin_user, engineer_user, viewer_user, inactive_user])
    test_db.commit()

    return {
        "admin": admin_user,
        "engineer": engineer_user,
        "viewer": viewer_user,
        "inactive": inactive_user,
    }


@pytest.fixture(scope="function")
def admin_headers(auth_users):
    token = create_access_token(subject="admin_test", role="admin")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def engineer_headers(auth_users):
    token = create_access_token(subject="engineer_test", role="engineer")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def viewer_headers(auth_users):
    token = create_access_token(subject="viewer_test", role="viewer")
    return {"Authorization": f"Bearer {token}"}
