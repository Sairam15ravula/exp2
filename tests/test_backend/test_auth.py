"""Tests for authentication, JWT security, and Role-Based Access Control (Rule 7).
"""

from __future__ import annotations

from datetime import timedelta
import pytest
from fastapi import HTTPException

from ev_battery.security.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_password_hashing_and_verification():
    """Verify bcrypt password hashing and constant-time check."""
    raw = "SuperSecret123!"
    hashed = hash_password(raw)

    assert hashed != raw
    assert hashed.startswith("$2b$")
    assert verify_password(raw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False


def test_jwt_token_creation_and_validation():
    """Verify token encoding and claims decoding."""
    token = create_access_token(
        subject="engineer_jane",
        role="engineer",
        expires_delta=timedelta(minutes=15),
    )
    payload = decode_access_token(token)

    assert payload["sub"] == "engineer_jane"
    assert payload["role"] == "engineer"
    assert "exp" in payload


def test_expired_jwt_token_raises_401():
    """Verify expired tokens are rejected."""
    token = create_access_token(
        subject="expired_user",
        role="viewer",
        expires_delta=timedelta(seconds=-10),  # expired in past
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_access_token(token)
    assert exc_info.value.status_code == 401


def test_login_flow(client, auth_users):
    """Test successful and invalid login attempts."""
    # 1. Successful login
    res = client.post(
        "/api/v1/auth/login",
        json={"username": "engineer_test", "password": "engineer_pass_123"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == "engineer"

    # 2. Invalid password
    res_bad = client.post(
        "/api/v1/auth/login",
        json={"username": "engineer_test", "password": "wrong_password"},
    )
    assert res_bad.status_code == 401

    # 3. Nonexistent user
    res_no = client.post(
        "/api/v1/auth/login",
        json={"username": "does_not_exist", "password": "any"},
    )
    assert res_no.status_code == 401


def test_get_current_user_profile(client, engineer_headers):
    """Verify /me profile resolves role from server database token (Rule 7)."""
    res = client.get("/api/v1/auth/me", headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "engineer_test"
    assert data["role"] == "engineer"


def test_inactive_user_cannot_access(client, auth_users):
    """Verify inactive account is blocked with 403."""
    token = create_access_token(subject="inactive_test", role="viewer")
    res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 403
    assert "inactive" in res.json()["detail"].lower()


def test_role_enforcement_on_write_routes(client, viewer_headers, engineer_headers):
    """Rule 7: Write route rejects viewer (403) and accepts engineer."""
    pack_payload = {
        "id": "PACK_TEST_WRITE",
        "name": "Write Test Pack",
        "series_cells": 4,
        "parallel_strings": 2,
        "nominal_cell_voltage_v": 3.6,
        "nominal_cell_capacity_ah": 2.0,
        "auto_generate_cells": False,
    }

    # 1. Unauthenticated request -> 401
    res_no_auth = client.post("/api/v1/packs", json=pack_payload)
    assert res_no_auth.status_code == 401

    # 2. Viewer role request -> 403 Forbidden (Roles NEVER accepted from client)
    res_viewer = client.post("/api/v1/packs", json=pack_payload, headers=viewer_headers)
    assert res_viewer.status_code == 403
    assert "Forbidden" in res_viewer.json()["detail"]

    # 3. Engineer role request -> 201 Created
    res_eng = client.post("/api/v1/packs", json=pack_payload, headers=engineer_headers)
    assert res_eng.status_code == 201
    assert res_eng.json()["id"] == "PACK_TEST_WRITE"


def test_cors_allow_list(client):
    """Rule 7: CORS allow-list: verify allowed origins vs rejected origins."""
    # Allowed origin: http://localhost:5173
    res_allowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert res_allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    # Disallowed origin: http://malicious-site.example.com
    res_disallowed = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://malicious-site.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Access-Control-Allow-Origin header MUST NOT be http://malicious-site.example.com
    assert res_disallowed.headers.get("access-control-allow-origin") != "http://malicious-site.example.com"
