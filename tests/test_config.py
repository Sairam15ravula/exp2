"""Tests for configuration module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ev_battery.config import Settings


class TestSettingsDefaults:
    """Settings should load with valid defaults when env vars are set."""

    def test_loads_with_required_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://u:p@localhost:5432/db")
        s = Settings(_env_file=None)
        assert s.app_env == "development"
        assert s.log_level == "INFO"
        assert s.access_token_expire_minutes == 30
        assert s.cors_origins == ["http://localhost:5173"]
        assert s.model_dir == "models"

    def test_missing_secret_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("SECRET_KEY", raising=False)
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_short_secret_key_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "too-short")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_invalid_app_env_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("APP_ENV", "invalid")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_invalid_log_level_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(ValidationError):
            Settings(_env_file=None)

    def test_cors_origins_from_json_string(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000","http://localhost:5173"]')
        s = Settings(_env_file=None)
        assert s.cors_origins == ["http://localhost:3000", "http://localhost:5173"]

    def test_is_production_property(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("APP_ENV", "production")
        s = Settings(_env_file=None)
        assert s.is_production is True

    def test_is_production_false_for_dev(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        monkeypatch.setenv("APP_ENV", "development")
        s = Settings(_env_file=None)
        assert s.is_production is False


class TestGetSettings:
    """get_settings() should return a cached instance."""

    def test_returns_same_instance(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SECRET_KEY", "a" * 32)
        from ev_battery.config import get_settings
        # Clear cache to avoid interference from other tests
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
