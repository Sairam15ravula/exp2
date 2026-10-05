"""Application configuration.

Loads settings from environment variables (or .env file).
Fails at startup if required variables are missing — rule 7.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "EV Battery Intelligence Platform"
    app_env: str = Field(default="development", pattern="^(development|staging|production)$")
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # --- Database ---
    database_url: str = Field(
        default="postgresql+psycopg2://ev_user:ev_pass@localhost:5432/ev_battery"
    )

    # --- Security ---
    secret_key: str = Field(min_length=32)
    access_token_expire_minutes: int = Field(default=30, ge=1, le=1440)

    # --- CORS ---
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # --- Model storage ---
    model_dir: str = "models"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors(cls, v: str | list[str]) -> list[str]:
        """Allow CORS_ORIGINS to be a JSON string or a list."""
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance.

    Raises:
        ValidationError: If required env vars are missing or invalid.
    """
    return Settings()
