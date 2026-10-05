"""Authentication and User schemas.
"""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    """User credentials for token issuance."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT bearer access token response."""
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_seconds: int


class UserCreate(BaseModel):
    """User registration schema."""
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=6)
    role: str = Field(default="viewer", pattern="^(admin|engineer|viewer)$")


class UserResponse(BaseModel):
    """User response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime
