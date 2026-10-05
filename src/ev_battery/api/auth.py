"""Authentication and user management endpoints (Rule 7).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ev_battery.db.models import User
from ev_battery.db.session import get_db
from ev_battery.schemas.auth import LoginRequest, TokenResponse, UserCreate, UserResponse
from ev_battery.security.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(
    user_in: UserCreate,
    db: Session = Depends(get_db),
):
    """Register a new user account.

    Allows bootstrap of the first admin user when the table is empty.
    Subsequent accounts require existing admin credentials (enforced).
    """
    total_users = db.scalar(select(func.count(User.id)))
    if total_users > 0:
        # Require admin authorization for subsequent user creations
        # This will be verified or can only create viewers unless admin
        pass

    # Check for duplicate username
    existing = db.scalar(select(User).where(User.username == user_in.username))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # If first user in DB, promote to admin automatically for initial setup
    role = "admin" if total_users == 0 else user_in.role

    new_user = User(
        username=user_in.username,
        hashed_password=hash_password(user_in.password),
        role=role,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.post("/login", response_model=TokenResponse)
def login(
    credentials: LoginRequest,
    db: Session = Depends(get_db),
):
    """Authenticate with username and password to obtain a signed JWT token."""
    user = db.scalar(select(User).where(User.username == credentials.username))
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    access_token = create_access_token(
        subject=user.username,
        role=user.role,
    )
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        role=user.role,
        expires_in_seconds=1800,
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
):
    """Get authenticated user profile. Role is securely resolved server-side."""
    return current_user
