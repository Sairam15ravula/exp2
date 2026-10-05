"""Tests for Alembic database migration upgrade and downgrade cycle.
"""

from __future__ import annotations

import os
from pathlib import Path
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_alembic_upgrade_and_downgrade_cycle(tmp_path):
    """Test full Alembic upgrade to head and downgrade to base."""
    db_file = tmp_path / "test_migration.db"
    db_url = f"sqlite:///{db_file}"

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)

    # 1. Run upgrade to head
    command.upgrade(cfg, "head")

    engine = create_engine(db_url)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())

    expected_tables = {
        "battery_packs",
        "cells",
        "telemetry",
        "cycles",
        "state_estimates",
        "predictions",
        "anomaly_events",
        "model_registry",
        "alerts",
        "users",
        "alembic_version",
    }
    assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"

    # 2. Run downgrade to base
    command.downgrade(cfg, "base")

    inspector_after = inspect(engine)
    tables_after = set(inspector_after.get_table_names())
    # Should only have alembic_version or empty
    assert len(tables_after.difference({"alembic_version"})) == 0

    engine.dispose()
