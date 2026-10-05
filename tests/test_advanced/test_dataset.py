"""Tests for sequence windowing and cross-temperature dataset preparation."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.advanced.dataset import (
    SEQUENCE_FEATURE_COLUMNS,
    create_sequence_windows,
    cross_temperature_split,
)
from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.soh.dataset import prepare_soh_dataset


class TestSequenceDataset:
    @pytest.fixture
    def multi_battery_df(self):
        b1 = generate_synthetic_battery("SYNTH_001", n_cycles=25, seed=1)
        b2 = generate_synthetic_battery("SYNTH_002", n_cycles=25, seed=2)
        return prepare_soh_dataset(b1 + b2)

    def test_window_shapes(self, multi_battery_df):
        window_size = 5
        X, y = create_sequence_windows(multi_battery_df, window_size=window_size)

        n_feats = len(SEQUENCE_FEATURE_COLUMNS)
        # For 2 batteries of 25 cycles each: (25 - 5 + 1) * 2 = 42 windows
        expected_windows = (25 - window_size + 1) * 2

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (expected_windows, window_size, n_feats)
        assert y.shape == (expected_windows,)

    def test_rule_2_target_leakage_prevented(self, multi_battery_df):
        """Rule 2: Sequence features must not include target columns."""
        with pytest.raises(ValueError, match="Target leakage detected"):
            create_sequence_windows(
                multi_battery_df,
                feature_columns=["cycle_number", "capacity"],
            )

        with pytest.raises(ValueError, match="Target leakage detected"):
            create_sequence_windows(
                multi_battery_df,
                feature_columns=["cycle_number", "soh"],
            )

    def test_invalid_window_size_raises(self, multi_battery_df):
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            create_sequence_windows(multi_battery_df, window_size=0)

    def test_cross_temperature_split_disjoint(self):
        """Rule 1: Cross-temperature split must have disjoint battery sets."""
        b1 = generate_synthetic_battery("SYNTH_ROOM_1", n_cycles=20, seed=1)
        b2 = generate_synthetic_battery("SYNTH_HOT_1", n_cycles=20, seed=2)

        # Elevate temperature for second battery
        for c in b2:
            c.temperature = c.temperature + 12.0

        df = prepare_soh_dataset(b1 + b2)
        train_df, test_df = cross_temperature_split(df, split_temp_threshold=38.0)

        train_bats = set(train_df["battery_id"].unique())
        test_bats = set(test_df["battery_id"].unique())

        assert len(train_bats) > 0
        assert len(test_bats) > 0
        assert train_bats.isdisjoint(test_bats)
