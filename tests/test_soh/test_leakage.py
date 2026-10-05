"""Tests enforcing Rule 1 (No Data Leakage) and Rule 2 (No Target Leakage)."""

from __future__ import annotations

import pytest

from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.soh.dataset import (
    FEATURE_COLUMNS,
    get_feature_matrix_and_target,
    leave_one_battery_out_splits,
    prepare_soh_dataset,
    validate_no_target_leakage,
)


class TestTargetLeakageRule2:
    """Rule 2: SOH = capacity / initial_capacity.

    Capacity and initial_capacity must NOT be input features when predicting SOH.
    Write a test that fails if a feature is a direct function of the target.
    """

    def test_feature_columns_do_not_contain_capacity(self):
        """FEATURE_COLUMNS must not include capacity or initial_capacity."""
        for col in FEATURE_COLUMNS:
            assert "capacity" not in col.lower()
            assert "initial_capacity" not in col.lower()
            assert "soh" not in col.lower()
            assert "target" not in col.lower()

    def test_validate_no_target_leakage_catches_leakage(self):
        """Validator must raise ValueError if capacity or target-derived columns are present."""
        leaky_features_1 = ["cycle_number", "capacity", "voltage_mean"]
        with pytest.raises(ValueError, match="Target leakage detected"):
            validate_no_target_leakage(leaky_features_1)

        leaky_features_2 = ["cycle_number", "initial_capacity"]
        with pytest.raises(ValueError, match="Target leakage detected"):
            validate_no_target_leakage(leaky_features_2)

        leaky_features_3 = ["cycle_number", "soh_ratio"]
        with pytest.raises(ValueError, match="Target leakage detected"):
            validate_no_target_leakage(leaky_features_3)

    def test_validate_no_target_leakage_passes_clean_features(self):
        """Clean features must pass without raising."""
        validate_no_target_leakage(FEATURE_COLUMNS)

    def test_get_feature_matrix_excludes_target_and_capacity(self):
        """Feature matrix X extracted from dataset must strictly exclude target columns."""
        cycles = generate_synthetic_battery(battery_id="SYNTH_001", n_cycles=10)
        df = prepare_soh_dataset(cycles)

        X, y = get_feature_matrix_and_target(df)

        assert "capacity" not in X.columns
        assert "initial_capacity" not in X.columns
        assert "soh" not in X.columns
        assert len(X) == 10
        assert len(y) == 10


class TestDataLeakageRule1:
    """Rule 1: Split train/test BY BATTERY (leave-one-battery-out).

    Never use random row splits on cycle data — rows from the same battery are correlated.
    """

    def test_lobo_splits_have_zero_battery_overlap(self):
        """Train and test splits in LOBO must have completely disjoint battery IDs."""
        c1 = generate_synthetic_battery(battery_id="SYNTH_001", n_cycles=15, seed=1)
        c2 = generate_synthetic_battery(battery_id="SYNTH_002", n_cycles=15, seed=2)
        c3 = generate_synthetic_battery(battery_id="SYNTH_003", n_cycles=15, seed=3)

        df = prepare_soh_dataset(c1 + c2 + c3)

        folds = list(leave_one_battery_out_splits(df))
        assert len(folds) == 3

        for held_out_battery, train_df, test_df in folds:
            train_bats = set(train_df["battery_id"].unique())
            test_bats = set(test_df["battery_id"].unique())

            # 1. Held out battery is the only battery in test set
            assert test_bats == {held_out_battery}
            # 2. Held out battery is NOT in train set
            assert held_out_battery not in train_bats
            # 3. Disjoint check
            assert train_bats.isdisjoint(test_bats)
            # 4. Total rows match
            assert len(train_df) + len(test_df) == len(df)
