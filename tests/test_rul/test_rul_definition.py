"""Tests enforcing Rule 4 (RUL Definition) and Rule 2 (No Target Leakage in RUL)."""

from __future__ import annotations

import pytest

from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.rul.dataset import (
    EOL_SOH_THRESHOLD,
    FEATURE_COLUMNS,
    find_eol_cycle,
    get_rul_feature_matrix_and_target,
    prepare_rul_dataset,
    validate_no_rul_target_leakage,
)
from ev_battery.soh.dataset import prepare_soh_dataset


class TestRULDefinitionRule4:
    """Rule 4: RUL is defined as cycles until SOH <= 80% (end of life).

    NOT cycles until the experiment ended.
    """

    def test_eol_soh_threshold_constant_is_80_percent(self):
        """EOL threshold must strictly be 0.80 (80%)."""
        assert EOL_SOH_THRESHOLD == pytest.approx(0.80)

    def test_rul_is_cycles_until_eol_not_experiment_end(self):
        """Rule 4: Ground truth RUL must equal (k_eol - k), NOT (experiment_end - k)."""
        # Generate 140 cycles with degradation rate 0.0025
        # 80% SOH is reached around cycle 90, while experiment ran until cycle 140
        cycles = generate_synthetic_battery(
            battery_id="SYNTH_TEST",
            n_cycles=140,
            degradation_rate=0.0025,
            seed=42,
        )
        experiment_duration_cycles = len(cycles)
        assert experiment_duration_cycles == 140

        df_soh = prepare_soh_dataset(cycles)
        eol_cycle = find_eol_cycle(df_soh, eol_threshold=0.80)
        assert eol_cycle is not None
        assert eol_cycle < experiment_duration_cycles, "EOL cycle must be strictly before experiment end"

        # Prepare RUL dataset
        df_rul = prepare_rul_dataset(cycles, eol_threshold=0.80)

        # 1. Total labeled rows must equal eol_cycle (cycles after EOL are excluded)
        assert len(df_rul) == eol_cycle

        # 2. Cycle 1 RUL must be (eol_cycle - 1)
        row_1 = df_rul[df_rul["cycle_number"] == 1].iloc[0]
        expected_rul_cycle_1 = float(eol_cycle - 1)
        wrong_experiment_rul_cycle_1 = float(experiment_duration_cycles - 1)

        assert row_1["rul"] == pytest.approx(expected_rul_cycle_1)
        assert row_1["rul"] != pytest.approx(wrong_experiment_rul_cycle_1)

        # 3. EOL cycle itself must have RUL == 0
        row_eol = df_rul[df_rul["cycle_number"] == eol_cycle].iloc[0]
        assert row_eol["rul"] == pytest.approx(0.0)

        # 4. Monotonic decrease: RUL(k+1) == RUL(k) - 1
        ruls = df_rul["rul"].to_numpy()
        diffs = ruls[1:] - ruls[:-1]
        assert (diffs == -1.0).all()

    def test_battery_never_reaching_eol_is_skipped(self):
        """Batteries that never drop below 80% SOH are skipped to avoid censored RUL."""
        # 10 cycles with very slow degradation will stay well above 80%
        cycles = generate_synthetic_battery(
            battery_id="SYNTH_HEALTHY",
            n_cycles=10,
            degradation_rate=0.0001,
            seed=42,
        )
        df = prepare_rul_dataset(cycles, eol_threshold=0.80)
        assert df.empty


class TestRULTargetLeakageRule2:
    """Rule 2: Features must not leak capacity, SOH, or RUL targets."""

    def test_feature_columns_do_not_contain_target(self):
        for col in FEATURE_COLUMNS:
            assert "capacity" not in col.lower()
            assert "soh" not in col.lower()
            assert "rul" not in col.lower()

    def test_validate_no_rul_target_leakage_catches_leakage(self):
        with pytest.raises(ValueError, match="Target leakage detected"):
            validate_no_rul_target_leakage(["cycle_number", "rul", "voltage_mean"])

        with pytest.raises(ValueError, match="Target leakage detected"):
            validate_no_rul_target_leakage(["cycle_number", "capacity"])

        with pytest.raises(ValueError, match="Target leakage detected"):
            validate_no_rul_target_leakage(["cycle_number", "eol_cycle"])

    def test_get_rul_feature_matrix_excludes_target(self):
        cycles = generate_synthetic_battery("SYNTH_001", n_cycles=120, degradation_rate=0.0025, seed=1)
        df_rul = prepare_rul_dataset(cycles)

        X, y = get_rul_feature_matrix_and_target(df_rul)
        assert "rul" not in X.columns
        assert "capacity" not in X.columns
        assert "soh" not in X.columns
        assert "eol_cycle" not in X.columns
        assert len(X) == len(y)
