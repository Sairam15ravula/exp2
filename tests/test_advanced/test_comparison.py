"""Tests for honest benchmarking comparing LSTM vs XGBoost (Rule 3)."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.advanced.comparison import (
    compare_lstm_vs_xgboost,
    count_monotonic_violations,
    format_advanced_comparison_table,
)
from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.soh.dataset import prepare_soh_dataset


class TestModelComparison:
    def test_count_monotonic_violations(self):
        # Monotonic decay -> 0% violations
        decaying = np.array([1.0, 0.95, 0.90, 0.85])
        assert count_monotonic_violations(decaying) == pytest.approx(0.0)

        # Monotonic rise -> 100% violations
        rising = np.array([0.80, 0.85, 0.90])
        assert count_monotonic_violations(rising) == pytest.approx(1.0)

        # 1 violation out of 3 transitions -> 33.3%
        mixed = np.array([1.0, 0.90, 0.95, 0.85])
        assert count_monotonic_violations(mixed) == pytest.approx(1.0 / 3.0)

    def test_compare_lstm_vs_xgboost_execution(self):
        b1 = generate_synthetic_battery("SYNTH_001", n_cycles=20, degradation_rate=0.0010, seed=1)
        b2 = generate_synthetic_battery("SYNTH_002", n_cycles=20, degradation_rate=0.0020, seed=2)
        b3 = generate_synthetic_battery("SYNTH_003", n_cycles=20, degradation_rate=0.0030, seed=3)
        df = prepare_soh_dataset(b1 + b2 + b3)

        res = compare_lstm_vs_xgboost(df, window_size=3, epochs=5, random_seed=42)

        assert res["n_folds"] == 3
        summary = res["summary"]

        for model_key in ["xgboost", "lstm_standard", "lstm_pinn"]:
            assert model_key in summary
            assert "rmse" in summary[model_key]
            assert "mae" in summary[model_key]
            assert "r2" in summary[model_key]
            assert "mono_violation_rate" in summary[model_key]

        table_str = format_advanced_comparison_table(res)
        assert "| Architecture | RMSE (mean ± std) |" in table_str
        assert "XGBoost (Phase 4)" in table_str
        assert "Standard LSTM" in table_str
        assert "Physics-Informed LSTM (PINN)" in table_str
