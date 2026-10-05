"""Tests for LOBO evaluation and baseline comparisons (Rule 1 & Rule 3)."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.soh.dataset import prepare_soh_dataset
from ev_battery.soh.evaluation import (
    evaluate_lobo,
    evaluate_predictions,
    format_evaluation_table,
)


class TestEvaluatePredictions:
    def test_perfect_predictions(self):
        y_true = np.array([1.0, 0.9, 0.8])
        y_pred = np.array([1.0, 0.9, 0.8])
        metrics = evaluate_predictions(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["max_error"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)

    def test_imperfect_predictions(self):
        y_true = np.array([1.0, 0.8])
        y_pred = np.array([0.9, 0.7])  # constant offset of -0.1
        metrics = evaluate_predictions(y_true, y_pred)
        assert metrics["rmse"] == pytest.approx(0.1)
        assert metrics["mae"] == pytest.approx(0.1)
        assert metrics["max_error"] == pytest.approx(0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Cannot evaluate empty"):
            evaluate_predictions([], [])


class TestLOBOEvaluationRule3:
    """Rule 3: ALWAYS BEAT A BASELINE.

    Report every model against (a) predict-the-mean and (b) a simple linear-in-cycle model.
    Report mean ± std across ALL folds.
    """

    @pytest.fixture
    def multi_battery_dataset(self):
        b1 = generate_synthetic_battery("SYNTH_001", n_cycles=50, degradation_rate=0.0010, seed=10)
        b2 = generate_synthetic_battery("SYNTH_002", n_cycles=50, degradation_rate=0.0020, seed=20)
        b3 = generate_synthetic_battery("SYNTH_003", n_cycles=50, degradation_rate=0.0030, seed=30)
        return prepare_soh_dataset(b1 + b2 + b3)

    def test_evaluate_lobo_structure_and_folds(self, multi_battery_dataset):
        results = evaluate_lobo(multi_battery_dataset, n_estimators=25, max_depth=3)

        assert results["n_folds"] == 3
        assert len(results["folds"]) == 3

        summary = results["summary"]
        for key in ["mean_baseline", "linear_baseline", "xgboost"]:
            assert key in summary
            assert "mean" in summary[key]["rmse"]
            assert "std" in summary[key]["rmse"]
            assert "mean" in summary[key]["mae"]
            assert "std" in summary[key]["mae"]
            assert "mean" in summary[key]["r2"]
            assert "std" in summary[key]["r2"]

    def test_xgboost_beats_baselines_across_folds(self, multi_battery_dataset):
        """Rule 3 enforcement: XGBoost must achieve lower RMSE than both baselines across all folds."""
        results = evaluate_lobo(
            multi_battery_dataset,
            n_estimators=40,
            max_depth=3,
            learning_rate=0.08,
            random_state=42,
        )

        xgb_rmse = results["summary"]["xgboost"]["rmse"]["mean"]
        mean_rmse = results["summary"]["mean_baseline"]["rmse"]["mean"]
        linear_rmse = results["summary"]["linear_baseline"]["rmse"]["mean"]

        # Predict-the-mean has poor performance on fading trajectories
        assert xgb_rmse < mean_rmse, f"XGBoost ({xgb_rmse}) failed to beat mean baseline ({mean_rmse})"
        assert results["beats_mean_baseline"]

        # Non-linear features (voltage drop rate, IR proxy, temp rise) allow XGBoost to beat linear model
        assert xgb_rmse < linear_rmse, f"XGBoost ({xgb_rmse}) failed to beat linear baseline ({linear_rmse})"
        assert results["beats_linear_baseline"]
        assert results["beats_all_baselines"]

    def test_format_evaluation_table(self, multi_battery_dataset):
        results = evaluate_lobo(multi_battery_dataset, n_estimators=10, max_depth=2)
        table = format_evaluation_table(results)

        assert "| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |" in table
        assert "Predict-the-mean" in table
        assert "Linear-in-cycle" in table
        assert "XGBoost (SOH)" in table
