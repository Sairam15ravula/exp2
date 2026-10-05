"""Tests for RUL LOBO evaluation and baseline beating (Rule 1 & Rule 3)."""

from __future__ import annotations

import numpy as np
import pytest

from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.rul.dataset import prepare_rul_dataset
from ev_battery.rul.evaluation import (
    evaluate_rul_lobo,
    evaluate_rul_predictions,
    format_rul_evaluation_table,
)


class TestEvaluateRULPredictions:
    def test_perfect_predictions_with_intervals(self):
        y_true = np.array([50.0, 30.0, 10.0])
        y_pred = np.array([50.0, 30.0, 10.0])
        y_lower = np.array([45.0, 25.0, 5.0])
        y_upper = np.array([55.0, 35.0, 15.0])

        metrics = evaluate_rul_predictions(y_true, y_pred, y_lower, y_upper)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["picp"] == pytest.approx(1.0)
        assert metrics["mpiw"] == pytest.approx(10.0)

    def test_empty_predictions_raises_error(self):
        with pytest.raises(ValueError, match="Cannot evaluate empty"):
            evaluate_rul_predictions([], [])


class TestRULLOBOEvaluationRule3:
    """Rule 3: ALWAYS BEAT A BASELINE.

    Report RUL model against predict-the-mean and linear-in-cycle baseline.
    Report mean ± std across ALL folds.
    """

    @pytest.fixture
    def multi_battery_rul_data(self):
        b1 = generate_synthetic_battery("SYNTH_001", n_cycles=130, degradation_rate=0.0020, seed=1)
        b2 = generate_synthetic_battery("SYNTH_002", n_cycles=130, degradation_rate=0.0025, seed=2)
        b3 = generate_synthetic_battery("SYNTH_003", n_cycles=130, degradation_rate=0.0030, seed=3)
        return prepare_rul_dataset(b1 + b2 + b3)

    def test_evaluate_rul_lobo_structure(self, multi_battery_rul_data):
        results = evaluate_rul_lobo(multi_battery_rul_data, n_estimators=20, max_depth=2)

        assert results["n_folds"] == 3
        assert len(results["folds"]) == 3

        summary = results["summary"]
        for key in ["mean_baseline", "linear_baseline", "rul_model"]:
            assert key in summary
            assert "mean" in summary[key]["rmse"]
            assert "std" in summary[key]["rmse"]
            assert "mean" in summary[key]["mae"]
            assert "std" in summary[key]["mae"]

        # RUL model must include uncertainty interval metrics
        assert "picp" in summary["rul_model"]
        assert "mpiw" in summary["rul_model"]

    def test_rul_model_beats_both_baselines_across_folds(self, multi_battery_rul_data):
        """Rule 3 enforcement: RULModel achieves lower RMSE than both baselines across all folds."""
        results = evaluate_rul_lobo(
            multi_battery_rul_data,
            n_estimators=40,
            max_depth=3,
            learning_rate=0.08,
            random_state=42,
        )

        rul_rmse = results["summary"]["rul_model"]["rmse"]["mean"]
        mean_rmse = results["summary"]["mean_baseline"]["rmse"]["mean"]
        linear_rmse = results["summary"]["linear_baseline"]["rmse"]["mean"]

        assert rul_rmse < mean_rmse, f"RULModel ({rul_rmse}) failed to beat mean baseline ({mean_rmse})"
        assert results["beats_mean_baseline"]

        assert rul_rmse < linear_rmse, f"RULModel ({rul_rmse}) failed to beat linear baseline ({linear_rmse})"
        assert results["beats_linear_baseline"]
        assert results["beats_all_baselines"]

    def test_format_rul_evaluation_table(self, multi_battery_rul_data):
        results = evaluate_rul_lobo(multi_battery_rul_data, n_estimators=10, max_depth=2)
        table = format_rul_evaluation_table(results)

        assert "| Model | RMSE (mean ± std) | MAE (mean ± std) |" in table
        assert "Predict-the-mean" in table
        assert "Linear-in-cycle" in table
        assert "RUL Quantile Model" in table
