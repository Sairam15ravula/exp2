"""Tests for SOHModel (XGBoost, monotonic constraints, Rule 6 metadata)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.soh.dataset import get_feature_matrix_and_target, prepare_soh_dataset
from ev_battery.soh.model import SOHModel, load_soh_model, save_soh_model


class TestSOHModel:
    def test_fit_and_predict(self):
        cycles = generate_synthetic_battery(battery_id="SYNTH_001", n_cycles=30, seed=42)
        df = prepare_soh_dataset(cycles)
        X, y = get_feature_matrix_and_target(df)

        model = SOHModel(n_estimators=20, max_depth=3, random_state=42)
        model.fit(X, y)
        assert model.is_fitted

        preds = model.predict(X)
        assert len(preds) == len(y)
        # SOH predictions must be in a plausible range
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.2)

    def test_unfitted_model_raises_runtime_error(self):
        model = SOHModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(pd.DataFrame())

    def test_monotone_constraint_experiment(self):
        """Monotonic constraint experiment: SOH must not increase as cycle count advances."""
        cycles = generate_synthetic_battery(battery_id="SYNTH_001", n_cycles=40, seed=42)
        df = prepare_soh_dataset(cycles)
        X, y = get_feature_matrix_and_target(df)

        # Train model with monotone constraint on cycle_number (-1 = decreasing)
        model = SOHModel(
            n_estimators=30,
            max_depth=3,
            monotone_constraints={"cycle_number": -1},
            random_state=42,
        )
        model.fit(X, y)

        # Create synthetic test grid where only cycle_number increases, other features fixed at mean
        mean_row = X.mean().to_dict()
        test_rows = []
        for c in range(1, 41):
            row = dict(mean_row)
            row["cycle_number"] = c
            test_rows.append(row)
        test_df = pd.DataFrame(test_rows)

        preds = model.predict(test_df)

        # Check monotonicity: preds[i+1] <= preds[i] + small epsilon for numerical stability
        diffs = np.diff(preds)
        assert np.all(diffs <= 1e-6), "Monotonicity violated: SOH increased as cycle count grew."

    def test_rule_6_reproducibility_and_persistence(self, tmp_path):
        """Rule 6: Store training metadata (dataset version, metrics, feature list, library versions, SHA256)."""
        cycles = generate_synthetic_battery(battery_id="SYNTH_001", n_cycles=25, seed=42)
        df = prepare_soh_dataset(cycles)
        X, y = get_feature_matrix_and_target(df)

        model = SOHModel(n_estimators=15, max_depth=3, random_state=42)
        model.fit(X, y, dataset_version="synth-v1")

        # Verify metadata dictionary in model
        meta = model.training_metadata
        assert meta["dataset_version"] == "synth-v1"
        assert len(meta["data_sha256"]) == 64  # Valid SHA-256 hex string
        assert meta["random_seed"] == 42
        assert "xgboost" in meta["library_versions"]
        assert "scikit-learn" in meta["library_versions"]
        assert "feature_list" in meta

        # Test save and load
        save_dir = tmp_path / "saved_soh_model"
        metrics = {"lobo_rmse_mean": 0.012}
        save_soh_model(model, save_dir, metrics=metrics)

        loaded_model, loaded_meta = load_soh_model(save_dir)
        assert loaded_model.is_fitted
        assert loaded_meta["metrics"]["lobo_rmse_mean"] == 0.012

        # Predictions from reloaded model must be identical
        orig_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        assert np.allclose(orig_preds, loaded_preds)
