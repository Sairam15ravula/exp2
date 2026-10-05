"""Tests for RUL quantile model, prediction intervals, and Rule 6 metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_battery.data.synthetic import generate_synthetic_battery
from ev_battery.rul.dataset import (
    get_rul_feature_matrix_and_target,
    prepare_rul_dataset,
)
from ev_battery.rul.model import RULModel, load_rul_model, save_rul_model


class TestRULModel:
    @pytest.fixture
    def rul_data(self):
        cycles = generate_synthetic_battery("SYNTH_001", n_cycles=120, degradation_rate=0.0025, seed=42)
        df_rul = prepare_rul_dataset(cycles)
        X, y = get_rul_feature_matrix_and_target(df_rul)
        return X, y

    def test_fit_and_point_predict(self, rul_data):
        X, y = rul_data
        model = RULModel(n_estimators=15, max_depth=2, random_state=42)
        model.fit(X, y)
        assert model.is_fitted

        preds = model.predict(X)
        assert len(preds) == len(y)
        # RUL predictions must be non-negative
        assert (preds >= 0.0).all()

    def test_predict_interval_non_crossing(self, rul_data):
        """Prediction interval must satisfy lower <= median <= upper for all samples."""
        X, y = rul_data
        model = RULModel(n_estimators=15, max_depth=2, random_state=42)
        model.fit(X, y)

        lower, median, upper = model.predict_interval(X)

        assert len(lower) == len(y)
        assert len(median) == len(y)
        assert len(upper) == len(y)

        # Enforce ordering
        assert (lower <= median).all(), "Lower quantile exceeds median"
        assert (median <= upper).all(), "Median exceeds upper quantile"
        assert (lower >= 0.0).all()

    def test_unfitted_model_raises_error(self):
        model = RULModel()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(pd.DataFrame())

        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict_interval(pd.DataFrame())

    def test_rule_6_reproducibility_and_persistence(self, rul_data, tmp_path):
        """Rule 6: Save and reload RUL model and metadata sidecar."""
        X, y = rul_data
        model = RULModel(n_estimators=15, max_depth=2, random_state=42)
        model.fit(X, y, dataset_version="rul-v1")

        meta = model.training_metadata
        assert meta["dataset_version"] == "rul-v1"
        assert len(meta["data_sha256"]) == 64
        assert meta["random_seed"] == 42
        assert "scikit-learn" in meta["library_versions"]

        save_dir = tmp_path / "rul_saved"
        metrics = {"lobo_rmse_mean": 5.4}
        save_rul_model(model, save_dir, metrics=metrics)

        loaded_model, loaded_meta = load_rul_model(save_dir)
        assert loaded_model.is_fitted
        assert loaded_meta["metrics"]["lobo_rmse_mean"] == 5.4

        orig_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        assert np.allclose(orig_preds, loaded_preds)

        orig_l, orig_m, orig_u = model.predict_interval(X)
        load_l, load_m, load_u = loaded_model.predict_interval(X)
        assert np.allclose(orig_l, load_l)
        assert np.allclose(orig_u, load_u)
