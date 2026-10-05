"""Tests for SOH baselines (Rule 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_battery.soh.baselines import LinearInCycleBaseline, PredictTheMeanBaseline


class TestPredictTheMeanBaseline:
    def test_predict_mean_returns_exact_mean(self):
        X = pd.DataFrame({"cycle_number": [1, 2, 3, 4]})
        y = pd.Series([1.0, 0.9, 0.8, 0.7])

        model = PredictTheMeanBaseline().fit(X, y)
        assert model.mean_ == pytest.approx(0.85)

        preds = model.predict(pd.DataFrame({"cycle_number": [10, 20]}))
        assert len(preds) == 2
        assert np.allclose(preds, 0.85)

    def test_unfitted_raises_runtime_error(self):
        model = PredictTheMeanBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(pd.DataFrame({"cycle_number": [1]}))

    def test_empty_train_raises_error(self):
        model = PredictTheMeanBaseline()
        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(pd.DataFrame(), pd.Series([], dtype=float))


class TestLinearInCycleBaseline:
    def test_linear_baseline_fits_trend(self):
        # Perfect linear degradation: SOH = -0.01 * cycle + 1.0
        cycles = np.array([1, 10, 50, 100])
        y = 1.0 - 0.01 * cycles
        X = pd.DataFrame({"cycle_number": cycles, "other_feature": [0, 0, 0, 0]})

        model = LinearInCycleBaseline().fit(X, y)
        assert model.slope_ == pytest.approx(-0.01, abs=1e-5)
        assert model.intercept_ == pytest.approx(1.0, abs=1e-5)

        test_X = pd.DataFrame({"cycle_number": [20, 60]})
        preds = model.predict(test_X)
        assert preds[0] == pytest.approx(0.8, abs=1e-5)
        assert preds[1] == pytest.approx(0.4, abs=1e-5)

    def test_unfitted_raises_runtime_error(self):
        model = LinearInCycleBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(pd.DataFrame({"cycle_number": [1]}))

    def test_clipped_to_valid_range(self):
        X = pd.DataFrame({"cycle_number": [1, 2]})
        y = pd.Series([1.0, 0.0])  # Steep decline
        model = LinearInCycleBaseline().fit(X, y)

        # cycle 100 would be negative without clipping
        preds = model.predict(pd.DataFrame({"cycle_number": [100]}))
        assert preds[0] >= 0.0
