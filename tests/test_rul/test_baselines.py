"""Tests for RUL baselines (Rule 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ev_battery.rul.baselines import (
    LinearExtrapolationRULBaseline,
    PredictTheMeanRULBaseline,
)


class TestPredictTheMeanRULBaseline:
    def test_predict_mean_returns_exact_mean(self):
        X = pd.DataFrame({"cycle_number": [1, 2, 3]})
        y = pd.Series([100.0, 50.0, 0.0])

        model = PredictTheMeanRULBaseline().fit(X, y)
        assert model.mean_rul_ == pytest.approx(50.0)

        preds = model.predict(pd.DataFrame({"cycle_number": [10, 20]}))
        assert len(preds) == 2
        assert np.allclose(preds, 50.0)

    def test_unfitted_raises_error(self):
        model = PredictTheMeanRULBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(pd.DataFrame({"cycle_number": [1]}))

    def test_empty_train_raises_error(self):
        model = PredictTheMeanRULBaseline()
        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(pd.DataFrame(), pd.Series([], dtype=float))


class TestLinearExtrapolationRULBaseline:
    def test_linear_baseline_fits_slope(self):
        # Perfect countdown: RUL = 100 - cycle
        cycles = np.array([1, 20, 50, 80])
        y = 100.0 - cycles
        X = pd.DataFrame({"cycle_number": cycles, "other": [0, 0, 0, 0]})

        model = LinearExtrapolationRULBaseline().fit(X, y)
        assert model.slope_ == pytest.approx(-1.0, abs=1e-5)
        assert model.intercept_ == pytest.approx(100.0, abs=1e-5)

        test_X = pd.DataFrame({"cycle_number": [30, 90]})
        preds = model.predict(test_X)
        assert preds[0] == pytest.approx(70.0, abs=1e-5)
        assert preds[1] == pytest.approx(10.0, abs=1e-5)

    def test_unfitted_raises_error(self):
        model = LinearExtrapolationRULBaseline()
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.predict(pd.DataFrame({"cycle_number": [1]}))

    def test_clipped_at_zero(self):
        # When cycle extends far past intercept, RUL cannot be negative
        X = pd.DataFrame({"cycle_number": [1, 2]})
        y = pd.Series([10.0, 5.0])
        model = LinearExtrapolationRULBaseline().fit(X, y)

        preds = model.predict(pd.DataFrame({"cycle_number": [100]}))
        assert preds[0] >= 0.0
