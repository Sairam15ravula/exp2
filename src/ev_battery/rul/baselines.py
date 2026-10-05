"""Baseline models for Remaining Useful Life (RUL) estimation.

Rule 3: ALWAYS BEAT A BASELINE.
Report every model against:
(a) predict-the-mean
(b) a simple linear-in-cycle model
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PredictTheMeanRULBaseline:
    """Predicts the mean RUL observed in the training set for all cycles.

    Serves as the simplest naive baseline (constant predictor).
    """

    def __init__(self) -> None:
        self.mean_rul_: float | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> PredictTheMeanRULBaseline:
        """Fit baseline by calculating the mean training RUL."""
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Training targets y cannot be empty.")
        self.mean_rul_ = float(np.mean(y_arr))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict constant mean RUL for all queries."""
        if self.mean_rul_ is None:
            raise RuntimeError("PredictTheMeanRULBaseline must be fitted before predict().")
        n = len(X)
        return np.full(n, self.mean_rul_, dtype=float)


class LinearExtrapolationRULBaseline:
    """Linear cycle baseline for RUL.

    Fits an OLS regression: RUL = slope * cycle_number + intercept.
    Captures the average rate at which remaining cycles decrease with cycle index.
    """

    def __init__(self) -> None:
        self.slope_: float | None = None
        self.intercept_: float | None = None

    def _extract_cycles(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract cycle_number from feature matrix."""
        if isinstance(X, pd.DataFrame):
            if "cycle_number" in X.columns:
                return X["cycle_number"].to_numpy(dtype=float)
            return X.iloc[:, 0].to_numpy(dtype=float)
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            return arr
        return arr[:, 0]

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> LinearExtrapolationRULBaseline:
        """Fit linear regression on cycle numbers."""
        cycles = self._extract_cycles(X)
        y_arr = np.asarray(y, dtype=float)

        if len(cycles) != len(y_arr):
            raise ValueError("Length of X and y must match.")
        if len(cycles) < 2:
            raise ValueError("At least 2 cycles required to fit linear RUL baseline.")

        # OLS 1D linear fit: RUL = slope * cycle + intercept
        coeffs = np.polyfit(cycles, y_arr, deg=1)
        self.slope_ = float(coeffs[0])
        self.intercept_ = float(coeffs[1])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict remaining useful life, clipped at 0 (cannot have negative remaining cycles)."""
        if self.slope_ is None or self.intercept_ is None:
            raise RuntimeError("LinearExtrapolationRULBaseline must be fitted before predict().")
        cycles = self._extract_cycles(X)
        preds = self.slope_ * cycles + self.intercept_
        return np.clip(preds, 0.0, None)
