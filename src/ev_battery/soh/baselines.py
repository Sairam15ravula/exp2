"""Baseline models for SOH estimation.

Rule 3: ALWAYS BEAT A BASELINE.
Report every model against:
(a) predict-the-mean
(b) a simple linear-in-cycle model
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PredictTheMeanBaseline:
    """Predicts the mean SOH of the training set for all queries.

    Serves as the simplest naive baseline (constant predictor).
    """

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> PredictTheMeanBaseline:
        """Fit baseline on training targets."""
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) == 0:
            raise ValueError("Training targets y cannot be empty.")
        self.mean_ = float(np.mean(y_arr))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict constant mean for all samples in X."""
        if self.mean_ is None:
            raise RuntimeError("PredictTheMeanBaseline must be fitted before predict().")
        n = len(X)
        return np.full(n, self.mean_, dtype=float)


class LinearInCycleBaseline:
    """Simple linear-in-cycle model: SOH = slope * cycle_number + intercept.

    Captures standard linear degradation over cycle count.
    """

    def __init__(self) -> None:
        self.slope_: float | None = None
        self.intercept_: float | None = None

    def _extract_cycles(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Extract cycle_number array from feature matrix."""
        if isinstance(X, pd.DataFrame):
            if "cycle_number" in X.columns:
                return X["cycle_number"].to_numpy(dtype=float)
            # If cycle_number column not found, use first column
            return X.iloc[:, 0].to_numpy(dtype=float)
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            return arr
        return arr[:, 0]

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> LinearInCycleBaseline:
        """Fit linear regression on cycle numbers."""
        cycles = self._extract_cycles(X)
        y_arr = np.asarray(y, dtype=float)

        if len(cycles) != len(y_arr):
            raise ValueError("Length of X and y must match.")
        if len(cycles) < 2:
            raise ValueError("At least 2 points required to fit linear baseline.")

        # OLS 1D linear fit: y = slope * cycles + intercept
        coeffs = np.polyfit(cycles, y_arr, deg=1)
        self.slope_ = float(coeffs[0])
        self.intercept_ = float(coeffs[1])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict SOH using fitted linear slope and intercept."""
        if self.slope_ is None or self.intercept_ is None:
            raise RuntimeError("LinearInCycleBaseline must be fitted before predict().")
        cycles = self._extract_cycles(X)
        preds = self.slope_ * cycles + self.intercept_
        return np.clip(preds, 0.0, 1.2)
