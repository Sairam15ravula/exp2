"""Remaining Useful Life (RUL) estimation model with quantile prediction intervals.

Produces median point predictions alongside 90% prediction intervals [q_0.05, q_0.95].
Rule 6: Stores training metadata (SHA256, feature list, library versions, metrics).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from ev_battery.rul.dataset import (
    FEATURE_COLUMNS,
    validate_no_rul_target_leakage,
)


def compute_rul_dataset_sha256(X: pd.DataFrame, y: pd.Series | np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of training data for Rule 6 reproducibility."""
    hasher = hashlib.sha256()
    hasher.update(",".join(X.columns).encode("utf-8"))
    hasher.update(X.to_numpy().tobytes())
    y_arr = np.asarray(y, dtype=float)
    hasher.update(y_arr.tobytes())
    return hasher.hexdigest()


class RULModel:
    """Quantile Gradient Boosting model for RUL estimation with uncertainty bands.

    Fits three quantile regressors:
    - alpha = 0.05 (lower bound of 90% prediction interval)
    - alpha = 0.50 (median point prediction)
    - alpha = 0.95 (upper bound of 90% prediction interval)
    """

    def __init__(
        self,
        n_estimators: int = 60,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 42,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize RUL quantile model.

        Args:
            n_estimators: Number of boosting stages.
            max_depth: Depth of individual trees.
            learning_rate: Shrinkage parameter.
            random_state: Seed for reproducibility (Rule 6).
            lower_quantile: Lower uncertainty quantile (default: 0.05).
            upper_quantile: Upper uncertainty quantile (default: 0.95).
            feature_names: Allowed non-leaky feature names.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.feature_names = feature_names if feature_names is not None else list(FEATURE_COLUMNS)

        validate_no_rul_target_leakage(self.feature_names)

        self.regressor_lower = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.lower_quantile,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.regressor_median = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.50,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self.regressor_upper = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.upper_quantile,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

        self.is_fitted = False
        self.training_metadata: dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        dataset_version: str = "v1.0-synthetic-rul",
    ) -> RULModel:
        """Fit all three quantile regressors on training features and RUL targets."""
        validate_no_rul_target_leakage(list(X.columns))

        X_df = X[self.feature_names].copy()
        y_arr = np.asarray(y, dtype=float)

        self.regressor_lower.fit(X_df, y_arr)
        self.regressor_median.fit(X_df, y_arr)
        self.regressor_upper.fit(X_df, y_arr)

        self.is_fitted = True

        import sklearn
        import xgboost

        self.training_metadata = {
            "dataset_version": dataset_version,
            "data_sha256": compute_rul_dataset_sha256(X_df, y_arr),
            "feature_list": self.feature_names,
            "random_seed": self.random_state,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile,
            },
            "library_versions": {
                "scikit-learn": sklearn.__version__,
                "xgboost": xgboost.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return point prediction (median quantile q_0.50), clipped at 0."""
        if not self.is_fitted:
            raise RuntimeError("RULModel must be fitted before predict().")
        validate_no_rul_target_leakage(list(X.columns))
        X_df = X[self.feature_names].copy()
        raw = self.regressor_median.predict(X_df)
        return np.clip(raw, 0.0, None)

    def predict_interval(
        self,
        X: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return 90% prediction interval and median: (lower_bound, median, upper_bound).

        Guarantees non-crossing quantiles: lower <= median <= upper.
        """
        if not self.is_fitted:
            raise RuntimeError("RULModel must be fitted before predict_interval().")
        validate_no_rul_target_leakage(list(X.columns))
        X_df = X[self.feature_names].copy()

        raw_lower = self.regressor_lower.predict(X_df)
        raw_median = self.regressor_median.predict(X_df)
        raw_upper = self.regressor_upper.predict(X_df)

        # Enforce non-crossing and non-negativity
        median = np.clip(raw_median, 0.0, None)
        lower = np.clip(np.minimum(raw_lower, median), 0.0, None)
        upper = np.clip(np.maximum(raw_upper, median), 0.0, None)

        return lower, median, upper


def save_rul_model(
    model: RULModel,
    directory: str | Path,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save trained RUL model ensemble and Rule 6 metadata sidecar."""
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)

    ensemble = {
        "lower": model.regressor_lower,
        "median": model.regressor_median,
        "upper": model.regressor_upper,
    }
    model_path = dest / "rul_quantile_ensemble.joblib"
    joblib.dump(ensemble, model_path)

    metadata = dict(model.training_metadata)
    if metrics:
        metadata["metrics"] = metrics

    meta_path = dest / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return dest


def load_rul_model(directory: str | Path) -> tuple[RULModel, dict[str, Any]]:
    """Load RUL model ensemble and training metadata."""
    dest = Path(directory)
    model_path = dest / "rul_quantile_ensemble.joblib"
    meta_path = dest / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model ensemble or metadata missing in {directory}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    hp = metadata["hyperparameters"]
    model = RULModel(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        random_state=metadata["random_seed"],
        lower_quantile=hp["lower_quantile"],
        upper_quantile=hp["upper_quantile"],
        feature_names=metadata["feature_list"],
    )

    ensemble = joblib.load(model_path)
    model.regressor_lower = ensemble["lower"]
    model.regressor_median = ensemble["median"]
    model.regressor_upper = ensemble["upper"]
    model.is_fitted = True
    model.training_metadata = metadata

    return model, metadata
