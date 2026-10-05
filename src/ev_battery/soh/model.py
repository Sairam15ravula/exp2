"""XGBoost model for SOH estimation with monotonic constraints and reproducibility metadata.

Rule 2: Features must never contain capacity or initial_capacity.
Rule 6: Store training metadata (dataset version, metrics, feature list, library versions, SHA256).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

from ev_battery.soh.dataset import FEATURE_COLUMNS, validate_no_target_leakage


def compute_dataset_sha256(X: pd.DataFrame, y: pd.Series | np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of training data for reproducibility (Rule 6)."""
    hasher = hashlib.sha256()
    # Serialize X columns and data
    hasher.update(",".join(X.columns).encode("utf-8"))
    hasher.update(X.to_numpy().tobytes())
    y_arr = np.asarray(y, dtype=float)
    hasher.update(y_arr.tobytes())
    return hasher.hexdigest()


class SOHModel:
    """XGBoost regressor for State of Health (SOH) estimation.

    Supports monotonic constraints (e.g. SOH non-increasing with cycle_number).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42,
        monotone_constraints: dict[str, int] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize SOH model.

        Args:
            n_estimators: Number of boosting trees.
            max_depth: Maximum tree depth.
            learning_rate: Step size shrinkage.
            random_state: Random seed for reproducibility (Rule 6).
            monotone_constraints: Dict mapping feature name to constraint:
                1 (increasing), -1 (decreasing), 0 (none).
            feature_names: Expected list of feature names. Defaults to FEATURE_COLUMNS.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.monotone_constraints = monotone_constraints
        self.feature_names = feature_names if feature_names is not None else list(FEATURE_COLUMNS)

        validate_no_target_leakage(self.feature_names)

        # Build XGBoost regressor
        xgb_kwargs: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }
        if self.monotone_constraints is not None:
            xgb_kwargs["monotone_constraints"] = self.monotone_constraints

        self.regressor = XGBRegressor(**xgb_kwargs)
        self.is_fitted = False
        self.training_metadata: dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        dataset_version: str = "v1.0-synthetic",
    ) -> SOHModel:
        """Fit model on training features and SOH targets."""
        validate_no_target_leakage(list(X.columns))

        # Ensure correct column ordering
        X_df = X[self.feature_names].copy()
        y_arr = np.asarray(y, dtype=float)

        self.regressor.fit(X_df, y_arr)
        self.is_fitted = True

        import sklearn
        import xgboost

        # Rule 6: Track training metadata
        self.training_metadata = {
            "dataset_version": dataset_version,
            "data_sha256": compute_dataset_sha256(X_df, y_arr),
            "feature_list": self.feature_names,
            "random_seed": self.random_state,
            "monotone_constraints": self.monotone_constraints,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
            "library_versions": {
                "xgboost": xgboost.__version__,
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict SOH for feature matrix X."""
        if not self.is_fitted:
            raise RuntimeError("SOHModel must be fitted before predict().")
        validate_no_target_leakage(list(X.columns))
        X_df = X[self.feature_names].copy()
        raw_preds = self.regressor.predict(X_df)
        return np.clip(raw_preds, 0.0, 1.2)


def save_soh_model(
    model: SOHModel,
    directory: str | Path,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save trained SOH model artifact and Rule 6 metadata sidecar.

    Args:
        model: Fitted SOHModel.
        directory: Destination directory.
        metrics: Optional dictionary of evaluation metrics to store in metadata.

    Returns:
        Path to the saved directory.
    """
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)

    # Save model weights / regressor
    model_path = dest / "soh_model.joblib"
    joblib.dump(model.regressor, model_path)

    # Save metadata JSON (Rule 6)
    metadata = dict(model.training_metadata)
    if metrics:
        metadata["metrics"] = metrics

    meta_path = dest / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return dest


def load_soh_model(directory: str | Path) -> tuple[SOHModel, dict[str, Any]]:
    """Load SOH model and associated training metadata."""
    dest = Path(directory)
    model_path = dest / "soh_model.joblib"
    meta_path = dest / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model or metadata missing in {directory}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SOHModel(
        n_estimators=metadata["hyperparameters"]["n_estimators"],
        max_depth=metadata["hyperparameters"]["max_depth"],
        learning_rate=metadata["hyperparameters"]["learning_rate"],
        random_state=metadata["random_seed"],
        monotone_constraints=metadata.get("monotone_constraints"),
        feature_names=metadata["feature_list"],
    )
    model.regressor = joblib.load(model_path)
    model.is_fitted = True
    model.training_metadata = metadata
    return model, metadata
