"""Feature importance and SHAP explainability for SOH models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap

from ev_battery.soh.model import SOHModel


def get_feature_importances(model: SOHModel) -> dict[str, float]:
    """Get native feature importance from the fitted XGBoost regressor (gain)."""
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted to get feature importances.")
    raw_importances = model.regressor.feature_importances_
    return {
        feat: float(imp)
        for feat, imp in sorted(
            zip(model.feature_names, raw_importances),
            key=lambda x: x[1],
            reverse=True,
        )
    }


def compute_shap_values(
    model: SOHModel,
    X: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute TreeSHAP explanation values and mean absolute SHAP impacts.

    Args:
        model: Fitted SOHModel.
        X: Feature matrix.

    Returns:
        (shap_values_matrix, mean_abs_shap_dict)
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted to compute SHAP values.")

    X_df = X[model.feature_names].copy()
    explainer = shap.TreeExplainer(model.regressor)
    shap_vals = explainer.shap_values(X_df)

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_summary = {
        feat: float(imp)
        for feat, imp in sorted(
            zip(model.feature_names, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )
    }

    return shap_vals, shap_summary
