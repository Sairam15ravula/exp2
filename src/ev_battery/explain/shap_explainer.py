"""TreeSHAP feature attribution module for battery health models.

Computes exact Shapley values, verifies local additivity, and identifies
primary degradation drivers and mitigating operational factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from ev_battery.rul.model import RULModel
from ev_battery.soh.dataset import FEATURE_COLUMNS, validate_no_target_leakage
from ev_battery.soh.model import SOHModel


@dataclass
class FeatureContribution:
    """Attribution metrics for a single feature on an individual prediction."""

    feature_name: str
    feature_value: float
    shap_value: float
    abs_impact: float
    direction: str  # 'increases_prediction' or 'decreases_prediction'


@dataclass
class SHAPExplanation:
    """Complete local SHAP explanation for a single cycle inference."""

    target_type: str
    predicted_value: float
    base_value: float
    contributions: list[FeatureContribution]
    top_positive_drivers: list[FeatureContribution]
    top_negative_drivers: list[FeatureContribution]
    is_additive: bool


class BatterySHAPExplainer:
    """Computes and interprets TreeSHAP attributions for SOH and RUL models."""

    def __init__(self, model: Union[SOHModel, RULModel, XGBRegressor, GradientBoostingRegressor], target_type: str = "SOH"):
        self.target_type = target_type.upper()
        self.feature_names = FEATURE_COLUMNS

        # Extract underlying tree model
        if isinstance(model, SOHModel):
            self.model = model.regressor
            self.feature_names = model.feature_names
        elif isinstance(model, RULModel):
            # For RUL quantile models, explain the median regressor
            self.model = model.regressor_median
            self.feature_names = model.feature_names
        else:
            self.model = model

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

        # Base value handling (handles scalar or 1-element array)
        raw_base = self.explainer.expected_value
        if isinstance(raw_base, (np.ndarray, list)):
            self.base_value = float(raw_base[0])
        else:
            self.base_value = float(raw_base)

    def explain_instance(self, features: dict[str, float] | pd.DataFrame | pd.Series) -> SHAPExplanation:
        """Explain a single cycle's prediction with exact SHAP contributions.

        Args:
            features: Dictionary or DataFrame row of feature names and values.

        Returns:
            SHAPExplanation containing feature attributions and top drivers.
        """
        if isinstance(features, dict):
            # Rule 2: Validate zero target leakage
            validate_no_target_leakage(list(features.keys()))
            df = pd.DataFrame([{col: features.get(col, 0.0) for col in self.feature_names}])
        elif isinstance(features, pd.Series):
            df = pd.DataFrame([features[self.feature_names]])
        else:
            df = features[self.feature_names].copy()

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(df)
        shap_vals = np.asarray(raw_shap).flatten()
        feat_vals = df.iloc[0].to_numpy()

        # Model prediction
        pred_val = float(self.model.predict(df)[0])

        # Verify additivity: base_value + sum(shap_vals) == predicted_value
        sum_shap = float(np.sum(shap_vals))
        is_additive = bool(np.isclose(self.base_value + sum_shap, pred_val, atol=1e-3))

        contributions: list[FeatureContribution] = []
        for name, val, s_val in zip(self.feature_names, feat_vals, shap_vals):
            direction = "increases_prediction" if s_val >= 0 else "decreases_prediction"
            contributions.append(
                FeatureContribution(
                    feature_name=name,
                    feature_value=float(val),
                    shap_value=float(s_val),
                    abs_impact=float(np.abs(s_val)),
                    direction=direction,
                )
            )

        # Sort by absolute impact
        contributions.sort(key=lambda c: c.abs_impact, reverse=True)

        # Positive drivers (pushing prediction higher)
        pos_drivers = [c for c in contributions if c.shap_value > 0]
        # Negative drivers (pushing prediction lower)
        neg_drivers = [c for c in contributions if c.shap_value < 0]

        return SHAPExplanation(
            target_type=self.target_type,
            predicted_value=round(pred_val, 4),
            base_value=round(self.base_value, 4),
            contributions=contributions,
            top_positive_drivers=pos_drivers[:3],
            top_negative_drivers=neg_drivers[:3],
            is_additive=is_additive,
        )

    def explain_global(self, X: pd.DataFrame) -> dict[str, float]:
        """Compute global mean absolute SHAP feature importance.

        Args:
            X: Feature matrix across multiple cycles.

        Returns:
            Dictionary mapping feature name to mean absolute impact.
        """
        X_df = X[self.feature_names].copy()
        raw_shap = self.explainer.shap_values(X_df)
        mean_abs = np.mean(np.abs(raw_shap), axis=0)

        importance_dict = {
            name: float(imp)
            for name, imp in sorted(
                zip(self.feature_names, mean_abs),
                key=lambda x: x[1],
                reverse=True,
            )
        }
        return importance_dict
