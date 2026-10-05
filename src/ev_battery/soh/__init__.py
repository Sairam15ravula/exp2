"""State of Health (SOH) estimation module.

Contains dataset preparation with zero target leakage, Leave-One-Battery-Out (LOBO)
cross-validation, baselines, XGBoost models with monotonic constraints,
reproducibility metadata storage, and feature explainability.
"""

from ev_battery.soh.baselines import LinearInCycleBaseline, PredictTheMeanBaseline
from ev_battery.soh.dataset import (
    FEATURE_COLUMNS,
    leave_one_battery_out_splits,
    prepare_soh_dataset,
    validate_no_target_leakage,
)
from ev_battery.soh.evaluation import (
    evaluate_lobo,
    evaluate_predictions,
)
from ev_battery.soh.model import SOHModel, load_soh_model, save_soh_model

__all__ = [
    "FEATURE_COLUMNS",
    "LinearInCycleBaseline",
    "PredictTheMeanBaseline",
    "SOHModel",
    "evaluate_lobo",
    "evaluate_predictions",
    "leave_one_battery_out_splits",
    "load_soh_model",
    "prepare_soh_dataset",
    "save_soh_model",
    "validate_no_target_leakage",
]
