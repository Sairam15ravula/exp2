"""Remaining Useful Life (RUL) prediction module with uncertainty quantification.

Rule 4: RUL is defined as cycles until SOH <= 80% (end of life),
NOT cycles until the experiment ended.
"""

from ev_battery.rul.baselines import (
    LinearExtrapolationRULBaseline,
    PredictTheMeanRULBaseline,
)
from ev_battery.rul.dataset import (
    EOL_SOH_THRESHOLD,
    FEATURE_COLUMNS,
    find_eol_cycle,
    get_rul_feature_matrix_and_target,
    leave_one_battery_out_rul_splits,
    prepare_rul_dataset,
)
from ev_battery.rul.evaluation import (
    evaluate_rul_lobo,
    evaluate_rul_predictions,
    format_rul_evaluation_table,
)
from ev_battery.rul.model import RULModel, load_rul_model, save_rul_model

__all__ = [
    "EOL_SOH_THRESHOLD",
    "FEATURE_COLUMNS",
    "LinearExtrapolationRULBaseline",
    "PredictTheMeanRULBaseline",
    "RULModel",
    "evaluate_rul_lobo",
    "evaluate_rul_predictions",
    "find_eol_cycle",
    "format_rul_evaluation_table",
    "get_rul_feature_matrix_and_target",
    "leave_one_battery_out_rul_splits",
    "load_rul_model",
    "prepare_rul_dataset",
    "save_rul_model",
]
