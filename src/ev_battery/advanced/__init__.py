"""Advanced deep learning models with physics-informed loss for battery prognostics.

Phase 7: Sequence modeling with LSTM and thermodynamic monotonicity constraints.
"""

from ev_battery.advanced.comparison import compare_lstm_vs_xgboost
from ev_battery.advanced.dataset import (
    SEQUENCE_FEATURE_COLUMNS,
    create_sequence_windows,
    cross_temperature_split,
)
from ev_battery.advanced.loss import PhysicsInformedLoss
from ev_battery.advanced.model import BatteryLSTM
from ev_battery.advanced.trainer import (
    LSTMTrainer,
    load_lstm_checkpoint,
    save_lstm_checkpoint,
)

__all__ = [
    "BatteryLSTM",
    "LSTMTrainer",
    "PhysicsInformedLoss",
    "SEQUENCE_FEATURE_COLUMNS",
    "compare_lstm_vs_xgboost",
    "create_sequence_windows",
    "cross_temperature_split",
    "load_lstm_checkpoint",
    "save_lstm_checkpoint",
]
