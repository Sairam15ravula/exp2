"""Honest benchmarking comparing deep sequence models (LSTM) vs tree ensembles (XGBoost).

Rule 3: Always beat a baseline and report honest results across all folds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn

from ev_battery.advanced.dataset import (
    SEQUENCE_FEATURE_COLUMNS,
    create_sequence_windows,
)
from ev_battery.advanced.loss import PhysicsInformedLoss
from ev_battery.advanced.model import BatteryLSTM
from ev_battery.advanced.trainer import LSTMTrainer
from ev_battery.soh.dataset import (
    FEATURE_COLUMNS,
    get_feature_matrix_and_target,
    leave_one_battery_out_splits,
)
from ev_battery.soh.evaluation import evaluate_predictions
from ev_battery.soh.model import SOHModel


def count_monotonic_violations(preds: np.ndarray) -> float:
    """Fraction of consecutive steps where SOH unphysically increases."""
    if len(preds) < 2:
        return 0.0
    diffs = np.diff(preds)
    violations = np.sum(diffs > 1e-4)
    return float(violations / (len(preds) - 1))


def compare_lstm_vs_xgboost(
    df: pd.DataFrame,
    window_size: int = 5,
    epochs: int = 40,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Perform honest LOBO comparison between XGBoost, standard LSTM, and Physics-Informed LSTM.

    Args:
        df: Prepared SOH dataset across multiple batteries.
        window_size: Sequence window size for LSTM.
        epochs: Training epochs per LSTM fold.
        random_seed: Reproducibility seed.

    Returns:
        Dictionary of per-model aggregate metrics (mean ± std) and comparison table.
    """
    results_xgb: list[dict[str, float]] = []
    results_lstm_std: list[dict[str, float]] = []
    results_lstm_pinn: list[dict[str, float]] = []

    for held_out_battery, train_df, test_df in leave_one_battery_out_splits(df):
        # 1. Evaluate XGBoost (Phase 4 model)
        X_train_xgb, y_train_xgb = get_feature_matrix_and_target(train_df, list(FEATURE_COLUMNS))
        X_test_xgb, y_test_xgb = get_feature_matrix_and_target(test_df, list(FEATURE_COLUMNS))

        xgb = SOHModel(n_estimators=40, max_depth=3, learning_rate=0.08, random_state=random_seed)
        xgb.fit(X_train_xgb, y_train_xgb)
        preds_xgb = xgb.predict(X_test_xgb)
        m_xgb = evaluate_predictions(y_test_xgb, preds_xgb)
        m_xgb["mono_violation_rate"] = count_monotonic_violations(preds_xgb)
        results_xgb.append(m_xgb)

        # 2. Prepare Sequence Windows for LSTMs
        X_train_seq, y_train_seq = create_sequence_windows(
            train_df,
            window_size=window_size,
            feature_columns=list(SEQUENCE_FEATURE_COLUMNS),
        )
        X_test_seq, y_test_seq = create_sequence_windows(
            test_df,
            window_size=window_size,
            feature_columns=list(SEQUENCE_FEATURE_COLUMNS),
        )

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue

        # Standard LSTM (MSE loss)
        model_std = BatteryLSTM(input_dim=len(SEQUENCE_FEATURE_COLUMNS), hidden_dim=24, num_layers=1)
        trainer_std = LSTMTrainer(
            model=model_std,
            loss_fn=nn.MSELoss(),
            epochs=epochs,
            lr=0.008,
            random_seed=random_seed,
        )
        trainer_std.fit(X_train_seq, y_train_seq)
        preds_std = model_std.predict_numpy(X_test_seq)
        m_std = evaluate_predictions(y_test_seq, preds_std)
        m_std["mono_violation_rate"] = count_monotonic_violations(preds_std)
        results_lstm_std.append(m_std)

        # Physics-Informed LSTM (PINN Loss)
        model_pinn = BatteryLSTM(input_dim=len(SEQUENCE_FEATURE_COLUMNS), hidden_dim=24, num_layers=1)
        trainer_pinn = LSTMTrainer(
            model=model_pinn,
            loss_fn=PhysicsInformedLoss(lambda_mono=1.0, lambda_bound=1.0),
            epochs=epochs,
            lr=0.008,
            random_seed=random_seed,
        )
        trainer_pinn.fit(X_train_seq, y_train_seq)
        preds_pinn = model_pinn.predict_numpy(X_test_seq)
        m_pinn = evaluate_predictions(y_test_seq, preds_pinn)
        m_pinn["mono_violation_rate"] = count_monotonic_violations(preds_pinn)
        results_lstm_pinn.append(m_pinn)

    def _agg(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
        keys = ["rmse", "mae", "r2", "mono_violation_rate"]
        summary = {}
        for k in keys:
            vals = [r[k] for r in records]
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return summary

    summary = {
        "xgboost": _agg(results_xgb),
        "lstm_standard": _agg(results_lstm_std),
        "lstm_pinn": _agg(results_lstm_pinn),
    }

    return {
        "n_folds": len(results_xgb),
        "summary": summary,
    }


def format_advanced_comparison_table(results: dict[str, Any]) -> str:
    """Format comparison table between XGBoost, standard LSTM, and Physics-Informed LSTM."""
    s = results["summary"]
    lines = [
        "| Architecture | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) | Monotonic Violations |",
        "|---|---|---|---|---|",
    ]

    models = [
        ("XGBoost (Phase 4)", "xgboost"),
        ("Standard LSTM", "lstm_standard"),
        ("Physics-Informed LSTM (PINN)", "lstm_pinn"),
    ]

    for label, key in models:
        m = s[key]
        rmse_s = f"{m['rmse']['mean']:.4f} ± {m['rmse']['std']:.4f}"
        mae_s = f"{m['mae']['mean']:.4f} ± {m['mae']['std']:.4f}"
        r2_s = f"{m['r2']['mean']:.4f} ± {m['r2']['std']:.4f}"
        viol_s = f"{m['mono_violation_rate']['mean']*100:.1f}%"
        lines.append(f"| {label} | {rmse_s} | {mae_s} | {r2_s} | {viol_s} |")

    return "\n".join(lines)
