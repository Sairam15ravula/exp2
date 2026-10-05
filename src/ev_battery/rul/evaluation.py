"""Evaluation routines for Remaining Useful Life (RUL) estimation with uncertainty.

Rule 1: Leave-One-Battery-Out cross-validation across all batteries.
Rule 3: ALWAYS BEAT A BASELINE (Predict-the-mean and Linear-in-cycle).
Report mean ± std across ALL folds.
Evaluates prediction intervals: PICP (coverage) and MPIW (sharpness/width).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ev_battery.rul.baselines import (
    LinearExtrapolationRULBaseline,
    PredictTheMeanRULBaseline,
)
from ev_battery.rul.dataset import (
    FEATURE_COLUMNS,
    get_rul_feature_matrix_and_target,
    leave_one_battery_out_rul_splits,
)
from ev_battery.rul.model import RULModel


def evaluate_rul_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_lower: pd.Series | np.ndarray | None = None,
    y_upper: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute RUL regression metrics and prediction interval quality.

    Args:
        y_true: True RUL values.
        y_pred: Predicted point RUL values.
        y_lower: Optional lower prediction interval bound (e.g. 5th percentile).
        y_upper: Optional upper prediction interval bound (e.g. 95th percentile).

    Returns:
        Dictionary of RMSE, MAE, max_error, R2, and optionally PICP and MPIW.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if len(yt) == 0:
        raise ValueError("Cannot evaluate empty predictions.")

    residuals = yt - yp
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    max_err = float(np.max(np.abs(residuals)))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    metrics: dict[str, float] = {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_err,
        "r2": r2,
    }

    if y_lower is not None and y_upper is not None:
        yl = np.asarray(y_lower, dtype=float)
        yu = np.asarray(y_upper, dtype=float)
        # Prediction Interval Coverage Probability (PICP)
        covered = (yt >= yl) & (yt <= yu)
        picp = float(np.mean(covered))
        # Mean Prediction Interval Width (MPIW)
        mpiw = float(np.mean(yu - yl))
        metrics["picp"] = picp
        metrics["mpiw"] = mpiw

    return metrics


def evaluate_rul_lobo(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    n_estimators: int = 60,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run Leave-One-Battery-Out cross-validation comparing RULModel against baselines.

    Args:
        df: Prepared RUL dataset.
        feature_columns: Feature names to include.
        n_estimators: Number of boosting trees.
        max_depth: Depth of trees.
        learning_rate: Learning rate.
        random_state: Fixed seed (Rule 6).

    Returns:
        Dictionary with per-fold results, aggregated mean ± std across all folds,
        and baseline comparison flags.
    """
    cols = feature_columns if feature_columns is not None else list(FEATURE_COLUMNS)
    fold_results: list[dict[str, Any]] = []

    for held_out_battery, train_df, test_df in leave_one_battery_out_rul_splits(df):
        X_train, y_train = get_rul_feature_matrix_and_target(train_df, cols)
        X_test, y_test = get_rul_feature_matrix_and_target(test_df, cols)

        # 1. Baseline: Predict the Mean
        mean_model = PredictTheMeanRULBaseline().fit(X_train, y_train)
        mean_preds = mean_model.predict(X_test)
        mean_metrics = evaluate_rul_predictions(y_test, mean_preds)

        # 2. Baseline: Linear in Cycle
        linear_model = LinearExtrapolationRULBaseline().fit(X_train, y_train)
        linear_preds = linear_model.predict(X_test)
        linear_metrics = evaluate_rul_predictions(y_test, linear_preds)

        # 3. Quantile RUL Model
        rul_model = RULModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            feature_names=cols,
        )
        rul_model.fit(X_train, y_train)
        lower, median, upper = rul_model.predict_interval(X_test)
        rul_metrics = evaluate_rul_predictions(y_test, median, lower, upper)

        fold_results.append({
            "held_out_battery": held_out_battery,
            "n_test_cycles": len(test_df),
            "mean_baseline": mean_metrics,
            "linear_baseline": linear_metrics,
            "rul_model": rul_metrics,
        })

    def _agg(model_key: str, metric_name: str) -> dict[str, float]:
        vals = [f[model_key][metric_name] for f in fold_results if metric_name in f[model_key]]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    summary = {
        "mean_baseline": {
            "rmse": _agg("mean_baseline", "rmse"),
            "mae": _agg("mean_baseline", "mae"),
            "r2": _agg("mean_baseline", "r2"),
        },
        "linear_baseline": {
            "rmse": _agg("linear_baseline", "rmse"),
            "mae": _agg("linear_baseline", "mae"),
            "r2": _agg("linear_baseline", "r2"),
        },
        "rul_model": {
            "rmse": _agg("rul_model", "rmse"),
            "mae": _agg("rul_model", "mae"),
            "r2": _agg("rul_model", "r2"),
            "picp": _agg("rul_model", "picp"),
            "mpiw": _agg("rul_model", "mpiw"),
        },
    }

    rul_rmse_mean = summary["rul_model"]["rmse"]["mean"]
    mean_rmse_mean = summary["mean_baseline"]["rmse"]["mean"]
    linear_rmse_mean = summary["linear_baseline"]["rmse"]["mean"]

    beats_mean = rul_rmse_mean < mean_rmse_mean
    beats_linear = rul_rmse_mean < linear_rmse_mean

    return {
        "n_folds": len(fold_results),
        "folds": fold_results,
        "summary": summary,
        "beats_mean_baseline": beats_mean,
        "beats_linear_baseline": beats_linear,
        "beats_all_baselines": beats_mean and beats_linear,
    }


def format_rul_evaluation_table(results: dict[str, Any]) -> str:
    """Format RUL evaluation results as a clean Markdown table."""
    summary = results["summary"]
    lines = [
        "| Model | RMSE (mean ± std) | MAE (mean ± std) | 90% PICP Coverage | 90% Interval Width (MPIW) |",
        "|---|---|---|---|---|",
    ]

    # Predict-the-mean
    mb = summary["mean_baseline"]
    lines.append(
        f"| Predict-the-mean | {mb['rmse']['mean']:.2f} ± {mb['rmse']['std']:.2f} | "
        f"{mb['mae']['mean']:.2f} ± {mb['mae']['std']:.2f} | N/A | N/A |"
    )

    # Linear-in-cycle
    lb = summary["linear_baseline"]
    lines.append(
        f"| Linear-in-cycle | {lb['rmse']['mean']:.2f} ± {lb['rmse']['std']:.2f} | "
        f"{lb['mae']['mean']:.2f} ± {lb['mae']['std']:.2f} | N/A | N/A |"
    )

    # RUL Model
    rm = summary["rul_model"]
    lines.append(
        f"| **RUL Quantile Model** | **{rm['rmse']['mean']:.2f} ± {rm['rmse']['std']:.2f}** | "
        f"**{rm['mae']['mean']:.2f} ± {rm['mae']['std']:.2f}** | "
        f"{rm['picp']['mean']*100:.1f}% ± {rm['picp']['std']*100:.1f}% | "
        f"{rm['mpiw']['mean']:.1f} ± {rm['mpiw']['std']:.1f} cycles |"
    )

    return "\n".join(lines)
