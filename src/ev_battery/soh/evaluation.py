"""Evaluation routines for SOH estimation.

Rule 1: Leave-One-Battery-Out cross-validation across all batteries.
Rule 3: ALWAYS BEAT A BASELINE.
Report every model against (a) predict-the-mean and (b) simple linear-in-cycle model.
Report mean ± std across ALL folds, never a single lucky fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ev_battery.soh.baselines import LinearInCycleBaseline, PredictTheMeanBaseline
from ev_battery.soh.dataset import (
    FEATURE_COLUMNS,
    get_feature_matrix_and_target,
    leave_one_battery_out_splits,
)
from ev_battery.soh.model import SOHModel


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics: RMSE, MAE, Max Error, R2."""
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

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_err,
        "r2": r2,
    }


def evaluate_lobo(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    monotone_constraints: dict[str, int] | None = None,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run Leave-One-Battery-Out evaluation comparing SOHModel against baselines.

    Args:
        df: Prepared SOH dataframe with multiple batteries.
        feature_columns: Input features (non-leaky). Defaults to FEATURE_COLUMNS.
        monotone_constraints: Monotonic constraints for XGBoost.
        n_estimators: Number of XGBoost trees.
        max_depth: Depth of XGBoost trees.
        learning_rate: XGBoost learning rate.
        random_state: Seed for reproducibility (Rule 6).

    Returns:
        Dictionary containing fold results, aggregate statistics (mean ± std),
        and baseline comparison verdict.
    """
    cols = feature_columns if feature_columns is not None else list(FEATURE_COLUMNS)

    fold_results: list[dict[str, Any]] = []

    for held_out_battery, train_df, test_df in leave_one_battery_out_splits(df):
        X_train, y_train = get_feature_matrix_and_target(train_df, cols)
        X_test, y_test = get_feature_matrix_and_target(test_df, cols)

        # 1. Baseline: Predict the Mean
        mean_model = PredictTheMeanBaseline().fit(X_train, y_train)
        mean_preds = mean_model.predict(X_test)
        mean_metrics = evaluate_predictions(y_test, mean_preds)

        # 2. Baseline: Linear in Cycle
        linear_model = LinearInCycleBaseline().fit(X_train, y_train)
        linear_preds = linear_model.predict(X_test)
        linear_metrics = evaluate_predictions(y_test, linear_preds)

        # 3. XGBoost SOH Model
        xgb_model = SOHModel(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            monotone_constraints=monotone_constraints,
            feature_names=cols,
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_metrics = evaluate_predictions(y_test, xgb_preds)

        fold_results.append({
            "held_out_battery": held_out_battery,
            "n_test_cycles": len(test_df),
            "mean_baseline": mean_metrics,
            "linear_baseline": linear_metrics,
            "xgboost": xgb_metrics,
        })

    def _agg_metrics(model_key: str) -> dict[str, dict[str, float]]:
        metrics_keys = ["rmse", "mae", "r2", "max_error"]
        aggregated = {}
        for m in metrics_keys:
            vals = [fold[model_key][m] for fold in fold_results]
            aggregated[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
        return aggregated

    summary = {
        "mean_baseline": _agg_metrics("mean_baseline"),
        "linear_baseline": _agg_metrics("linear_baseline"),
        "xgboost": _agg_metrics("xgboost"),
    }

    # Rule 3 Check: Beat both baselines across all folds
    xgb_rmse_mean = summary["xgboost"]["rmse"]["mean"]
    mean_rmse_mean = summary["mean_baseline"]["rmse"]["mean"]
    linear_rmse_mean = summary["linear_baseline"]["rmse"]["mean"]

    beats_mean = xgb_rmse_mean < mean_rmse_mean
    beats_linear = xgb_rmse_mean < linear_rmse_mean

    return {
        "n_folds": len(fold_results),
        "folds": fold_results,
        "summary": summary,
        "beats_mean_baseline": beats_mean,
        "beats_linear_baseline": beats_linear,
        "beats_all_baselines": beats_mean and beats_linear,
    }


def format_evaluation_table(results: dict[str, Any]) -> str:
    """Format evaluation summary as a clean markdown table for reports."""
    summary = results["summary"]
    lines = [
        "| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |",
        "|---|---|---|---|",
    ]
    labels = [
        ("Predict-the-mean", "mean_baseline"),
        ("Linear-in-cycle", "linear_baseline"),
        ("XGBoost (SOH)", "xgboost"),
    ]
    for label, key in labels:
        s = summary[key]
        rmse_str = f"{s['rmse']['mean']:.4f} ± {s['rmse']['std']:.4f}"
        mae_str = f"{s['mae']['mean']:.4f} ± {s['mae']['std']:.4f}"
        r2_str = f"{s['r2']['mean']:.4f} ± {s['r2']['std']:.4f}"
        lines.append(f"| {label} | {rmse_str} | {mae_str} | {r2_str} |")

    return "\n".join(lines)
