"""
Shared evaluation metrics for all demand forecasting models.
"""
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Full metric suite for demand regression.

    Metrics
    -------
    MAE   : Mean Absolute Error — average dollar error.
    RMSE  : Root Mean Squared Error — penalises large errors more.
    MdAE  : Median Absolute Error — robust to outlier machines.
    R²    : Coefficient of determination — fraction of variance explained.
    SMAPE : Symmetric MAPE (%) — percentage error safe for zero-demand days.
    Bias  : Mean signed error (pred − true). Positive → systematic overestimation.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mdae = float(np.median(np.abs(y_true - y_pred)))
    r2   = r2_score(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))

    # SMAPE: symmetric, handles zero denominator
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_vals = np.where(denom == 0, 0.0, np.abs(y_true - y_pred) / denom)
    smape = float(np.mean(smape_vals) * 100)

    return {
        "MAE":       round(mae,  4),
        "RMSE":      round(rmse, 4),
        "MdAE":      round(mdae, 4),
        "R2":        round(r2,   4),
        "SMAPE (%)": round(smape, 2),
        "Bias":      round(bias,  4),
    }


def compute_metrics_by_cluster(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clusters: np.ndarray,
) -> "pd.DataFrame":
    import pandas as pd

    rows = []
    for c in sorted(np.unique(clusters[~np.isnan(clusters.astype(float))])):
        mask = clusters == c
        m = compute_metrics(y_true[mask], y_pred[mask])
        rows.append({"cluster": int(c), "n": int(mask.sum()), **m})
    return pd.DataFrame(rows)
