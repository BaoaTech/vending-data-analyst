"""
XGBoost supervised regression for per-machine daily demand forecasting.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.metrics import compute_metrics, compute_metrics_by_cluster


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict = None,
) -> xgb.XGBRegressor:
    default_params = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    if params:
        default_params.update(params)

    model = xgb.XGBRegressor(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[np.ndarray, Dict[str, float]]:
    y_pred = np.clip(model.predict(X_test), 0, None)
    return y_pred, compute_metrics(y_test.values, y_pred)


def evaluate_by_cluster(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cluster_labels: pd.Series,
) -> pd.DataFrame:
    y_pred   = np.clip(model.predict(X_test), 0, None)
    y_true   = y_test.reset_index(drop=True).values
    clusters = cluster_labels.reset_index(drop=True).values
    return compute_metrics_by_cluster(y_true, y_pred, clusters)


def save_model(model: xgb.XGBRegressor, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def load_model(path: str) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model
