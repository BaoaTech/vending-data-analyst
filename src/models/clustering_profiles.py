"""
K-Means cluster profile baseline for demand prediction.

Rationale: each machine is assigned a cluster from EDA (Axis 5).
The cluster's mean daily demand in the training period becomes the
point forecast for every machine in that cluster during the test period.
This serves as a naive baseline to benchmark supervised models against.
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.models.metrics import compute_metrics, compute_metrics_by_cluster


def build_cluster_profiles(
    train_daily: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cluster-level demand statistics from the training period.

    Returns a DataFrame indexed by cluster with columns:
        mean_daily_sale, median_daily_sale, std_daily_sale, n_observations, n_machines
    """
    if "cluster" in train_daily.columns:
        merged = train_daily.copy()
    else:
        merged = train_daily.merge(
            clusters[["vm_control", "cluster"]], on="vm_control", how="left"
        )
    profiles = (
        merged.groupby("cluster")["total_sale"]
        .agg(
            mean_daily_sale="mean",
            median_daily_sale="median",
            std_daily_sale="std",
            n_observations="count",
        )
        .reset_index()
    )
    machine_counts = (
        clusters.groupby("cluster")["vm_control"].count()
        .reset_index()
        .rename(columns={"vm_control": "n_machines"})
    )
    profiles = profiles.merge(machine_counts, on="cluster")
    return profiles


def predict_cluster_baseline(
    test_daily: pd.DataFrame,
    clusters: pd.DataFrame,
    profiles: pd.DataFrame,
    strategy: str = "mean",
) -> np.ndarray:
    """
    Assign cluster mean (or median) demand to each row in test_daily.
    strategy: 'mean' | 'median'
    """
    col = "mean_daily_sale" if strategy == "mean" else "median_daily_sale"
    if "cluster" not in test_daily.columns:
        df = test_daily.merge(clusters[["vm_control", "cluster"]], on="vm_control", how="left")
    else:
        df = test_daily.copy()
    df = df.merge(profiles[["cluster", col]], on="cluster", how="left")
    return df[col].values


def evaluate_cluster_baseline(
    test_daily: pd.DataFrame,
    clusters: pd.DataFrame,
    profiles: pd.DataFrame,
    strategy: str = "mean",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Evaluate the cluster-mean baseline against actual test demand.
    Returns (y_pred, full_metrics_dict).
    """
    y_pred = predict_cluster_baseline(test_daily, clusters, profiles, strategy)
    y_true = test_daily["total_sale"].values

    valid  = ~np.isnan(y_pred)
    return y_pred, compute_metrics(y_true[valid], y_pred[valid])


def evaluate_by_cluster(
    test_daily: pd.DataFrame,
    clusters: pd.DataFrame,
    profiles: pd.DataFrame,
    strategy: str = "mean",
) -> pd.DataFrame:
    """Return full metrics per cluster."""
    col = "mean_daily_sale" if strategy == "mean" else "median_daily_sale"
    if "cluster" not in test_daily.columns:
        df = test_daily.merge(clusters[["vm_control", "cluster"]], on="vm_control", how="left")
    else:
        df = test_daily.copy()
    df  = df.merge(profiles[["cluster", col]], on="cluster", how="left")

    valid  = ~df[col].isna()
    y_true = df.loc[valid, "total_sale"].values
    y_pred = df.loc[valid, col].values
    clust  = df.loc[valid, "cluster"].values

    return compute_metrics_by_cluster(y_true, y_pred, clust)
