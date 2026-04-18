"""
Feature engineering pipeline: raw snapshots → daily machine-level feature matrix.
"""
from pathlib import Path
import pandas as pd
import numpy as np


# ── Column aliases (raw QuickBase export) ──────────────────────────────────────
_DATE_COL = "Sale Register Date"
_SALE_COL = "Total Sale"
_QTY_COL = "Quantity Sale"
_ID_COL = "SALES_DETAIL ID"
_VM_COL = "vm_control"

# Feature columns used by supervised models (must exist after build_feature_matrix)
FEATURE_COLS = [
    "dow", "month", "is_weekend", "week_of_year",
    "lag_1", "lag_7", "lag_14",
    "roll_mean_7", "roll_mean_14",
    "roll_std_7", "roll_std_14",
    "cluster",
]

# Compact feature set for LSTM sequences (low cardinality, no leakage)
LSTM_COLS = ["total_sale", "dow", "is_weekend", "cluster"]

TARGET_COL = "total_sale"


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_latest_snapshot(snapshots_dir: str) -> pd.DataFrame:
    files = sorted(Path(snapshots_dir).glob("buckkmrgh_*.csv"))
    if not files:
        raise FileNotFoundError(f"No snapshot files in {snapshots_dir}")
    df = pd.read_csv(files[-1], low_memory=False)
    return df


def load_clusters(clusters_path: str) -> pd.DataFrame:
    df = pd.read_csv(clusters_path)
    return df[[_VM_COL, "cluster"]].copy()


# ── Core transforms ────────────────────────────────────────────────────────────

def build_daily_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw transaction rows → one row per (machine, date)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df[_DATE_COL], utc=True).dt.normalize().dt.tz_localize(None)
    df[_VM_COL] = df[_VM_COL].astype(int)

    daily = (
        df.groupby([_VM_COL, "date"])
        .agg(
            total_sale=(  _SALE_COL, "sum"),
            n_transac=(   _ID_COL,   "count"),
            qty_sold=(    _QTY_COL,  "sum"),
        )
        .reset_index()
        .sort_values([_VM_COL, "date"])
    )
    return daily


def fill_missing_dates(daily: pd.DataFrame) -> pd.DataFrame:
    """Create a full calendar grid per machine (missing days → 0 demand)."""
    date_min = daily["date"].min()
    date_max = daily["date"].max()
    date_range = pd.date_range(date_min, date_max, freq="D")

    machines = daily[_VM_COL].unique()
    idx = pd.MultiIndex.from_product([machines, date_range], names=[_VM_COL, "date"])

    full = (
        daily.set_index([_VM_COL, "date"])
        .reindex(idx, fill_value=0)
        .reset_index()
        .sort_values([_VM_COL, "date"])
    )
    return full


def add_temporal_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.dayofweek          # 0=Mon, 6=Sun
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    return daily


def add_lag_features(daily: pd.DataFrame, lags=(1, 7, 14)) -> pd.DataFrame:
    daily = daily.sort_values([_VM_COL, "date"]).copy()
    for lag in lags:
        daily[f"lag_{lag}"] = daily.groupby(_VM_COL)[TARGET_COL].shift(lag)
    return daily


def add_rolling_features(daily: pd.DataFrame, windows=(7, 14)) -> pd.DataFrame:
    """Rolling stats shifted by 1 day to prevent leakage."""
    daily = daily.sort_values([_VM_COL, "date"]).copy()
    for w in windows:
        grp = daily.groupby(_VM_COL)[TARGET_COL]
        daily[f"roll_mean_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(w // 2, 1)).mean()
        )
        daily[f"roll_std_{w}"] = grp.transform(
            lambda x: x.shift(1).rolling(w, min_periods=max(w // 2, 1)).std().fillna(0)
        )
    return daily


def merge_clusters(daily: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    return daily.merge(clusters, on=_VM_COL, how="left")


# ── Full pipeline ──────────────────────────────────────────────────────────────

def build_feature_matrix(
    snapshots_dir: str,
    clusters_path: str,
    lags=(1, 7, 14),
    windows=(7, 14),
    fill_dates: bool = True,
) -> pd.DataFrame:
    """
    End-to-end pipeline:
      load snapshot → aggregate daily → fill calendar → temporal → lags →
      rolling → merge cluster labels.
    """
    df_raw = load_latest_snapshot(snapshots_dir)
    clusters = load_clusters(clusters_path)

    daily = build_daily_timeseries(df_raw)
    if fill_dates:
        daily = fill_missing_dates(daily)
    daily = add_temporal_features(daily)
    daily = add_lag_features(daily, lags)
    daily = add_rolling_features(daily, windows)
    daily = merge_clusters(daily, clusters)
    return daily


# ── Train / test split (chronological) ────────────────────────────────────────

def time_split(
    daily: pd.DataFrame,
    test_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    Split by date — training is the first (1-test_frac) of unique dates,
    test is the remainder. Returns (train_df, test_df, cutoff_date).
    """
    dates = sorted(daily["date"].unique())
    cutoff = dates[int(len(dates) * (1 - test_frac))]
    train = daily[daily["date"] < cutoff].copy()
    test = daily[daily["date"] >= cutoff].copy()
    return train, test, cutoff
