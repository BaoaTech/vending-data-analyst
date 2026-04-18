"""
LSTM time-series model for per-machine daily demand forecasting.
Sequences are pooled across all machines (global model approach).
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.models.metrics import compute_metrics


SEQ_LEN = 14  # look-back window in days


# ── Sequence construction ──────────────────────────────────────────────────────

def create_sequences(
    daily: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "total_sale",
    seq_len: int = SEQ_LEN,
    scaler: MinMaxScaler = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Build (X, y) arrays pooled across all machines.

    Parameters
    ----------
    daily       : DataFrame with vm_control, date, feature_cols, target_col.
    feature_cols: Columns used as per-timestep inputs (target_col must be first).
    scaler      : Pre-fitted MinMaxScaler. Fitted here when fit_scaler=True.
    fit_scaler  : Fit a new scaler on this data (use True for training, False for test).

    Returns
    -------
    X      : shape (n_sequences, seq_len, n_features)
    y      : shape (n_sequences,)  — target in scaled space
    scaler : fitted MinMaxScaler
    """
    all_cols = list(dict.fromkeys([target_col] + feature_cols))   # deduplicated, target first
    data = daily.sort_values(["vm_control", "date"]).copy()

    if fit_scaler or scaler is None:
        scaler = MinMaxScaler()
        data[all_cols] = scaler.fit_transform(data[all_cols])
    else:
        data[all_cols] = scaler.transform(data[all_cols])

    X_list, y_list = [], []
    for _, grp in data.groupby("vm_control"):
        vals = grp[all_cols].values       # shape (days, n_cols)
        for i in range(seq_len, len(vals)):
            X_list.append(vals[i - seq_len: i, :])    # all cols as input features
            y_list.append(vals[i, 0])                  # target_col = first column

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), scaler


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(input_shape: Tuple[int, int]):
    """LSTM → LSTM → Dense regression head."""
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.models import Sequential

    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
    )
    return model


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
):
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5, verbose=1),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


# ── Evaluation ─────────────────────────────────────────────────────────────────

def inverse_target(values_scaled: np.ndarray, scaler: MinMaxScaler, n_cols: int) -> np.ndarray:
    """Inverse-transform a 1-D array that corresponds to column 0 of the scaler."""
    dummy = np.zeros((len(values_scaled), n_cols), dtype=np.float32)
    dummy[:, 0] = values_scaled
    return scaler.inverse_transform(dummy)[:, 0]


def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    n_cols: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Predict on X_test, inverse-transform both pred and truth, compute full metrics.

    Returns
    -------
    y_pred_real : predictions in original $ scale
    y_true_real : actuals in original $ scale
    metrics     : dict with MAE, RMSE, MdAE, R², SMAPE, Bias
    """
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred_real = np.clip(inverse_target(y_pred_scaled, scaler, n_cols), 0, None)
    y_true_real = inverse_target(y_test, scaler, n_cols)

    return y_pred_real, y_true_real, compute_metrics(y_true_real, y_pred_real)


# ── Persistence ────────────────────────────────────────────────────────────────

def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model_from_path(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)
