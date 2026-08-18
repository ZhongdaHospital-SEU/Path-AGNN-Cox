"""Data loading and preprocessing for benchmark cohorts.

Expected processed format (CSV): rows = samples, columns =
    [sample_id, OS_time, OS_event, <gene columns...>]
External GEO cohorts are expected in the same format after R preprocessing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_survival_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("sample_id", "OS_time", "OS_event"):
        assert col in df.columns, f"missing required column {col} in {path}"
    return df


def split_features(df: pd.DataFrame, feature_cols: list[str] | None = None
                   ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (X, time, event) with gene columns only."""
    meta = ["sample_id", "OS_time", "OS_event"]
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in meta]
    X = df[feature_cols]
    time = df["OS_time"].to_numpy(dtype=float)
    event = df["OS_event"].to_numpy(dtype=int)
    return X, time, event


def intersect_genes(df: pd.DataFrame, gene_order: np.ndarray,
                    meta: tuple = ("sample_id", "OS_time", "OS_event")
                    ) -> pd.DataFrame:
    """Keep only genes present in the pathway gene order (in that order)."""
    cols = list(gene_order)
    cols = [c for c in cols if c in df.columns]
    return df[list(meta) + cols].copy()


def standardize(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None
                ) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    """Z-score features using train statistics; return (Xtr, Xte)."""
    mu, sd = X_train.mean(), X_train.std(ddof=0).replace(0, 1.0)
    Xtr = (X_train - mu) / sd
    if X_test is not None:
        Xte = (X_test - mu) / sd
        return Xtr, Xte
    return Xtr, None


def make_structured_survival(time: np.ndarray, event: np.ndarray):
    """sksurv-compatible structured array: [('event', bool), ('time', float)]."""
    y = np.zeros(time.shape[0], dtype=[("event", bool), ("time", float)])
    y["event"] = event.astype(bool)
    y["time"] = time.astype(float)
    return y