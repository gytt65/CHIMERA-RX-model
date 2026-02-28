#!/usr/bin/env python3
"""
walkforward_runner.py
=====================

Purged walk-forward splitting and runner helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass
class WalkForwardConfig:
    n_splits: int = 6
    purge_days: int = 2
    min_train_rows: int = 200


def _require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for walk-forward runner")


def purged_walkforward_splits(
    df: "pd.DataFrame",
    ts_col: str = "ts",
    config: WalkForwardConfig = WalkForwardConfig(),
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    _require_pandas()
    x = df.copy()
    x[ts_col] = pd.to_datetime(x[ts_col], errors="coerce")
    x = x.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    if len(x) < config.min_train_rows:
        return iter(())

    idx = np.arange(len(x))
    chunks = np.array_split(idx, config.n_splits)
    for i in range(1, len(chunks)):
        test_idx = chunks[i]
        if test_idx.size == 0:
            continue
        test_start = x.loc[test_idx[0], ts_col]
        purge_cutoff = test_start - pd.Timedelta(days=int(config.purge_days))
        train_idx = idx[x[ts_col].values < purge_cutoff.to_datetime64()]
        if train_idx.size < config.min_train_rows:
            continue
        yield train_idx, test_idx


def run_walkforward(
    df: "pd.DataFrame",
    feature_cols: List[str],
    target_col: str,
    model_factory: Callable[[], object],
    config: WalkForwardConfig = WalkForwardConfig(),
) -> Dict:
    _require_pandas()
    rows = []
    for train_idx, test_idx in purged_walkforward_splits(df, config=config):
        model = model_factory()
        X_tr = df.iloc[train_idx][feature_cols].to_numpy(dtype=float)
        y_tr = df.iloc[train_idx][target_col].to_numpy(dtype=float)
        X_te = df.iloc[test_idx][feature_cols].to_numpy(dtype=float)
        y_te = df.iloc[test_idx][target_col].to_numpy(dtype=float)
        if hasattr(model, "fit"):
            model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te) if hasattr(model, "predict") else np.zeros_like(y_te)
        mae = float(np.mean(np.abs(y_te - y_hat)))
        rmse = float(np.sqrt(np.mean((y_te - y_hat) ** 2)))
        rows.append({"n_train": int(len(train_idx)), "n_test": int(len(test_idx)), "mae": mae, "rmse": rmse})

    if not rows:
        return {"n_folds": 0, "mae": None, "rmse": None, "folds": []}
    return {
        "n_folds": len(rows),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "rmse": float(np.mean([r["rmse"] for r in rows])),
        "folds": rows,
    }
