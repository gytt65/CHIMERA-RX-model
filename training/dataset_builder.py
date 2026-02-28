#!/usr/bin/env python3
"""
dataset_builder.py
==================

Builds leak-safe training datasets from chain snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass
class DatasetBuildConfig:
    horizon_days: int = 1
    purge_days: int = 1
    spread_floor: float = 1e-4


def _require_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for dataset building")


def build_dataset_from_rows(
    rows: Iterable[Dict],
    config: Optional[DatasetBuildConfig] = None,
) -> "pd.DataFrame":
    """
    Expected row keys:
    - ts, symbol, option_type, strike, spot, market_price
    - bid, ask, iv, vix, dte, regime
    - model_price
    """
    _require_pandas()
    cfg = config or DatasetBuildConfig()
    df = pd.DataFrame(list(rows)).copy()
    if df.empty:
        return df

    for col in ("ts", "symbol", "option_type", "strike", "spot", "market_price", "dte"):
        if col not in df.columns:
            raise ValueError(f"missing required column: {col}")

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values(["symbol", "option_type", "strike", "ts"]).reset_index(drop=True)

    # Features
    df["log_moneyness"] = np.log(np.maximum(df["strike"].astype(float), 1e-6) / np.maximum(df["spot"].astype(float), 1e-6))
    df["spread_ratio"] = (
        (pd.to_numeric(df.get("ask"), errors="coerce") - pd.to_numeric(df.get("bid"), errors="coerce"))
        / np.maximum(pd.to_numeric(df["market_price"], errors="coerce"), cfg.spread_floor)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["volume_oi_ratio"] = pd.to_numeric(df.get("volume_oi_ratio", 0.0), errors="coerce").fillna(0.0)
    df["iv"] = pd.to_numeric(df.get("iv", 0.0), errors="coerce").fillna(0.0)
    df["vix"] = pd.to_numeric(df.get("vix", 0.0), errors="coerce").fillna(0.0)
    df["dte"] = pd.to_numeric(df["dte"], errors="coerce").fillna(0).astype(int)
    df["regime"] = df.get("regime", "Unknown").astype(str)
    for r in ("Bull-Low Vol", "Bear-High Vol", "Sideways", "Bull-High Vol"):
        c = "regime_" + r.lower().replace(" ", "_").replace("-", "_")
        df[c] = (df["regime"] == r).astype(float)

    # Targets (no lookahead leakage: shifted inside contract group)
    gcols = ["symbol", "option_type", "strike"]
    df["market_price_fwd"] = df.groupby(gcols)["market_price"].shift(-cfg.horizon_days)
    df["ret_1d"] = (df["market_price_fwd"] - df["market_price"]) / np.maximum(df["market_price"], cfg.spread_floor)
    df["target_direction"] = (df["ret_1d"] > 0).astype(float)
    if "model_price" in df.columns:
        df["target_residual"] = df["market_price_fwd"] - df["model_price"]
    else:
        df["target_residual"] = df["market_price_fwd"] - df["market_price"]

    # Purge trailing rows that cannot be labeled
    df = df.dropna(subset=["market_price_fwd"]).reset_index(drop=True)
    return df
