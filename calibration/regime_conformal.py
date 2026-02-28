#!/usr/bin/env python3
"""
regime_conformal.py
===================

Split-conformal intervals keyed by (regime, moneyness bucket, dte bucket).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _m_bucket(log_m: float) -> str:
    x = float(log_m)
    if x < -0.05:
        return "PUT_WING"
    if x > 0.05:
        return "CALL_WING"
    return "ATM"


def _dte_bucket(dte: int) -> str:
    d = int(max(dte, 0))
    if d <= 3:
        return "D0_3"
    if d <= 7:
        return "D4_7"
    if d <= 21:
        return "D8_21"
    return "D22P"


def bucket_key(regime: str, log_moneyness: float, dte: int) -> Tuple[str, str, str]:
    return (str(regime or "Unknown"), _m_bucket(log_moneyness), _dte_bucket(dte))


@dataclass
class ConformalRecord:
    y_true: float
    y_pred: float
    regime: str
    log_moneyness: float
    dte: int


class RegimeConformalIntervals:
    """
    Stores absolute residual quantiles by bucket and predicts intervals.
    """

    def __init__(self, alpha: float = 0.10, min_bucket_samples: int = 25):
        self.alpha = float(np.clip(alpha, 0.01, 0.40))
        self.min_bucket_samples = int(max(min_bucket_samples, 5))
        self._q_global = 0.0
        self._q_by_bucket: Dict[Tuple[str, str, str], float] = {}
        self._n_by_bucket: Dict[Tuple[str, str, str], int] = {}

    def fit(self, records: Iterable[ConformalRecord]) -> Dict:
        rows = list(records)
        if not rows:
            self._q_global = 0.0
            self._q_by_bucket = {}
            self._n_by_bucket = {}
            return {"n": 0, "q_global": 0.0, "n_buckets": 0}

        abs_res = np.array([abs(float(r.y_true) - float(r.y_pred)) for r in rows], dtype=float)
        self._q_global = float(np.quantile(abs_res, 1.0 - self.alpha))

        grouped: Dict[Tuple[str, str, str], List[float]] = {}
        for r in rows:
            key = bucket_key(r.regime, r.log_moneyness, r.dte)
            grouped.setdefault(key, []).append(abs(float(r.y_true) - float(r.y_pred)))

        self._q_by_bucket = {}
        self._n_by_bucket = {}
        for k, vals in grouped.items():
            n = len(vals)
            self._n_by_bucket[k] = int(n)
            if n >= self.min_bucket_samples:
                self._q_by_bucket[k] = float(np.quantile(np.asarray(vals, dtype=float), 1.0 - self.alpha))

        return {
            "n": int(len(rows)),
            "q_global": float(self._q_global),
            "n_buckets": int(len(self._q_by_bucket)),
        }

    def interval(
        self,
        y_pred: float,
        regime: str,
        log_moneyness: float,
        dte: int,
        scale: float = 1.0,
    ) -> Tuple[float, float, float]:
        key = bucket_key(regime, log_moneyness, dte)
        q = float(self._q_by_bucket.get(key, self._q_global))
        q = max(0.0, q * float(max(scale, 0.0)))
        y = float(y_pred)
        lo = max(0.01, y - q)
        hi = max(lo, y + q)
        return lo, hi, q

    def to_dict(self) -> Dict:
        return {
            "alpha": float(self.alpha),
            "min_bucket_samples": int(self.min_bucket_samples),
            "q_global": float(self._q_global),
            "q_by_bucket": {str(k): float(v) for k, v in self._q_by_bucket.items()},
            "n_by_bucket": {str(k): int(v) for k, v in self._n_by_bucket.items()},
        }
