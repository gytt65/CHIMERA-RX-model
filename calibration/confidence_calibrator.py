#!/usr/bin/env python3
"""
confidence_calibrator.py
========================

Confidence calibration utilities used by NIRV/OMEGA/NOVA pipelines.

The calibrator supports:
1. Isotonic calibration on raw confidence percentages.
2. Temperature scaling on logits/probabilities.
3. Blended calibrated confidence output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    _SK_ISOTONIC = True
except Exception:
    IsotonicRegression = None
    _SK_ISOTONIC = False


def _safe_prob(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _safe_prob(p)
    return np.log(p / (1.0 - p))


@dataclass
class CalibrationStats:
    n_samples: int
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float


class ConfidenceCalibrator:
    """
    Calibrates model confidence to empirical correctness.

    Input confidence should be in percentage [0, 100].
    """

    def __init__(self):
        self._iso = None
        self._temp = 1.0
        self._fitted = False
        self._stats: Optional[CalibrationStats] = None

    @staticmethod
    def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
        p = _safe_prob(p)
        y = np.asarray(y, dtype=float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i < n_bins - 1:
                m = (p >= lo) & (p < hi)
            else:
                m = (p >= lo) & (p <= hi)
            if not np.any(m):
                continue
            conf = float(np.mean(p[m]))
            acc = float(np.mean(y[m]))
            ece += float(np.mean(m)) * abs(conf - acc)
        return float(ece)

    @staticmethod
    def _fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
        """
        Lightweight temperature fit by grid-search on NLL.
        """
        p = _safe_prob(p)
        y = np.asarray(y, dtype=float)
        z = _logit(p)
        best_t = 1.0
        best_nll = float("inf")
        for t in np.linspace(0.5, 3.0, 101):
            p_t = 1.0 / (1.0 + np.exp(-z / t))
            nll = -np.mean(y * np.log(p_t) + (1.0 - y) * np.log(1.0 - p_t))
            if nll < best_nll:
                best_nll = float(nll)
                best_t = float(t)
        return best_t

    def fit(
        self,
        confidence_pct: Iterable[float],
        outcomes: Iterable[int],
        blend_alpha: float = 0.6,
    ) -> CalibrationStats:
        """
        Fit calibrator from confidence percentages and binary outcomes.
        """
        conf = np.asarray(list(confidence_pct), dtype=float)
        y = np.asarray(list(outcomes), dtype=float)
        if conf.size != y.size or conf.size < 20:
            self._fitted = False
            self._stats = CalibrationStats(
                n_samples=int(conf.size),
                brier_before=0.0,
                brier_after=0.0,
                ece_before=0.0,
                ece_after=0.0,
            )
            return self._stats

        p = _safe_prob(conf / 100.0)
        y = np.clip(y, 0.0, 1.0)

        brier_before = float(np.mean((p - y) ** 2))
        ece_before = self._ece(p, y)

        # Temperature scaling
        self._temp = self._fit_temperature(p, y)
        p_temp = 1.0 / (1.0 + np.exp(-_logit(p) / self._temp))

        # Isotonic calibration
        p_iso = p_temp.copy()
        if _SK_ISOTONIC and IsotonicRegression is not None:
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(p_temp, y)
            p_iso = self._iso.predict(p_temp)
        else:
            self._iso = None

        # Blend calibrated and temperature-only to reduce overfitting.
        alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        p_cal = alpha * p_iso + (1.0 - alpha) * p_temp
        p_cal = _safe_prob(p_cal)

        brier_after = float(np.mean((p_cal - y) ** 2))
        ece_after = self._ece(p_cal, y)

        self._fitted = True
        self._stats = CalibrationStats(
            n_samples=int(conf.size),
            brier_before=brier_before,
            brier_after=brier_after,
            ece_before=ece_before,
            ece_after=ece_after,
        )
        return self._stats

    def transform(self, confidence_pct: float) -> float:
        """
        Transform a confidence percentage to calibrated percentage.
        """
        p = float(np.clip(confidence_pct / 100.0, 1e-6, 1.0 - 1e-6))
        if not self._fitted:
            return float(np.clip(confidence_pct, 0.0, 99.9))
        p_t = 1.0 / (1.0 + np.exp(-_logit(np.array([p])) / self._temp))[0]
        if self._iso is not None:
            p_t = float(self._iso.predict([p_t])[0])
        return float(np.clip(p_t * 100.0, 0.0, 99.9))

    def to_dict(self) -> Dict:
        return {
            "fitted": bool(self._fitted),
            "temperature": float(self._temp),
            "stats": None if self._stats is None else self._stats.__dict__,
        }
