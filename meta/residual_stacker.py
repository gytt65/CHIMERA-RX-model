#!/usr/bin/env python3
"""
residual_stacker.py
===================

Regime-aware residual correction model.

Primary model:
- LightGBM regressor (if installed)

Fallbacks:
- sklearn GradientBoostingRegressor
- numpy ridge (closed form) when sklearn is unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    lgb = None
    _HAS_LGBM = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    _HAS_SK = True
except Exception:
    GradientBoostingRegressor = None
    _HAS_SK = False


DEFAULT_FEATURES = [
    "log_moneyness",
    "dte",
    "iv",
    "vix",
    "spread_ratio",
    "volume_oi_ratio",
    "regime_bull_low",
    "regime_bear_high",
    "regime_sideways",
    "regime_bull_high",
]


@dataclass
class ResidualSample:
    features: Dict[str, float]
    residual: float


class ResidualStacker:
    def __init__(self, feature_names: Optional[Sequence[str]] = None):
        self.feature_names = list(feature_names or DEFAULT_FEATURES)
        self._model = None
        self._ridge_w = None
        self._ridge_b = 0.0
        self._fitted = False
        self.backend = "none"

    def _vec(self, features: Dict[str, float]) -> np.ndarray:
        arr = []
        for name in self.feature_names:
            try:
                arr.append(float(features.get(name, 0.0)))
            except Exception:
                arr.append(0.0)
        return np.asarray(arr, dtype=float)

    @property
    def is_fitted(self) -> bool:
        return bool(self._fitted)

    def fit(self, samples: Iterable[ResidualSample]) -> Dict:
        rows = list(samples)
        if len(rows) < 30:
            self._fitted = False
            return {"fitted": False, "n": len(rows), "backend": "none"}

        X = np.vstack([self._vec(s.features) for s in rows])
        y = np.asarray([float(s.residual) for s in rows], dtype=float)

        if _HAS_LGBM:
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            model.fit(X, y)
            self._model = model
            self.backend = "lightgbm"
            self._fitted = True
            return {"fitted": True, "n": len(rows), "backend": self.backend}

        if _HAS_SK and GradientBoostingRegressor is not None:
            model = GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=3,
                random_state=42,
            )
            model.fit(X, y)
            self._model = model
            self.backend = "sklearn_gbr"
            self._fitted = True
            return {"fitted": True, "n": len(rows), "backend": self.backend}

        # Numpy ridge fallback
        lam = 1e-3
        XtX = X.T @ X
        Xty = X.T @ y
        self._ridge_w = np.linalg.solve(XtX + lam * np.eye(X.shape[1]), Xty)
        self._ridge_b = float(np.mean(y - X @ self._ridge_w))
        self.backend = "numpy_ridge"
        self._fitted = True
        return {"fitted": True, "n": len(rows), "backend": self.backend}

    def predict(self, features: Dict[str, float]) -> float:
        if not self._fitted:
            return 0.0
        x = self._vec(features).reshape(1, -1)
        if self._model is not None:
            try:
                return float(self._model.predict(x)[0])
            except Exception:
                return 0.0
        if self._ridge_w is None:
            return 0.0
        return float(x[0] @ self._ridge_w + self._ridge_b)
