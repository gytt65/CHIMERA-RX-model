#!/usr/bin/env python3
"""
hpo_optuna.py
=============

Optional Optuna runner with random-search fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import optuna
    _HAS_OPTUNA = True
except Exception:
    optuna = None
    _HAS_OPTUNA = False


@dataclass
class TrialResult:
    params: Dict
    score: float


class HPORunner:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def run(
        self,
        objective_fn: Callable[[Dict], float],
        search_space: Dict[str, Dict],
        n_trials: int = 50,
        direction: str = "maximize",
    ) -> Dict:
        if _HAS_OPTUNA:
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            study = optuna.create_study(direction=direction, sampler=sampler)

            def _obj(trial):
                params = {}
                for k, spec in search_space.items():
                    t = spec.get("type", "float")
                    lo = spec.get("low")
                    hi = spec.get("high")
                    if t == "int":
                        params[k] = trial.suggest_int(k, int(lo), int(hi))
                    elif t == "cat":
                        params[k] = trial.suggest_categorical(k, list(spec.get("choices", [])))
                    else:
                        params[k] = trial.suggest_float(k, float(lo), float(hi))
                return float(objective_fn(params))

            study.optimize(_obj, n_trials=int(max(n_trials, 1)))
            return {
                "backend": "optuna",
                "best_params": dict(study.best_params),
                "best_score": float(study.best_value),
                "n_trials": int(len(study.trials)),
            }

        # Random search fallback
        best: Optional[TrialResult] = None
        history: List[TrialResult] = []
        for _ in range(int(max(n_trials, 1))):
            params = {}
            for k, spec in search_space.items():
                t = spec.get("type", "float")
                if t == "int":
                    params[k] = int(self.rng.integers(int(spec["low"]), int(spec["high"]) + 1))
                elif t == "cat":
                    choices = list(spec.get("choices", []))
                    params[k] = choices[int(self.rng.integers(0, max(len(choices), 1)))] if choices else None
                else:
                    params[k] = float(self.rng.uniform(float(spec["low"]), float(spec["high"])))
            score = float(objective_fn(params))
            tr = TrialResult(params=params, score=score)
            history.append(tr)
            if best is None:
                best = tr
            else:
                if direction == "maximize" and tr.score > best.score:
                    best = tr
                if direction != "maximize" and tr.score < best.score:
                    best = tr

        return {
            "backend": "random_search",
            "best_params": dict(best.params if best else {}),
            "best_score": float(best.score if best else 0.0),
            "n_trials": int(len(history)),
        }
