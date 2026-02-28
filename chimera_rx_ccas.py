"""Causal-compression abstention system for CHIMERA-RX."""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi
from statistics import mean
from typing import Mapping, Sequence


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _var(vals: Sequence[float]) -> float:
    if not vals:
        return 0.0
    mu = mean(vals)
    return sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)


@dataclass(frozen=True)
class FrontierPoint:
    tau: float
    coverage: float
    accuracy: float
    n_actions: int


@dataclass(frozen=True)
class CompressionDiagnostics:
    L0: float
    LM: float
    CR_t: float
    tau: float
    action: bool
    abstain: bool
    coverage: float
    accuracy: float
    n_samples: int


def mdl_lengths(errors: Sequence[float], k: int = 6) -> tuple[float, float]:
    vals = [abs(float(e)) for e in errors if e is not None]
    if not vals:
        vals = [0.15]
    n = len(vals)
    mse = max(mean([v * v for v in vals]), 1e-9)
    nll_model = 0.5 * n * (log(2.0 * pi * mse) + 1.0)
    lm = nll_model + 0.5 * k * log(max(n, 2))

    # Null description: weaker structure with a broader baseline error scale.
    mad = max(mean(vals), 1e-9)
    null_scale = max(0.12, mad * 1.25)
    baseline_var = max(null_scale * null_scale, mse)
    nll_null = 0.5 * n * (log(2.0 * pi * baseline_var) + 1.0)
    l0 = nll_null + 0.5 * max(k * 2, 8) * log(max(n, 2))
    return (float(max(l0, 1e-9)), float(max(lm, 1e-9)))


def build_frontier(
    samples: Sequence[Mapping[str, float]] | None,
    thresholds: Sequence[float] | None = None,
) -> list[FrontierPoint]:
    rows = list(samples or [])
    if not rows:
        return [
            FrontierPoint(tau=1.2, coverage=0.70, accuracy=0.78, n_actions=0),
            FrontierPoint(tau=1.5, coverage=0.55, accuracy=0.82, n_actions=0),
            FrontierPoint(tau=2.0, coverage=0.40, accuracy=0.85, n_actions=0),
        ]

    cr_vals = sorted([float(r.get("cr", 1.0)) for r in rows])
    if thresholds is None:
        q = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85, 0.95]
        thresholds = [cr_vals[min(int((len(cr_vals) - 1) * t), len(cr_vals) - 1)] for t in q]
        thresholds = sorted(set([round(float(t), 3) for t in thresholds]))
        if not thresholds:
            thresholds = [1.2, 1.5, 2.0]

    frontier: list[FrontierPoint] = []
    n = max(len(rows), 1)
    for tau in sorted(float(t) for t in thresholds):
        actions = [r for r in rows if float(r.get("cr", 0.0)) > tau]
        cov = len(actions) / n
        if actions:
            acc = mean([float(_clip(r.get("correct", 0.0), 0.0, 1.0)) for r in actions])
        else:
            acc = 0.0
        frontier.append(
            FrontierPoint(
                tau=float(tau),
                coverage=float(cov),
                accuracy=float(acc),
                n_actions=len(actions),
            )
        )
    return frontier


def calibrate_tau(
    samples: Sequence[Mapping[str, float]] | None,
    target_accuracy: float = 0.80,
    coverage_bounds: tuple[float, float] = (0.35, 0.55),
    default_tau: float = 1.5,
) -> tuple[float, FrontierPoint, list[FrontierPoint]]:
    frontier = build_frontier(samples)
    lo, hi = float(coverage_bounds[0]), float(coverage_bounds[1])
    mid = 0.5 * (lo + hi)

    feasible = [
        p for p in frontier
        if p.coverage >= lo and p.coverage <= hi and p.accuracy >= float(target_accuracy)
    ]
    if feasible:
        feasible = sorted(
            feasible,
            key=lambda p: (p.accuracy, -abs(p.coverage - mid), -abs(p.tau - default_tau)),
            reverse=True,
        )
        best = feasible[0]
        return (float(best.tau), best, frontier)

    def _penalty(p: FrontierPoint) -> float:
        cov_pen = 0.0
        if p.coverage < lo:
            cov_pen = lo - p.coverage
        elif p.coverage > hi:
            cov_pen = p.coverage - hi
        acc_pen = max(0.0, float(target_accuracy) - p.accuracy)
        tau_pen = 0.05 * abs(p.tau - default_tau)
        return cov_pen + acc_pen + tau_pen

    best = min(frontier, key=_penalty)
    return (float(best.tau), best, frontier)


def compute_compression_diagnostics(
    errors: Sequence[float],
    samples: Sequence[Mapping[str, float]] | None,
    current_correct: float | None = None,
    target_accuracy: float = 0.80,
    coverage_bounds: tuple[float, float] = (0.35, 0.55),
    k: int = 6,
) -> tuple[CompressionDiagnostics, list[dict], list[FrontierPoint]]:
    l0, lm = mdl_lengths(errors=errors, k=k)
    cr = float(l0 / max(lm, 1e-9))

    hist = [dict(x) for x in (samples or [])]
    if current_correct is None:
        current_correct = 1.0 if abs(float(errors[0] if errors else 0.0)) <= 0.12 else 0.0
    hist.append({"cr": float(cr), "correct": float(_clip(current_correct, 0.0, 1.0))})
    if len(hist) > 256:
        hist = hist[-256:]

    tau, best_point, frontier = calibrate_tau(
        samples=hist,
        target_accuracy=target_accuracy,
        coverage_bounds=coverage_bounds,
    )
    action = bool(cr > tau)
    diag = CompressionDiagnostics(
        L0=float(l0),
        LM=float(lm),
        CR_t=float(cr),
        tau=float(tau),
        action=action,
        abstain=not action,
        coverage=float(best_point.coverage),
        accuracy=float(best_point.accuracy),
        n_samples=len(hist),
    )
    return (diag, hist, frontier)
