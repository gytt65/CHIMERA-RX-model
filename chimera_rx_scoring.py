"""CHIMERA-RX scoring utilities (additive, model-isolated)."""

from __future__ import annotations

from dataclasses import dataclass


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clip10(x: float) -> float:
    return max(0.0, min(10.0, float(x)))


@dataclass(frozen=True)
class ModelQualityScore:
    """Research-grade quality score on [0, 10]."""

    A_t: float
    R_t: float
    N_t: float
    G_t: float
    S_t: float
    Q_t: float


def compute_quality_score(
    accuracy: float,
    robustness: float,
    no_arb_integrity: float,
    regime_lead: float,
    selective_accuracy: float,
) -> ModelQualityScore:
    """
    Compute quality score using fixed CHIMERA-RX weights.

    Inputs are expected in [0, 1]. Values are clipped for stability.
    """
    A_t = _clip01(accuracy)
    R_t = _clip01(robustness)
    N_t = _clip01(no_arb_integrity)
    G_t = _clip01(regime_lead)
    S_t = _clip01(selective_accuracy)

    q = 10.0 * (
        0.30 * A_t
        + 0.20 * R_t
        + 0.20 * N_t
        + 0.15 * G_t
        + 0.15 * S_t
    )
    return ModelQualityScore(A_t=A_t, R_t=R_t, N_t=N_t, G_t=G_t, S_t=S_t, Q_t=_clip10(q))
