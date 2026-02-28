"""Phase-transition diagnostics for CHIMERA-RX."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from statistics import mean
from typing import Dict, Mapping, Sequence

from chimera_rx_reflexive import ReflexiveState
from chimera_rx_state import StateTensor


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _std(vals: Sequence[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return sqrt(sum((v - mu) ** 2 for v in vals) / max(len(vals) - 1, 1))


@dataclass(frozen=True)
class PTDSignals:
    curvature: float
    critical_slowing: float
    susceptibility: float
    symmetry_break: float
    score: float
    alert: str


def compute_ptd_signals(
    state: StateTensor,
    reflexive: ReflexiveState,
    iv_residual: float,
    timeline: Sequence[Mapping[str, float | str]] | None = None,
) -> PTDSignals:
    hist = list(timeline or [])
    prev_scores = [float(x.get("score", 0.0)) for x in hist if "score" in x]
    prev_last = prev_scores[-1] if prev_scores else 0.0
    score_delta = abs(prev_last - (prev_scores[-2] if len(prev_scores) > 1 else prev_last))
    score_std = _std(prev_scores[-8:]) if prev_scores else 0.0

    curvature = _clip(
        0.35 * abs(state.behavioral_flow)
        + 0.30 * abs(state.microstructure_friction)
        + 0.20 * reflexive.regime_intensity
        + 0.15 * abs(iv_residual),
        0.0,
        1.0,
    )
    critical_slowing = _clip(
        0.40 * _clip(reflexive.hawkes_intensity / 2.0, 0.0, 1.0)
        + 0.35 * abs(state.vix_norm)
        + 0.15 * _clip(1.0 - score_delta, 0.0, 1.0)
        + 0.10 * _clip(score_std, 0.0, 1.0),
        0.0,
        1.0,
    )
    susceptibility = _clip(
        0.50 * abs(state.fii_flow_norm)
        + 0.30 * abs(state.overnight_gap)
        + 0.20 * reflexive.regime_intensity,
        0.0,
        1.0,
    )
    symmetry_break = _clip(
        0.50 * abs(state.behavioral_flow - state.macro)
        + 0.30 * abs(state.pcr_norm - state.fii_flow_norm)
        + 0.20 * abs(state.microstructure_friction),
        0.0,
        1.0,
    )
    score = _clip(
        0.33 * curvature + 0.25 * critical_slowing + 0.22 * susceptibility + 0.20 * symmetry_break,
        0.0,
        1.0,
    )

    alert = "NORMAL"
    if score > 0.52:
        alert = "ELEVATED"
    if score > 0.72:
        alert = "STRESS"

    return PTDSignals(
        curvature=float(curvature),
        critical_slowing=float(critical_slowing),
        susceptibility=float(susceptibility),
        symmetry_break=float(symmetry_break),
        score=float(score),
        alert=alert,
    )


def update_ptd_timeline(
    timeline: Sequence[Mapping[str, float | str]] | None,
    signals: PTDSignals,
    max_len: int = 64,
    ts: str | None = None,
) -> list[Dict[str, float | str]]:
    row = {
        "ts": ts or datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "score": float(signals.score),
        "alert": str(signals.alert),
        "curvature": float(signals.curvature),
        "critical_slowing": float(signals.critical_slowing),
        "susceptibility": float(signals.susceptibility),
        "symmetry_break": float(signals.symmetry_break),
    }
    out = list(timeline or [])
    out.append(row)
    if len(out) > max_len:
        out = out[-max_len:]
    return out
