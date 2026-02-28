"""Reflexive rough-Hawkes proxy dynamics for CHIMERA-RX."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from chimera_rx_state import StateTensor


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


@dataclass(frozen=True)
class ReflexiveState:
    rough_component: float
    hawkes_intensity: float
    regime_intensity: float
    drift_adjust: float
    vol_adjust: float
    natural_grad_step: float


def update_reflexive_state(
    state: StateTensor,
    hurst: float,
    hawkes_cluster: float,
    prev_intensity: float = 0.0,
) -> ReflexiveState:
    h = _clip(hurst, 0.01, 0.49)
    rough = _clip((0.5 - h) / 0.5, 0.0, 1.0)
    hk = _clip(hawkes_cluster, 0.0, 3.0)

    base = 0.25 + 0.45 * abs(state.behavioral_flow) + 0.30 * state.stress_index
    intensity = _clip(0.55 * _clip(prev_intensity, 0.0, 2.0) + 0.45 * (base + 0.35 * hk), 0.0, 2.0)
    regime_intensity = _clip(
        0.45 * rough
        + 0.45 * _clip(intensity / 1.2, 0.0, 1.0)
        + 0.10 * abs(state.overnight_gap),
        0.0,
        1.0,
    )

    drift_adjust = _clip(0.12 * state.behavioral_flow - 0.08 * state.macro, -0.15, 0.15)
    vol_adjust = _clip(
        0.20 * regime_intensity + 0.10 * abs(state.microstructure_friction) + 0.08 * abs(state.overnight_gap),
        0.0,
        0.55,
    )
    nat_step = _clip(0.02 + 0.04 * rough + 0.03 * abs(state.fii_flow_norm), 0.01, 0.10)

    return ReflexiveState(
        rough_component=float(rough),
        hawkes_intensity=float(intensity),
        regime_intensity=float(regime_intensity),
        drift_adjust=float(drift_adjust),
        vol_adjust=float(vol_adjust),
        natural_grad_step=float(nat_step),
    )


def reflexive_price_adjustment(eq_price: float, reflexive: ReflexiveState) -> float:
    mult = exp(reflexive.drift_adjust) * (1.0 + 0.35 * reflexive.vol_adjust)
    return float(max(0.01, eq_price * _clip(mult, 0.65, 1.45)))

