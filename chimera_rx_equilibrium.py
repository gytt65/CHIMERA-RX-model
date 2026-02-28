"""Equilibrium solver for CHIMERA-RX."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log, sqrt
from typing import Iterable, List, Sequence, Tuple

from chimera_rx_state import StateTensor


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _normalize_simplex(v: Sequence[float], eps: float = 1e-12) -> Tuple[float, float, float]:
    x = [max(float(t), eps) for t in v]
    s = sum(x)
    if s <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (x[0] / s, x[1] / s, x[2] / s)


def _hellinger_fisher_sq(q: Sequence[float], p: Sequence[float]) -> float:
    qn = _normalize_simplex(q)
    pn = _normalize_simplex(p)
    return float(4.0 * sum((sqrt(qn[i]) - sqrt(pn[i])) ** 2 for i in range(3)))


def _kl(q: Sequence[float], p: Sequence[float], eps: float = 1e-12) -> float:
    qn = _normalize_simplex(q, eps)
    pn = _normalize_simplex(p, eps)
    return float(sum(qn[i] * log((qn[i] + eps) / (pn[i] + eps)) for i in range(3)))


def _state_underlyings(spot: float, sigma: float, T: float) -> Tuple[float, float, float]:
    amp = max(0.05, 1.75 * max(sigma, 1e-4) * sqrt(max(T, 1e-6)))
    drift = 0.0
    s_dn = max(spot * exp(drift - amp), 1e-6)
    s_md = max(spot * exp(drift), 1e-6)
    s_up = max(spot * exp(drift + amp), 1e-6)
    return (s_dn, s_md, s_up)


def discrete_call_prices(
    q: Sequence[float],
    spot: float,
    sigma: float,
    T: float,
    r: float,
    strike_grid: Sequence[float],
) -> List[float]:
    qn = _normalize_simplex(q)
    states = _state_underlyings(spot=spot, sigma=sigma, T=T)
    disc = exp(-max(r, -0.02) * max(T, 1e-6))
    out: List[float] = []
    for k in strike_grid:
        c = 0.0
        for i in range(3):
            c += qn[i] * max(states[i] - k, 0.0)
        out.append(float(disc * c))
    return out


def no_arb_penalty(
    q: Sequence[float],
    spot: float,
    sigma: float,
    T: float,
    r: float,
) -> float:
    qn = _normalize_simplex(q)
    k1, k2, k3 = spot * 0.95, spot, spot * 1.05
    c1, c2, c3 = discrete_call_prices(qn, spot=spot, sigma=sigma, T=T, r=r, strike_grid=(k1, k2, k3))
    p = 0.0
    p += max(0.0, c2 - c1)
    p += max(0.0, c3 - c2)
    second_diff = c1 - 2.0 * c2 + c3
    p += max(0.0, -second_diff)
    p += max(0.0, -c1) + max(0.0, -c2) + max(0.0, -c3)
    # Stability companion term: discourage extreme concentration that destabilizes
    # cross-strike diagnostics in small-sample chains.
    concentration = max(qn) - min(qn)
    p += max(0.0, concentration - 0.85) * 0.05
    return float(p / max(spot, 1.0))


def _no_arb_grad_fd(
    q: Sequence[float],
    spot: float,
    sigma: float,
    T: float,
    r: float,
    eps: float = 1e-5,
) -> Tuple[float, float, float]:
    base = _normalize_simplex(q)
    grad = []
    for j in range(3):
        qp = list(base)
        qm = list(base)
        qp[j] += eps
        qm[j] -= eps
        qp = list(_normalize_simplex(qp))
        qm = list(_normalize_simplex(qm))
        fp = no_arb_penalty(qp, spot=spot, sigma=sigma, T=T, r=r)
        fm = no_arb_penalty(qm, spot=spot, sigma=sigma, T=T, r=r)
        grad.append((fp - fm) / (2.0 * eps))
    return (float(grad[0]), float(grad[1]), float(grad[2]))


def _belief_probs_from_skew(skew: float) -> Tuple[float, float, float]:
    s = _clip(skew, -1.0, 1.0)
    down = _clip(0.33 - 0.19 * s, 0.05, 0.90)
    mid = _clip(0.34 - 0.04 * abs(s), 0.05, 0.90)
    up = max(1.0 - down - mid, 0.05)
    return _normalize_simplex((down, mid, up))


@dataclass(frozen=True)
class AgentBelief:
    name: str
    weight: float
    probs: Tuple[float, float, float]
    skew_proxy: float = 0.0


@dataclass(frozen=True)
class EquilibriumArtifacts:
    q_star: Tuple[float, float, float]
    objective: float
    objective_trace: Tuple[float, ...]
    iterations: int
    converged: bool
    distance_term: float
    entropy_term: float
    no_arb_term: float
    beliefs: Tuple[AgentBelief, ...] = field(default_factory=tuple)


def build_agent_beliefs(state: StateTensor, option_type: str = "CE") -> Tuple[AgentBelief, ...]:
    side = 1.0 if str(option_type).upper() in ("CE", "CALL", "C") else -1.0

    retail_skew = _clip(side * (0.55 * state.behavioral_flow + 0.30 * state.overnight_gap), -1.0, 1.0)
    inst_skew = _clip(-0.65 * side * state.stress_index - 0.20 * side * state.behavioral_flow, -1.0, 1.0)
    mm_skew = _clip(0.15 * side * state.gex_norm, -1.0, 1.0)
    fii_skew = _clip(0.75 * side * state.fii_flow_norm, -1.0, 1.0)
    arb_skew = _clip(0.10 * side * (state.macro - state.microstructure_friction), -1.0, 1.0)

    weights = {
        "retail": _clip(0.23 + 0.07 * abs(state.behavioral_flow), 0.10, 0.38),
        "institutional": _clip(0.18 + 0.09 * max(state.vix_norm, 0.0), 0.10, 0.36),
        "market_maker": 0.26,
        "fii": _clip(0.14 + 0.10 * abs(state.fii_flow_norm), 0.08, 0.34),
        "arbitrageur": 0.16,
    }
    s = sum(weights.values())
    for k in list(weights.keys()):
        weights[k] = weights[k] / max(s, 1e-9)

    beliefs = (
        AgentBelief("retail", weights["retail"], _belief_probs_from_skew(retail_skew), retail_skew),
        AgentBelief("institutional", weights["institutional"], _belief_probs_from_skew(inst_skew), inst_skew),
        AgentBelief("market_maker", weights["market_maker"], _belief_probs_from_skew(mm_skew), mm_skew),
        AgentBelief("fii", weights["fii"], _belief_probs_from_skew(fii_skew), fii_skew),
        AgentBelief("arbitrageur", weights["arbitrageur"], _belief_probs_from_skew(arb_skew), arb_skew),
    )
    return beliefs


def _objective_terms(
    q: Sequence[float],
    beliefs: Iterable[AgentBelief],
    q_prev: Sequence[float],
    lambda_ent: float,
    lambda_arb: float,
    spot: float,
    sigma: float,
    T: float,
    r: float,
    q_ema: Sequence[float] | None = None,
    lambda_drift: float = 0.0,
    shrinkage_alpha: float = 0.0,
) -> Tuple[float, float, float, float, float, float]:
    dist = 0.0
    for b in beliefs:
        dist += float(b.weight) * _hellinger_fisher_sq(q, b.probs)
    ent = _kl(q, q_prev)
    arb = no_arb_penalty(q, spot=spot, sigma=sigma, T=T, r=r)

    # Drift penalty: penalise deviation from recent EMA trend
    drift = 0.0
    if q_ema is not None and lambda_drift > 0:
        drift = sum((q[i] - q_ema[i]) ** 2 for i in range(3))

    # Bayesian shrinkage toward uniform when beliefs are dispersed
    shrink = 0.0
    if shrinkage_alpha > 0:
        uniform = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        shrink = sum((q[i] - uniform[i]) ** 2 for i in range(3))

    total = dist + lambda_ent * ent + lambda_arb * arb + lambda_drift * drift + shrinkage_alpha * shrink
    return float(total), float(dist), float(ent), float(arb), float(drift), float(shrink)


def solve_equilibrium(
    beliefs: Sequence[AgentBelief],
    q_prev: Sequence[float] | None,
    spot: float,
    sigma: float,
    T: float,
    r: float,
    lambda_ent: float = 0.15,
    lambda_arb: float = 2.0,
    lambda_drift: float = 0.08,
    lr: float = 0.12,
    max_iter: int = 120,
    tol: float = 1e-7,
    q_ema: Sequence[float] | None = None,
    ema_decay: float = 0.85,
) -> EquilibriumArtifacts:
    if not beliefs:
        beliefs = [AgentBelief("neutral", 1.0, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))]

    if q_prev is None:
        q0 = [0.0, 0.0, 0.0]
        for b in beliefs:
            q0[0] += b.weight * b.probs[0]
            q0[1] += b.weight * b.probs[1]
            q0[2] += b.weight * b.probs[2]
        q_prev_n = _normalize_simplex(q0)
    else:
        q_prev_n = _normalize_simplex(q_prev)

    # Compute belief dispersion → Bayesian shrinkage strength
    all_probs = [b.probs for b in beliefs]
    dispersion = 0.0
    if len(all_probs) >= 2:
        for i in range(3):
            vals = [p[i] for p in all_probs]
            mu = sum(vals) / len(vals)
            dispersion += sum((v - mu) ** 2 for v in vals) / len(vals)
    shrinkage_alpha = min(0.15, 0.5 * dispersion)

    # EMA of q_star for drift penalty
    if q_ema is None:
        q_ema_n = q_prev_n
    else:
        q_ema_n = _normalize_simplex(q_ema)

    q = q_prev_n
    trace: List[float] = []
    converged = False

    for it in range(1, max_iter + 1):
        eps = 1e-12
        grad_dist = [0.0, 0.0, 0.0]
        for b in beliefs:
            p = b.probs
            w = float(b.weight)
            for j in range(3):
                grad_dist[j] += w * 4.0 * (1.0 - sqrt((p[j] + eps) / (q[j] + eps)))

        grad_ent = [log((q[j] + eps) / (q_prev_n[j] + eps)) + 1.0 for j in range(3)]
        grad_arb = _no_arb_grad_fd(q, spot=spot, sigma=sigma, T=T, r=r)

        # Drift penalty gradient: 2 * lambda_drift * (q_j - ema_j)
        grad_drift = [lambda_drift * 2.0 * (q[j] - q_ema_n[j]) for j in range(3)]

        # Shrinkage gradient: 2 * alpha * (q_j - 1/3)
        uniform_j = 1.0 / 3.0
        grad_shrink = [shrinkage_alpha * 2.0 * (q[j] - uniform_j) for j in range(3)]

        grad = [
            grad_dist[j] + lambda_ent * grad_ent[j] + lambda_arb * grad_arb[j]
            + grad_drift[j] + grad_shrink[j]
            for j in range(3)
        ]

        q_new = _normalize_simplex([q[j] * exp(-lr * grad[j]) for j in range(3)])
        obj, _, _, _, _, _ = _objective_terms(
            q_new,
            beliefs=beliefs,
            q_prev=q_prev_n,
            lambda_ent=lambda_ent,
            lambda_arb=lambda_arb,
            spot=spot,
            sigma=sigma,
            T=T,
            r=r,
            q_ema=q_ema_n,
            lambda_drift=lambda_drift,
            shrinkage_alpha=shrinkage_alpha,
        )
        trace.append(obj)

        delta = sum(abs(q_new[j] - q[j]) for j in range(3))
        q = q_new
        if delta < tol:
            converged = True
            break

    obj, dist, ent, arb, _, _ = _objective_terms(
        q,
        beliefs=beliefs,
        q_prev=q_prev_n,
        lambda_ent=lambda_ent,
        lambda_arb=lambda_arb,
        spot=spot,
        sigma=sigma,
        T=T,
        r=r,
        q_ema=q_ema_n,
        lambda_drift=lambda_drift,
        shrinkage_alpha=shrinkage_alpha,
    )
    return EquilibriumArtifacts(
        q_star=_normalize_simplex(q),
        objective=float(obj),
        objective_trace=tuple(float(t) for t in trace),
        iterations=int(len(trace)),
        converged=bool(converged),
        distance_term=float(dist),
        entropy_term=float(ent),
        no_arb_term=float(arb),
        beliefs=tuple(beliefs),
    )


def equilibrium_option_price(
    q_star: Sequence[float],
    spot: float,
    strike: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "CE",
) -> float:
    qn = _normalize_simplex(q_star)
    states = _state_underlyings(spot=spot, sigma=sigma, T=T)
    disc = exp(-max(r, -0.02) * max(T, 1e-6))

    v = 0.0
    is_call = str(option_type).upper() in ("CE", "CALL", "C")
    for i in range(3):
        if is_call:
            payoff = max(states[i] - strike, 0.0)
        else:
            payoff = max(strike - states[i], 0.0)
        v += qn[i] * payoff
    return float(max(0.01, disc * v))


def equilibrium_iv_proxy(q_star: Sequence[float], sigma_base: float) -> float:
    qn = _normalize_simplex(q_star)
    x = (-1.0, 0.0, 1.0)
    mu = sum(qn[i] * x[i] for i in range(3))
    var = sum(qn[i] * (x[i] - mu) ** 2 for i in range(3))
    var_n = _clip(var / (2.0 / 3.0), 0.25, 2.5)
    return float(max(0.01, sigma_base * sqrt(var_n)))
