"""CHIMERA-RX shadow engine (additive-only, no OMEGA/NIRV/TVR mutation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import erf, exp, log, sqrt
from typing import Any, Dict, List, Optional, Sequence

from chimera_rx_ccas import CompressionDiagnostics, FrontierPoint, compute_compression_diagnostics
from chimera_rx_equilibrium import (
    EquilibriumArtifacts,
    build_agent_beliefs,
    equilibrium_iv_proxy,
    equilibrium_option_price,
    solve_equilibrium,
)
from chimera_rx_ptd import PTDSignals, compute_ptd_signals, update_ptd_timeline
from chimera_rx_reflexive import ReflexiveState, reflexive_price_adjustment, update_reflexive_state
from chimera_rx_scoring import ModelQualityScore, compute_quality_score
from chimera_rx_state import build_state_tensor


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _bsm_price(spot: float, strike: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    if T <= 0:
        if option_type.upper() in ("CE", "CALL"):
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    sigma = max(float(sigma), 1e-6)
    d1 = (log(max(spot, 1e-6) / max(strike, 1e-6)) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option_type.upper() in ("CE", "CALL"):
        return spot * exp(-q * T) * _norm_cdf(d1) - strike * exp(-r * T) * _norm_cdf(d2)
    return strike * exp(-r * T) * _norm_cdf(-d2) - spot * exp(-q * T) * _norm_cdf(-d1)


def _implied_vol_from_price(
    price: float,
    spot: float,
    strike: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
) -> float:
    target = max(float(price), 1e-6)
    lo, hi = 1e-4, 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = _bsm_price(spot, strike, T, r, q, mid, option_type)
        if p > target:
            hi = mid
        else:
            lo = mid
    return float(_clip(0.5 * (lo + hi), 1e-4, 4.0))


@dataclass(frozen=True)
class CHIMERARXInput:
    spot: float
    strike: float
    T: float
    r: float
    q: float
    sigma: float
    option_type: str
    market_price: float

    # optional state
    vix: float = 15.0
    pcr: float = 1.0
    vrp_30d: float = 0.0
    hawkes_cluster: float = 0.0
    hurst: float = 0.2
    fii_net_flow_cr: float = 0.0
    geopolitical_proxy: float = 0.0
    climate_monsoon_proxy: float = 0.0

    # Indian friction proxies
    margin_proxy: float = 0.0
    stt_itm_proxy: float = 0.0
    oi_cluster_proxy: float = 0.0
    gex_proxy: float = 0.0
    overnight_gap_proxy: float = 0.0

    # Behavioral bridging (from BehavioralAgents)
    behavioral_distorted_iv: Optional[float] = None
    # Structural friction bridging (from StructuralFrictions)
    structural_margin_premium: Optional[float] = None
    structural_stt_distortion: Optional[float] = None
    structural_overnight_adj: Optional[float] = None

    # optional references (read-only)
    nirv_price: Optional[float] = None
    omega_price: Optional[float] = None

    # optional histories / diagnostics input
    surface_observations: Optional[List[Dict[str, Any]]] = None
    ccas_history: Optional[List[Dict[str, float]]] = None
    ptd_timeline: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class FrictionVector:
    margin: float
    stt_itm: float
    oi_cluster: float
    gex: float
    overnight_gap: float
    total: float


@dataclass(frozen=True)
class AbstentionTuple:
    CR_t: float
    tau: float
    abstain: bool
    action: bool


@dataclass(frozen=True)
class CHIMERARXOutput:
    eq_price: float
    deformed_price: float
    tail_price: float
    fair_value: float
    base_fair_value: float
    overlay_adjustment: float
    abstain_fallback_used: bool
    interval_low: float
    interval_high: float
    market_price: float
    mispricing: float
    friction: FrictionVector
    abstention: AbstentionTuple
    q_score: ModelQualityScore
    confidence: float
    regime_alert: str
    diagnostics: Dict[str, Any]
    iv_surface_diag: Dict[str, Any]
    frontier: List[Dict[str, float]]
    ptd_timeline: List[Dict[str, Any]]
    readiness: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["CR_t"] = self.abstention.CR_t
        out["abstain"] = self.abstention.abstain
        out["action"] = self.abstention.action
        out["q_t"] = self.q_score.Q_t
        return out


class CHIMERARXEngine:
    """Shadow-mode evaluator for CHIMERA-RX."""

    def __init__(
        self,
        tau: float = 1.15,
        alpha_t: float = 3.0,
        xi_t: float = 0.15,
        kappa: float = 1.0,
        lambda_ent: float = 0.15,
        lambda_arb: float = 2.0,
    ):
        self.tau = float(tau)
        self.alpha_t = float(alpha_t)
        self.xi_t = float(xi_t)
        self.kappa = float(max(kappa, 1e-6))
        self.lambda_ent = float(max(lambda_ent, 0.0))
        self.lambda_arb = float(max(lambda_arb, 0.0))
        self._q_prev: Sequence[float] | None = None
        self._q_ema: Sequence[float] | None = None
        self._ema_decay: float = 0.85
        self._hawkes_intensity: float = 0.0

    def apply_india_frictions(self, x: CHIMERARXInput) -> FrictionVector:
        b1, b2, b3, b4, b5 = 0.06, 0.04, 0.05, 0.03, 0.08
        # Use structural friction values when bridged, else fall back to flat proxies
        margin_val = (
            _clip(x.structural_margin_premium, 0.0, 0.10)
            if x.structural_margin_premium is not None
            else b1 * _clip(x.margin_proxy, -1.5, 1.5)
        )
        stt_val = (
            _clip(x.structural_stt_distortion, 0.0, 0.08)
            if x.structural_stt_distortion is not None
            else b2 * _clip(x.stt_itm_proxy, 0.0, 2.0)
        )
        oi_cluster = b3 * _clip(x.oi_cluster_proxy, -1.5, 1.5)
        gex = b4 * _clip(x.gex_proxy, -2.0, 2.0)
        overnight_val = (
            _clip(x.structural_overnight_adj, -0.10, 0.10)
            if x.structural_overnight_adj is not None
            else b5 * _clip(x.overnight_gap_proxy, -1.5, 1.5)
        )
        total = margin_val + stt_val + oi_cluster + gex + overnight_val
        return FrictionVector(
            margin=float(margin_val),
            stt_itm=float(stt_val),
            oi_cluster=float(oi_cluster),
            gex=float(gex),
            overnight_gap=float(overnight_val),
            total=float(total),
        )

    def apply_tail_correction(self, deformed_price: float, x: CHIMERARXInput) -> float:
        m = abs(log(max(x.strike, 1e-6) / max(x.spot, 1e-6)))
        tail_mult = (1.0 + m / self.kappa) ** (-self.alpha_t) * exp(-self.xi_t * m)
        tail_adj = 1.0 + 0.35 * _clip(tail_mult, 0.0, 1.0)
        return float(max(0.01, deformed_price * tail_adj))

    def _compute_iv_surface_diag(
        self,
        x: CHIMERARXInput,
        eq: EquilibriumArtifacts,
        friction: FrictionVector,
        eq_iv: float,
        stress: float,
    ) -> Dict[str, Any]:
        is_call = str(x.option_type).upper() in ("CE", "CALL", "C")
        if is_call:
            strikes = {
                "ATM": x.spot,
                "OTM": x.spot * 1.03,
                "DEEP_OTM": x.spot * 1.08,
            }
        else:
            strikes = {
                "ATM": x.spot,
                "OTM": x.spot * 0.97,
                "DEEP_OTM": x.spot * 0.92,
            }
        maturities = [max(x.T * 0.5, 1.0 / 365.0), max(x.T, 1.0 / 365.0), min(max(x.T * 1.8, 2.0 / 365.0), 0.9)]

        rows: List[Dict[str, float | str]] = []
        errors: List[float] = []
        diff_bp: List[float] = []
        for b, k in strikes.items():
            wing = abs(log(max(k, 1e-6) / max(x.spot, 1e-6)))
            for t in maturities:
                # Use behavioral distorted IV as baseline when available
                base_iv = x.behavioral_distorted_iv if x.behavioral_distorted_iv is not None else eq_iv
                local_sigma = max(0.01, base_iv * (1.0 + 0.12 * wing * 10.0))
                eq_p = equilibrium_option_price(
                    q_star=eq.q_star,
                    spot=x.spot,
                    strike=k,
                    T=t,
                    r=x.r,
                    sigma=local_sigma,
                    option_type=x.option_type,
                )
                eq_p = max(0.01, eq_p * exp(0.75 * friction.total))
                mkt_sigma = max(
                    0.01,
                    x.sigma * (1.0 + 0.16 * stress + 0.10 * wing * 10.0),
                )
                mkt_p = _bsm_price(x.spot, k, t, x.r, x.q, mkt_sigma, x.option_type)

                eq_iv_k = _implied_vol_from_price(eq_p, x.spot, k, t, x.r, x.q, x.option_type)
                mkt_iv_k = _implied_vol_from_price(mkt_p, x.spot, k, t, x.r, x.q, x.option_type)
                d_bp = (eq_iv_k - mkt_iv_k) * 10000.0
                rel = abs(eq_p - mkt_p) / max(mkt_p, 1e-6)

                rows.append(
                    {
                        "bucket": b,
                        "T": float(t),
                        "strike": float(k),
                        "eq_iv": float(eq_iv_k),
                        "mkt_iv": float(mkt_iv_k),
                        "diff_bp": float(d_bp),
                    }
                )
                errors.append(float(rel))
                diff_bp.append(float(d_bp))

        if diff_bp:
            rmse_bp = sqrt(sum(v * v for v in diff_bp) / len(diff_bp))
            atm = [r["diff_bp"] for r in rows if r["bucket"] == "ATM"]
            otm = [r["diff_bp"] for r in rows if r["bucket"] in ("OTM", "DEEP_OTM")]
        else:
            rmse_bp = 0.0
            atm = []
            otm = []

        summary = {
            "rmse_bp": float(rmse_bp),
            "atm_mean_bp": float(sum(atm) / max(len(atm), 1)),
            "otm_mean_bp": float(sum(otm) / max(len(otm), 1)),
            "n_points": int(len(rows)),
        }
        return {"rows": rows, "errors": errors, "summary": summary}

    def _compute_quality(
        self,
        x: CHIMERARXInput,
        model_price: float,
        gate: AbstentionTuple,
        eq: EquilibriumArtifacts,
        ptd: PTDSignals,
        ccas: CompressionDiagnostics,
        reflexive: ReflexiveState,
    ) -> ModelQualityScore:
        rel_err = abs(model_price - max(x.market_price, 1e-6)) / max(x.market_price, 1e-6)
        accuracy = _clip(1.0 - rel_err, 0.0, 1.0)
        robustness = _clip(
            0.70 - 0.30 * ptd.score + 0.20 * (1.0 - abs(reflexive.vol_adjust)),
            0.0,
            1.0,
        )
        no_arb = _clip(1.0 - 8.0 * eq.no_arb_term, 0.0, 1.0)
        regime_lead = _clip(0.60 * ptd.score + 0.40 * reflexive.regime_intensity, 0.0, 1.0)
        selective = _clip(0.65 * ccas.accuracy + 0.35 * (1.0 if gate.action else 0.4), 0.0, 1.0)
        return compute_quality_score(
            accuracy=accuracy,
            robustness=robustness,
            no_arb_integrity=no_arb,
            regime_lead=regime_lead,
            selective_accuracy=selective,
        )

    def _build_readiness(
        self,
        q_score: ModelQualityScore,
        eq: EquilibriumArtifacts,
        ccas: CompressionDiagnostics,
        gate: AbstentionTuple,
    ) -> Dict[str, Any]:
        checks = [
            {"name": "Q_t >= 8.5", "pass": bool(q_score.Q_t >= 8.5), "value": round(float(q_score.Q_t), 3), "target": 8.5},
            {"name": "No-Arb Penalty <= 0.02", "pass": bool(eq.no_arb_term <= 0.02), "value": round(float(eq.no_arb_term), 5), "target": 0.02},
            {"name": "Actionable Accuracy >= 0.80", "pass": bool(ccas.accuracy >= 0.80), "value": round(float(ccas.accuracy), 3), "target": 0.80},
            {
                "name": "Coverage in [0.35, 0.55]",
                "pass": bool(0.35 <= ccas.coverage <= 0.55),
                "value": round(float(ccas.coverage), 3),
                "target": "[0.35,0.55]",
            },
        ]
        status = "READY" if all(bool(c["pass"]) for c in checks) else "HOLD"
        action_label = "ADVISORY_ACTION" if (status == "READY" and gate.action) else "ABSTAIN"
        return {"status": status, "action_label": action_label, "checks": checks}

    def run(self, x: CHIMERARXInput) -> CHIMERARXOutput:
        state = build_state_tensor(
            {
                "india_vix": x.vix,
                "pcr_oi": x.pcr,
                "fii_net_flow": x.fii_net_flow_cr,
                "gex_total": x.gex_proxy,
                "overnight_gap": x.overnight_gap_proxy,
                "seller_margin_premium": x.margin_proxy,
                "stt_itm_proxy": x.stt_itm_proxy,
                "oi_cluster_proxy": x.oi_cluster_proxy,
                "geopolitical_proxy": x.geopolitical_proxy,
                "climate_monsoon_proxy": x.climate_monsoon_proxy,
            }
        )
        beliefs = build_agent_beliefs(state, option_type=x.option_type)
        eq_art = solve_equilibrium(
            beliefs=beliefs,
            q_prev=self._q_prev,
            spot=x.spot,
            sigma=x.sigma,
            T=x.T,
            r=x.r,
            lambda_ent=self.lambda_ent,
            lambda_arb=self.lambda_arb,
            q_ema=self._q_ema,
        )
        self._q_prev = eq_art.q_star
        # Update EMA for drift penalty
        if self._q_ema is None:
            self._q_ema = eq_art.q_star
        else:
            d = self._ema_decay
            self._q_ema = tuple(
                d * self._q_ema[j] + (1.0 - d) * eq_art.q_star[j]
                for j in range(3)
            )

        eq_iv = equilibrium_iv_proxy(eq_art.q_star, sigma_base=x.sigma)
        eq_price_raw = equilibrium_option_price(
            q_star=eq_art.q_star,
            spot=x.spot,
            strike=x.strike,
            T=x.T,
            r=x.r,
            sigma=eq_iv,
            option_type=x.option_type,
        )

        reflexive = update_reflexive_state(
            state=state,
            hurst=x.hurst,
            hawkes_cluster=x.hawkes_cluster,
            prev_intensity=self._hawkes_intensity,
        )
        self._hawkes_intensity = reflexive.hawkes_intensity
        eq_price = reflexive_price_adjustment(eq_price_raw, reflexive)

        friction = self.apply_india_frictions(x)
        deformed_price = float(max(0.01, eq_price * exp(friction.total)))
        tail_price = self.apply_tail_correction(deformed_price, x)

        iv_diag = self._compute_iv_surface_diag(
            x=x,
            eq=eq_art,
            friction=friction,
            eq_iv=eq_iv,
            stress=state.stress_index,
        )

        mkt = max(x.market_price, 1e-6)
        rel_error = abs(tail_price - mkt) / mkt
        error_stream = [float(rel_error)] + [float(e) for e in iv_diag.get("errors", [])]

        ccas_diag, ccas_history, frontier = compute_compression_diagnostics(
            errors=error_stream,
            samples=x.ccas_history,
            current_correct=1.0 if rel_error <= 0.12 else 0.0,
            target_accuracy=0.80,
            coverage_bounds=(0.35, 0.55),
            k=7,
        )
        gate = AbstentionTuple(
            CR_t=float(ccas_diag.CR_t),
            tau=float(ccas_diag.tau),
            abstain=bool(ccas_diag.abstain),
            action=bool(ccas_diag.action),
        )

        ptd = compute_ptd_signals(
            state=state,
            reflexive=reflexive,
            iv_residual=float(iv_diag.get("summary", {}).get("rmse_bp", 0.0)) / 10000.0,
            timeline=x.ptd_timeline,
        )
        ptd_timeline = update_ptd_timeline(x.ptd_timeline, ptd)

        q_score = self._compute_quality(
            x=x,
            model_price=tail_price,
            gate=gate,
            eq=eq_art,
            ptd=ptd,
            ccas=ccas_diag,
            reflexive=reflexive,
        )
        readiness = self._build_readiness(q_score=q_score, eq=eq_art, ccas=ccas_diag, gate=gate)
        base_candidates = []
        if x.nirv_price is not None:
            try:
                v = float(x.nirv_price)
                if v > 0 and abs(v) < 1e9:
                    base_candidates.append(v)
            except Exception:
                pass
        if x.omega_price is not None:
            try:
                v = float(x.omega_price)
                if v > 0 and abs(v) < 1e9:
                    base_candidates.append(v)
            except Exception:
                pass
        if len(base_candidates) >= 2:
            base_fair_value = 0.45 * base_candidates[0] + 0.55 * base_candidates[1]
        elif len(base_candidates) == 1:
            base_fair_value = base_candidates[0]
        else:
            base_fair_value = float(tail_price)

        ptd_bias = float(_clip(2.0 * ptd.score - 1.0, -1.0, 1.0))
        reflexive_bias = float(_clip(
            0.8 * reflexive.drift_adjust + 0.3 * reflexive.regime_intensity - 0.6 * abs(reflexive.vol_adjust),
            -1.0,
            1.0,
        ))
        eq_bias = float(_clip(eq_art.q_star[0] - eq_art.q_star[1], -1.0, 1.0))
        overlay_pct = float(_clip(
            0.020 * ptd_bias + 0.010 * reflexive_bias + 0.008 * eq_bias - 0.006 * friction.total,
            -0.06,
            0.06,
        ))
        overlay_adjustment = float(base_fair_value * overlay_pct)

        abstain_fallback_used = bool(gate.abstain or readiness.get("status") != "READY" or not gate.action)
        fair_value = float(base_fair_value if abstain_fallback_used else (base_fair_value + overlay_adjustment))
        fair_value = float(max(fair_value, 0.01))
        if abstain_fallback_used:
            overlay_adjustment = 0.0
            readiness = dict(readiness)
            readiness["action_label"] = "ABSTAIN"

        interval_half_pct = float(_clip(
            0.025 + 0.030 * ptd.score + 0.015 * abs(reflexive.vol_adjust) + 0.010 * state.stress_index,
            0.01,
            0.20,
        ))
        if abstain_fallback_used:
            interval_half_pct *= 1.2
        interval_low = float(max(0.01, fair_value * (1.0 - interval_half_pct)))
        interval_high = float(max(interval_low, fair_value * (1.0 + interval_half_pct)))

        mispricing = float((fair_value - mkt) / mkt * 100.0)
        confidence = float(_clip(0.30 + 0.35 * (ccas_diag.CR_t - 1.0) + 0.25 * (q_score.Q_t / 10.0), 0.0, 1.0))
        if abstain_fallback_used:
            confidence = float(_clip(confidence * 0.80, 0.0, 1.0))

        diagnostics = {
            "tau_input": float(self.tau),
            "alpha_t": float(self.alpha_t),
            "xi_t": float(self.xi_t),
            "kappa": float(self.kappa),
            "lambda_ent": float(self.lambda_ent),
            "lambda_arb": float(self.lambda_arb),
            "rel_error": float(rel_error),
            "eq_objective": float(eq_art.objective),
            "eq_no_arb_term": float(eq_art.no_arb_term),
            "eq_entropy_term": float(eq_art.entropy_term),
            "eq_distance_term": float(eq_art.distance_term),
            "eq_converged": bool(eq_art.converged),
            "base_fair_value": float(base_fair_value),
            "overlay_adjustment": float(overlay_adjustment),
            "overlay_pct": float(overlay_pct),
            "abstain_fallback_used": bool(abstain_fallback_used),
            "interval_half_pct": float(interval_half_pct),
            "L0": float(ccas_diag.L0),
            "LM": float(ccas_diag.LM),
            "coverage": float(ccas_diag.coverage),
            "actionable_accuracy": float(ccas_diag.accuracy),
            "ccas_history": ccas_history,
            "ptd": {
                "score": float(ptd.score),
                "curvature": float(ptd.curvature),
                "critical_slowing": float(ptd.critical_slowing),
                "susceptibility": float(ptd.susceptibility),
                "symmetry_break": float(ptd.symmetry_break),
            },
            "reflexive": {
                "rough_component": float(reflexive.rough_component),
                "hawkes_intensity": float(reflexive.hawkes_intensity),
                "regime_intensity": float(reflexive.regime_intensity),
                "drift_adjust": float(reflexive.drift_adjust),
                "vol_adjust": float(reflexive.vol_adjust),
            },
        }

        frontier_rows = [
            {"tau": float(p.tau), "coverage": float(p.coverage), "accuracy": float(p.accuracy), "n_actions": float(p.n_actions)}
            for p in frontier
        ]

        return CHIMERARXOutput(
            eq_price=float(eq_price),
            deformed_price=float(deformed_price),
            tail_price=float(tail_price),
            fair_value=float(fair_value),
            base_fair_value=float(base_fair_value),
            overlay_adjustment=float(overlay_adjustment),
            abstain_fallback_used=bool(abstain_fallback_used),
            interval_low=float(interval_low),
            interval_high=float(interval_high),
            market_price=float(x.market_price),
            mispricing=mispricing,
            friction=friction,
            abstention=gate,
            q_score=q_score,
            confidence=confidence,
            regime_alert=str(ptd.alert),
            diagnostics=diagnostics,
            iv_surface_diag={"rows": iv_diag.get("rows", []), "summary": iv_diag.get("summary", {})},
            frontier=frontier_rows,
            ptd_timeline=ptd_timeline,
            readiness=readiness,
        )

    def predict(self, x: CHIMERARXInput) -> CHIMERARXOutput:
        """
        Backward/forward-compatible alias for callers expecting predict(...).
        """
        return self.run(x)
