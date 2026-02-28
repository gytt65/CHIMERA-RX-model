"""State tensor construction for CHIMERA-RX."""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign, log1p
from typing import Any, Mapping, Tuple

try:
    from behavioral_state_engine import BehavioralInputs, BehavioralStateEngine
    _BSE_AVAILABLE = True
except Exception:  # pragma: no cover
    _BSE_AVAILABLE = False


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _signed_log_scale(x: float, scale: float, cap: float = 1.0) -> float:
    v = _f(x, 0.0)
    s = max(_f(scale, 1.0), 1e-6)
    mag = log1p(abs(v) / s) / log1p(cap + 1.0)
    return _clip(copysign(mag, v), -1.0, 1.0)


@dataclass(frozen=True)
class StateTensor:
    """Compact state vector used by equilibrium/reflexive modules."""

    macro: float
    geopolitical: float
    climate_monsoon: float
    behavioral_flow: float
    microstructure_friction: float
    overnight_gap: float
    vix_norm: float
    pcr_norm: float
    fii_flow_norm: float
    gex_norm: float
    stress_index: float
    # Behavioral state engine signals (bridged)
    sentiment: float = 0.0
    lottery_demand: float = 0.0
    limits_to_arb: float = 0.0

    def as_vector(self) -> Tuple[float, ...]:
        return (
            self.macro,
            self.geopolitical,
            self.climate_monsoon,
            self.behavioral_flow,
            self.microstructure_friction,
            self.overnight_gap,
            self.vix_norm,
            self.pcr_norm,
            self.fii_flow_norm,
            self.gex_norm,
            self.stress_index,
            self.sentiment,
            self.lottery_demand,
            self.limits_to_arb,
        )


def build_state_tensor(values: Mapping[str, Any]) -> StateTensor:
    """Build bounded state factors from app/session features."""
    vix = _f(values.get("india_vix", values.get("vix", 15.0)), 15.0)
    pcr = _f(values.get("pcr_oi", values.get("pcr", 1.0)), 1.0)
    fii_flow = _f(values.get("fii_net_flow", values.get("fii_net_flow_cr", 0.0)), 0.0)
    gex = _f(values.get("gex_total", values.get("gex", 0.0)), 0.0)
    overnight_gap = _f(values.get("overnight_gap", values.get("overnight_gap_proxy", 0.0)), 0.0)

    margin = _f(values.get("seller_margin_premium", values.get("margin_proxy", 0.0)), 0.0)
    stt = _f(values.get("stt_itm_proxy", values.get("stt_exercise_distortion", 0.0)), 0.0)
    oi_cluster = _f(values.get("oi_cluster_proxy", values.get("oi_concentration", 0.0)), 0.0)

    geopolitical = _clip(_f(values.get("geopolitical_proxy", 0.0), 0.0), -1.0, 1.0)
    climate = _clip(_f(values.get("climate_monsoon_proxy", values.get("monsoon_proxy", 0.0)), 0.0), -1.0, 1.0)

    vix_norm = _clip((vix - 14.0) / 20.0, -1.0, 1.0)
    pcr_norm = _clip((pcr - 1.0) / 0.6, -1.0, 1.0)
    fii_flow_norm = _signed_log_scale(fii_flow, scale=400.0, cap=8.0)
    gex_norm = _signed_log_scale(gex, scale=5e5, cap=15.0)
    gap_norm = _clip(overnight_gap, -1.5, 1.5) / 1.5

    micro = _clip(
        0.35 * _clip(margin, -1.5, 1.5) / 1.5
        + 0.20 * _clip(stt, -1.5, 1.5) / 1.5
        + 0.20 * _clip(oi_cluster, -1.5, 1.5) / 1.5
        + 0.25 * gex_norm,
        -1.0,
        1.0,
    )

    # --- Behavioral State Engine bridging ---
    sentiment = 0.0
    lottery_demand = 0.0
    limits_to_arb = 0.0
    if _BSE_AVAILABLE:
        try:
            bse = BehavioralStateEngine()
            bse_inp = BehavioralInputs(
                news_sentiment=_f(values.get("news_sentiment", 0.0), 0.0),
                social_sentiment=_f(values.get("social_sentiment", 0.0), 0.0),
                otm_call_skew=_f(values.get("otm_call_skew", 0.0), 0.0),
                put_call_ratio=_f(values.get("pcr_oi", values.get("pcr", 1.0)), 1.0),
                bid_ask_spread=_f(values.get("bid_ask_spread", 0.01), 0.01),
                volume_oi_ratio=_f(values.get("volume_oi_ratio", 0.1), 0.1),
                total_gex=_f(values.get("gex_total", values.get("gex", 0.0)), 0.0),
                dealer_flow_imbalance=_f(values.get("dealer_flow_imbalance", 0.0), 0.0),
            )
            sentiment = _clip(bse.compute_sentiment(bse_inp), -1.0, 1.0)
            lottery_demand = _clip(bse.compute_lottery_demand(bse_inp), -1.0, 1.0)
            limits_to_arb = _clip(bse.compute_limits_to_arb(bse_inp), 0.0, 1.0)
        except Exception:
            pass  # graceful fallback to defaults

    # Enrich behavioral_flow with BSE sentiment & lottery demand
    behavioral = _clip(
        0.40 * pcr_norm
        - 0.28 * fii_flow_norm
        + 0.12 * _clip(_f(values.get("retail_lottery_proxy", 0.0), 0.0), -1.0, 1.0)
        + 0.12 * sentiment
        + 0.08 * lottery_demand,
        -1.0,
        1.0,
    )

    macro = _clip(
        0.55 * vix_norm
        - 0.25 * fii_flow_norm
        + 0.20 * _clip(_f(values.get("usd_inr_proxy", 0.0), 0.0), -1.0, 1.0),
        -1.0,
        1.0,
    )

    # Enrich stress_index with limits-to-arb
    stress = _clip(
        0.26 * abs(macro)
        + 0.22 * abs(behavioral)
        + 0.22 * abs(micro)
        + 0.17 * abs(gap_norm)
        + 0.13 * limits_to_arb,
        0.0,
        1.0,
    )

    return StateTensor(
        macro=macro,
        geopolitical=geopolitical,
        climate_monsoon=climate,
        behavioral_flow=behavioral,
        microstructure_friction=micro,
        overnight_gap=gap_norm,
        vix_norm=vix_norm,
        pcr_norm=pcr_norm,
        fii_flow_norm=fii_flow_norm,
        gex_norm=gex_norm,
        stress_index=stress,
        sentiment=sentiment,
        lottery_demand=lottery_demand,
        limits_to_arb=limits_to_arb,
    )

