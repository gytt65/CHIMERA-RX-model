"""CHIMERA-RX data adapter from app/session state (read-only)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from chimera_rx_engine import CHIMERARXInput

try:
    from behavioral_agents import BehavioralAgents
    _BA_AVAILABLE = True
except Exception:  # pragma: no cover
    _BA_AVAILABLE = False

try:
    from structural_frictions import StructuralFrictions
    _SF_AVAILABLE = True
except Exception:  # pragma: no cover
    _SF_AVAILABLE = False


def _pick(d: Mapping[str, Any], keys, default):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _f(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_surface_observations(
    session_state: MutableMapping[str, Any],
    option_type: str,
) -> List[Dict[str, float]]:
    raw = session_state.get("raw_option_chain")
    rows: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        rows = [r for r in raw if isinstance(r, dict)]
    elif isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            rows = [r for r in raw.get("data", []) if isinstance(r, dict)]
        elif isinstance(raw.get("records"), list):
            rows = [r for r in raw.get("records", []) if isinstance(r, dict)]

    if not rows:
        return []

    is_call = str(option_type).upper() in ("CE", "CALL", "C")
    out: List[Dict[str, float]] = []
    for row in rows:
        strike = _f(row.get("strike_price"), -1.0)
        if strike <= 0:
            continue

        branch = row.get("call_options") if is_call else row.get("put_options")
        if not isinstance(branch, dict):
            branch = row.get("call_options") or row.get("put_options") or {}
        md = branch.get("market_data", {}) if isinstance(branch, dict) else {}
        if not isinstance(md, dict):
            md = {}
        greeks = branch.get("option_greeks", {}) if isinstance(branch, dict) else {}
        if not isinstance(greeks, dict):
            greeks = branch.get("greeks", {}) if isinstance(branch, dict) else {}
            if not isinstance(greeks, dict):
                greeks = {}

        ltp = _f(md.get("ltp", md.get("last_price", row.get("ltp"))), -1.0)
        iv = _f(greeks.get("iv", row.get("iv", row.get("iv_decimal"))), -1.0)
        if iv > 1.5:
            iv = iv / 100.0

        if ltp <= 0 or iv <= 0:
            continue

        out.append(
            {
                "strike": float(strike),
                "iv": float(iv),
                "price": float(ltp),
                "oi": float(_f(md.get("oi", row.get("open_interest", 0)), 0.0)),
            }
        )
        if len(out) >= 80:
            break
    return out


def build_chimera_rx_input(
    session_state: MutableMapping[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> CHIMERARXInput:
    """Build CHIMERARXInput from Streamlit session state with robust fallbacks."""
    ov = overrides or {}

    parsed = session_state.get("parsed_option")
    parsed = parsed if isinstance(parsed, dict) else {}

    spot = _f(_pick(ov, ["spot"], None), _f(_pick(parsed, ["spot", "underlying_price"], _pick(session_state, ["spot", "spot_price", "underlying_price"], 0.0))))
    strike = _f(_pick(ov, ["strike"], None), _f(_pick(parsed, ["strike", "strike_price"], _pick(session_state, ["strike", "selected_strike"], 0.0))))
    T = _f(_pick(ov, ["T"], None), _f(_pick(parsed, ["T", "time_to_expiry"], _pick(session_state, ["T", "time_to_expiry"], 0.05))))
    r = _f(_pick(ov, ["r"], None), _f(_pick(session_state, ["r", "risk_free_rate"], 0.065)))
    q = _f(_pick(ov, ["q"], None), _f(_pick(session_state, ["q", "dividend_yield"], 0.012)))
    sigma = _f(_pick(ov, ["sigma"], None), _f(_pick(parsed, ["iv", "sigma", "atm_iv"], _pick(session_state, ["sigma", "iv", "atm_iv"], 0.15))))

    option_type = _pick(ov, ["option_type"], None)
    if option_type is None:
        option_type = _pick(parsed, ["option_type", "type"], _pick(session_state, ["option_type", "ce_pe"], "CE"))
    option_type = str(option_type or "CE")

    market_price = _f(_pick(ov, ["market_price"], None), _f(_pick(parsed, ["market_price", "ltp", "price"], _pick(session_state, ["market_price", "ltp", "last_price"], 0.0))))

    # read-only references from existing model outputs if available
    nirv_price = None
    omega_price = None
    try:
        nirv_res = session_state.get("nirv_result")
        if nirv_res is not None and hasattr(nirv_res, "fair_value"):
            nirv_price = float(nirv_res.fair_value)
    except Exception:
        nirv_price = None
    try:
        omega_res = session_state.get("omega_result")
        if isinstance(omega_res, dict):
            omega_price = float(_pick(omega_res, ["fair_value", "omega_fair_value", "price"], 0.0))
    except Exception:
        omega_price = None

    ccas_history = session_state.get("chimera_rx_ccas_history")
    if not isinstance(ccas_history, list):
        ccas_history = None
    ptd_timeline = session_state.get("chimera_rx_ptd_timeline")
    if not isinstance(ptd_timeline, list):
        ptd_timeline = None
    surface_obs = _extract_surface_observations(session_state, option_type=option_type)

    # --- Behavioral bridging: compute distorted IV via BehavioralAgents ---
    behavioral_distorted_iv: Optional[float] = None
    if _BA_AVAILABLE and spot > 0 and strike > 0 and T > 0 and sigma > 0:
        try:
            ba = BehavioralAgents()
            fii_flow_val = _f(_pick(session_state, ["fii_net_flow", "fii_net_flow_cr"], 0.0))
            distorted_iv = ba.apply_behavioral_distortions(
                spot=spot, strike=strike, T=T,
                base_iv=sigma, fii_net_flow_cr=fii_flow_val,
            )
            if distorted_iv is not None and math.isfinite(distorted_iv) and distorted_iv > 0:
                behavioral_distorted_iv = float(distorted_iv)
        except Exception:
            pass

    # --- Structural friction bridging ---
    structural_margin_premium: Optional[float] = None
    structural_stt_distortion: Optional[float] = None
    structural_overnight_adj: Optional[float] = None
    if _SF_AVAILABLE and spot > 0 and strike > 0 and T > 0:
        try:
            sf = StructuralFrictions(
                spot=spot, strike=strike, T=T,
                r=r, lot_size=int(_f(_pick(session_state, ["lot_size"], 75), 75)),
            )
            vix_val = _f(_pick(session_state, ["india_vix", "vix"], 15.0))
            dte = max(T * 365.0, 1.0)
            mp = sf.calculate_seller_margin_premium(iv=sigma, days_to_expiry=dte, vix=vix_val)
            if mp is not None and math.isfinite(mp):
                structural_margin_premium = float(mp)

            intrinsic = max(spot - strike, 0.0) if option_type.upper() in ("CE", "CALL", "C") else max(strike - spot, 0.0)
            if intrinsic > 0:
                effective = sf.apply_stt_exercise_distortion(
                    option_type=option_type, intrinsic_value=intrinsic,
                )
                stt_loss = (intrinsic - effective) / max(spot, 1.0)
                if math.isfinite(stt_loss):
                    structural_stt_distortion = float(stt_loss)

            overnight_adj_iv = sf.overnight_gap_risk_adjustment(iv=sigma, is_overnight=True)
            if overnight_adj_iv is not None and math.isfinite(overnight_adj_iv):
                structural_overnight_adj = float((overnight_adj_iv - sigma))
        except Exception:
            pass

    return CHIMERARXInput(
        spot=max(spot, 0.0),
        strike=max(strike, 0.0),
        T=max(T, 1e-6),
        r=r,
        q=q,
        sigma=max(sigma, 1e-6),
        option_type=option_type,
        market_price=max(market_price, 0.01),
        vix=_f(_pick(session_state, ["india_vix", "vix", "atm_iv"], 15.0)),
        pcr=_f(_pick(session_state, ["pcr_oi", "pcr"], 1.0)),
        vrp_30d=_f(_pick(session_state, ["vrp_30d", "vrp"], 0.0)),
        hawkes_cluster=_f(_pick(session_state, ["hawkes_cluster", "jump_cluster"], 0.0)),
        hurst=_f(_pick(session_state, ["hurst", "hurst_estimate"], 0.2)),
        fii_net_flow_cr=_f(_pick(session_state, ["fii_net_flow", "fii_net_flow_cr"], 0.0)),
        geopolitical_proxy=_f(_pick(session_state, ["geopolitical_proxy"], 0.0)),
        climate_monsoon_proxy=_f(_pick(session_state, ["climate_monsoon_proxy", "monsoon_proxy"], 0.0)),
        margin_proxy=_f(_pick(session_state, ["seller_margin_premium", "margin_proxy"], 0.0)),
        stt_itm_proxy=_f(_pick(session_state, ["stt_itm_proxy", "stt_exercise_distortion"], 0.0)),
        oi_cluster_proxy=_f(_pick(session_state, ["oi_cluster_proxy", "oi_concentration"], 0.0)),
        gex_proxy=_f(_pick(session_state, ["gex_total", "gex", "gex_proxy"], 0.0)),
        overnight_gap_proxy=_f(_pick(session_state, ["overnight_gap", "overnight_gap_proxy"], 0.0)),
        behavioral_distorted_iv=behavioral_distorted_iv,
        structural_margin_premium=structural_margin_premium,
        structural_stt_distortion=structural_stt_distortion,
        structural_overnight_adj=structural_overnight_adj,
        nirv_price=nirv_price,
        omega_price=omega_price,
        surface_observations=surface_obs if surface_obs else None,
        ccas_history=ccas_history,
        ptd_timeline=ptd_timeline,
    )
