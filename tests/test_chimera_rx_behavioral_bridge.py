"""Tests for CHIMERA-RX behavioral bridging and identifiability controls."""

from chimera_rx_engine import CHIMERARXEngine, CHIMERARXInput
from chimera_rx_equilibrium import build_agent_beliefs, solve_equilibrium
from chimera_rx_state import StateTensor, build_state_tensor


def _base_input(**overrides) -> CHIMERARXInput:
    defaults = dict(
        spot=22500.0,
        strike=22600.0,
        T=0.08,
        r=0.065,
        q=0.012,
        sigma=0.16,
        option_type="CE",
        market_price=180.0,
        vix=14.2,
        pcr=0.95,
        vrp_30d=0.10,
        hawkes_cluster=0.4,
        hurst=0.14,
        fii_net_flow_cr=-350.0,
        margin_proxy=0.2,
        stt_itm_proxy=0.1,
        oi_cluster_proxy=0.15,
        gex_proxy=-0.2,
        overnight_gap_proxy=0.05,
    )
    defaults.update(overrides)
    return CHIMERARXInput(**defaults)


# -- StateTensor BSE bridging --


def test_state_tensor_contains_bse_fields():
    st = build_state_tensor({"india_vix": 16.0, "pcr_oi": 1.1})
    assert hasattr(st, "sentiment")
    assert hasattr(st, "lottery_demand")
    assert hasattr(st, "limits_to_arb")
    vec = st.as_vector()
    assert len(vec) == 14  # 11 original + 3 BSE fields


# -- Behavioral distorted IV --


def test_behavioral_distorted_iv_changes_output():
    engine = CHIMERARXEngine()
    x_no_biv = _base_input(behavioral_distorted_iv=None)
    x_with_biv = _base_input(behavioral_distorted_iv=0.22)
    out_base = engine.run(x_no_biv)

    engine2 = CHIMERARXEngine()
    out_biv = engine2.run(x_with_biv)

    base_summary = out_base.iv_surface_diag.get("summary", {})
    biv_summary = out_biv.iv_surface_diag.get("summary", {})
    assert (
        base_summary.get("rmse_bp", 0) != biv_summary.get("rmse_bp", 0)
        or base_summary.get("otm_mean_bp", 0) != biv_summary.get("otm_mean_bp", 0)
    )


# -- Structural friction bridging --


def test_structural_friction_values_override_flat_proxies():
    engine = CHIMERARXEngine()
    x_flat = _base_input(structural_margin_premium=None)
    x_struct = _base_input(structural_margin_premium=0.008)
    out_flat = engine.run(x_flat)

    engine2 = CHIMERARXEngine()
    out_struct = engine2.run(x_struct)

    assert out_flat.friction.margin != out_struct.friction.margin


def test_structural_stt_overrides_proxy():
    engine = CHIMERARXEngine()
    x = _base_input(structural_stt_distortion=0.003)
    out = engine.run(x)
    assert 0.0 <= out.friction.stt_itm <= 0.08


# -- Identifiability controls --


def test_drift_penalty_stabilises_consecutive_calls():
    engine = CHIMERARXEngine()
    results = []
    for i in range(5):
        x = _base_input(vix=14.0 + 0.1 * i, pcr=0.95 + 0.01 * i)
        out = engine.run(x)
        results.append(out)

    for i in range(1, len(results)):
        assert results[i].diagnostics.get("eq_converged", True)


def test_bayesian_shrinkage_reduces_extreme_concentration():
    st = build_state_tensor({
        "india_vix": 30.0,
        "pcr_oi": 0.5,
        "fii_net_flow": -2000.0,
        "gex_total": -500000.0,
    })
    beliefs = build_agent_beliefs(st, "CE")
    eq = solve_equilibrium(
        beliefs=beliefs,
        q_prev=None,
        spot=22500.0,
        sigma=0.25,
        T=0.08,
        r=0.065,
        lambda_drift=0.08,
    )
    assert max(eq.q_star) < 0.85
    assert min(eq.q_star) > 0.02


def test_engine_output_contract_unchanged():
    engine = CHIMERARXEngine()
    x = _base_input()
    out = engine.run(x)

    assert out.eq_price > 0
    assert out.deformed_price > 0
    assert out.tail_price > 0
    assert -1000.0 <= out.mispricing <= 1000.0
    assert out.abstention.tau > 0
    assert 0.0 <= out.q_score.Q_t <= 10.0
    assert out.iv_surface_diag.get("rows")
    assert out.frontier
    assert out.ptd_timeline
    assert out.readiness.get("status") in ("READY", "HOLD")
