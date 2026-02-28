from chimera_rx_equilibrium import (
    build_agent_beliefs,
    no_arb_penalty,
    solve_equilibrium,
)
from chimera_rx_state import build_state_tensor


def test_equilibrium_solver_returns_valid_simplex():
    st = build_state_tensor(
        {
            "india_vix": 16.5,
            "pcr_oi": 1.08,
            "fii_net_flow": -550.0,
            "gex_total": -220000.0,
            "overnight_gap": 0.12,
            "seller_margin_premium": 0.15,
            "stt_itm_proxy": 0.08,
            "oi_cluster_proxy": 0.12,
        }
    )
    beliefs = build_agent_beliefs(st, option_type="CE")
    eq = solve_equilibrium(
        beliefs=beliefs,
        q_prev=None,
        spot=22500.0,
        sigma=0.16,
        T=0.08,
        r=0.065,
    )

    q = eq.q_star
    assert len(q) == 3
    assert abs(sum(q) - 1.0) < 1e-8
    assert all(v > 0.0 for v in q)
    assert eq.iterations > 0
    assert len(eq.objective_trace) == eq.iterations


def test_no_arb_penalty_higher_for_extreme_concentration():
    p_bal = no_arb_penalty(q=(0.33, 0.34, 0.33), spot=22500.0, sigma=0.16, T=0.08, r=0.065)
    p_conc = no_arb_penalty(q=(0.99, 0.005, 0.005), spot=22500.0, sigma=0.16, T=0.08, r=0.065)
    assert p_conc >= p_bal

