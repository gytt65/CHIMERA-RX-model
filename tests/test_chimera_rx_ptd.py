from chimera_rx_ptd import compute_ptd_signals, update_ptd_timeline
from chimera_rx_reflexive import update_reflexive_state
from chimera_rx_state import build_state_tensor


def test_ptd_escalates_under_stress():
    st_low = build_state_tensor({"india_vix": 13.5, "pcr_oi": 1.0, "fii_net_flow": 50.0, "gex_total": 10000.0, "overnight_gap": 0.01})
    st_high = build_state_tensor(
        {
            "india_vix": 28.0,
            "pcr_oi": 1.35,
            "fii_net_flow": -1800.0,
            "gex_total": -750000.0,
            "overnight_gap": 1.2,
            "seller_margin_premium": 0.9,
            "oi_cluster_proxy": 0.8,
        }
    )
    rf_low = update_reflexive_state(st_low, hurst=0.32, hawkes_cluster=0.2)
    rf_high = update_reflexive_state(st_high, hurst=0.08, hawkes_cluster=2.2)

    sig_low = compute_ptd_signals(st_low, rf_low, iv_residual=0.01, timeline=[])
    sig_high = compute_ptd_signals(st_high, rf_high, iv_residual=0.18, timeline=[])

    assert sig_high.score >= sig_low.score
    assert sig_high.alert in ("ELEVATED", "STRESS")


def test_ptd_timeline_rolls_and_appends():
    st = build_state_tensor({"india_vix": 18.0, "pcr_oi": 1.1, "fii_net_flow": -300.0, "gex_total": -120000.0, "overnight_gap": 0.2})
    rf = update_reflexive_state(st, hurst=0.18, hawkes_cluster=0.9)
    sig = compute_ptd_signals(st, rf, iv_residual=0.05, timeline=[])

    timeline = []
    for _ in range(70):
        timeline = update_ptd_timeline(timeline, sig, max_len=32)

    assert len(timeline) == 32
    assert "score" in timeline[-1]

