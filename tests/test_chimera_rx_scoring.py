from chimera_rx_scoring import compute_quality_score


def test_quality_score_bounds_and_weights():
    q = compute_quality_score(
        accuracy=0.80,
        robustness=0.70,
        no_arb_integrity=0.90,
        regime_lead=0.60,
        selective_accuracy=0.85,
    )

    assert 0.0 <= q.A_t <= 1.0
    assert 0.0 <= q.R_t <= 1.0
    assert 0.0 <= q.N_t <= 1.0
    assert 0.0 <= q.G_t <= 1.0
    assert 0.0 <= q.S_t <= 1.0
    assert 0.0 <= q.Q_t <= 10.0


def test_quality_score_improves_with_accuracy():
    q_lo = compute_quality_score(0.40, 0.70, 0.80, 0.70, 0.70)
    q_hi = compute_quality_score(0.90, 0.70, 0.80, 0.70, 0.70)
    assert q_hi.Q_t > q_lo.Q_t
