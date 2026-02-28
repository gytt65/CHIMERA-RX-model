from chimera_rx_ccas import calibrate_tau, compute_compression_diagnostics


def test_compression_ratio_higher_for_better_fit():
    d_good, _, _ = compute_compression_diagnostics(
        errors=[0.01, 0.02, 0.015, 0.018],
        samples=[],
        current_correct=1.0,
    )
    d_bad, _, _ = compute_compression_diagnostics(
        errors=[0.20, 0.18, 0.22, 0.25],
        samples=[],
        current_correct=0.0,
    )
    assert d_good.CR_t >= d_bad.CR_t


def test_frontier_coverage_monotone_with_threshold():
    samples = [
        {"cr": 0.9, "correct": 0.0},
        {"cr": 1.0, "correct": 0.0},
        {"cr": 1.1, "correct": 0.0},
        {"cr": 1.3, "correct": 1.0},
        {"cr": 1.4, "correct": 1.0},
        {"cr": 1.5, "correct": 1.0},
        {"cr": 1.7, "correct": 1.0},
        {"cr": 1.9, "correct": 1.0},
        {"cr": 2.1, "correct": 1.0},
    ]
    tau, best, frontier = calibrate_tau(samples=samples, target_accuracy=0.80, coverage_bounds=(0.35, 0.55))

    assert tau > 0.0
    assert best.coverage >= 0.0
    covers = [p.coverage for p in frontier]
    assert covers == sorted(covers, reverse=True)

