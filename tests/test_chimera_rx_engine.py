from chimera_rx_engine import CHIMERARXEngine, CHIMERARXInput


def test_chimera_rx_engine_smoke_output_contract():
    engine = CHIMERARXEngine(tau=1.05)
    x = CHIMERARXInput(
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

def test_chimera_rx_compression_ratio_worsens_on_large_mismatch():
    x0 = CHIMERARXInput(
        spot=23000.0,
        strike=23000.0,
        T=0.05,
        r=0.065,
        q=0.012,
        sigma=0.14,
        option_type="PE",
        market_price=140.0,
    )
    probe = CHIMERARXEngine().run(x0)

    x_good = CHIMERARXInput(**{**x0.__dict__, "market_price": float(probe.tail_price)})
    x_bad = CHIMERARXInput(**{**x0.__dict__, "market_price": float(probe.tail_price * 2.0)})
    out_good = CHIMERARXEngine().run(x_good)
    out_bad = CHIMERARXEngine().run(x_bad)

    assert out_good.abstention.CR_t >= out_bad.abstention.CR_t
    assert out_good.q_score.Q_t >= out_bad.q_score.Q_t
