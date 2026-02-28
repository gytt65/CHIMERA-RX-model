from chimera_rx_engine import CHIMERARXEngine, CHIMERARXInput


def test_surface_diag_includes_atm_otm_and_deep_otm():
    out = CHIMERARXEngine().run(
        CHIMERARXInput(
            spot=22500.0,
            strike=22500.0,
            T=0.07,
            r=0.065,
            q=0.012,
            sigma=0.15,
            option_type="CE",
            market_price=180.0,
            vix=15.0,
            pcr=1.0,
        )
    )

    rows = out.iv_surface_diag.get("rows", [])
    buckets = {r.get("bucket") for r in rows}
    maturities = {round(float(r.get("T", 0.0)), 6) for r in rows}

    assert "ATM" in buckets
    assert "OTM" in buckets
    assert "DEEP_OTM" in buckets
    assert len(maturities) >= 3
    assert out.iv_surface_diag.get("summary", {}).get("n_points", 0) >= 9

