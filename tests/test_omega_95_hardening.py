import types

import numpy as np

from omega_features import set_features
from omega_model import OMEGAModel, PredictionTracker


class _DummyNIRV:
    def price_option(
        self,
        spot,
        strike,
        T,
        r,
        q,
        option_type,
        market_price,
        india_vix,
        fii_net_flow,
        dii_net_flow,
        days_to_rbi,
        pcr_oi,
        returns_30d,
        inr_usd_vol=0.05,
        **kwargs,
    ):
        return types.SimpleNamespace(
            fair_value=104.0,
            market_price=float(market_price),
            mispricing_pct=4.0,
            signal="BUY",
            profit_probability=58.0,
            physical_profit_prob=71.0,
            confidence_level=84.0,
            expected_pnl=1200.0,
            physical_expected_pnl=980.0,
            regime="Bull-Low Vol",
            greeks={
                "delta": 0.42,
                "gamma": 0.001,
                "theta": -2.0,
                "vega": 11.0,
                "quant_extras": {
                    "regime_blend": {
                        "mode": "single",
                        "primary_regime": "Bull-Low Vol",
                        "primary_prob": 0.85,
                    }
                },
            },
            tc_details={},
        )


def _pred(signal, actual_return, regime="bull_low"):
    f = {
        "regime_bull_low": 1.0 if regime == "bull_low" else 0.0,
        "regime_bear_high": 1.0 if regime == "bear_high" else 0.0,
        "regime_sideways": 1.0 if regime == "sideways" else 0.0,
        "regime_bull_high": 1.0 if regime == "bull_high" else 0.0,
    }
    return {
        "pred": {"signal": signal},
        "outcome": {"actual_return": float(actual_return)},
        "features": f,
    }


def test_price_option_blocks_directional_signal_when_costs_exceed_edge(tmp_path):
    try:
        set_features(
            USE_CONFORMAL_INTERVALS=False,
            USE_RESEARCH_HIGH_CONVICTION=False,
            USE_OOS_RELIABILITY_GATE=False,
            USE_OMEGA_DRIFT_GUARD=False,
        )
        model = OMEGAModel(nirv_model=_DummyNIRV(), data_dir=str(tmp_path))
        model.ml = None  # isolate deterministic behavior
        out = model.price_option(
            spot=23500.0,
            strike=23500.0,
            T=7.0 / 365.0,
            r=0.065,
            q=0.012,
            option_type="CE",
            market_price=100.0,
            india_vix=14.0,
            fii_net_flow=0.0,
            dii_net_flow=0.0,
            days_to_rbi=20,
            pcr_oi=1.0,
            returns_30d=np.zeros(30),
            bid=70.0,
            ask=130.0,
            volume_oi_ratio=0.02,
        )
        assert out.edge_covers_costs is False
        assert out.net_edge < 0.0
        assert out.signal == "HOLD"
        assert "EDGE_BELOW_TRANSACTION_COST" in (out.actionability_reasons or [])
    finally:
        set_features()


def test_reliability_gate_relaxes_threshold_during_regime_transition():
    tracker = PredictionTracker.__new__(PredictionTracker)
    preds = []
    preds.extend(_pred("BUY", -0.005, "bull_low") for _ in range(22))
    preds.extend(_pred("BUY", 0.012, "bull_low") for _ in range(28))
    tracker.predictions = preds

    gate = tracker.get_reliability_gate_decision(
        signal="BUY",
        features={
            "regime_bull_low": 1.0,
            "regime_primary_prob": 0.45,  # uncertain transition state
        },
        min_samples=40,
        min_accuracy_pct=58.0,
        min_avg_edge_pct=0.10,
    )
    assert gate["required"] is True
    assert gate["effective_min_accuracy_pct"] < 58.0
    assert gate["passed"] is True
