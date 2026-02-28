import datetime

import pytest

from omega_features import set_features
from omega_model import OMEGAModel


@pytest.fixture(autouse=True)
def _reset_features():
    set_features()
    yield
    set_features()


def _pred_record(pred_id, *, fair_value, market_price, contract_key, ts_epoch, outcome=None):
    return {
        "id": pred_id,
        "features": {
            "regime_sideways": 1.0,
            "time_to_expiry": 7.0 / 365.0,
        },
        "pred": {
            "signal": "BUY",
            "fair_value": float(fair_value),
            "base_fair_value": float(fair_value),
            "market_price": float(market_price),
            "contract_key": contract_key,
        },
        "contract_key": contract_key,
        "ts": datetime.datetime.fromtimestamp(ts_epoch).isoformat(),
        "ts_epoch": float(ts_epoch),
        "outcome": outcome,
    }


def test_rebuild_final_conformal_from_tracker(tmp_path):
    model = OMEGAModel(data_dir=str(tmp_path))
    model._omega_final_min_samples = 3

    ckey = "CE:23500:0.019178"
    model.tracker.predictions = [
        _pred_record("p1", fair_value=100, market_price=100, contract_key=ckey, ts_epoch=100.0,
                     outcome={"actual_return": 0.01, "actual_price": 105.0, "ts": "x"}),
        _pred_record("p2", fair_value=100, market_price=100, contract_key=ckey, ts_epoch=101.0,
                     outcome={"actual_return": -0.01, "actual_price": 96.0, "ts": "x"}),
        _pred_record("p3", fair_value=100, market_price=100, contract_key=ckey, ts_epoch=102.0,
                     outcome={"actual_return": 0.02, "actual_price": 109.0, "ts": "x"}),
    ]

    model._rebuild_final_conformal_from_tracker()
    assert model._omega_final_q_global is not None
    assert model._omega_final_q_global > 0


def test_auto_reconcile_pending_marks_outcome(tmp_path):
    model = OMEGAModel(data_dir=str(tmp_path))
    model.ml = None  # isolate reconciliation behavior

    now_ts = 1_000.0
    old_ts = now_ts - 600.0
    ckey = "CE:23500:0.019178"

    model.tracker.predictions = [
        _pred_record(
            "pending_1",
            fair_value=100.0,
            market_price=100.0,
            contract_key=ckey,
            ts_epoch=old_ts,
            outcome=None,
        )
    ]
    model._quote_cache[ckey] = {"ts": now_ts, "price": 108.0}

    closed = model._auto_reconcile_pending(now_ts)
    assert closed == 1
    assert model.tracker.predictions[0].get("outcome") is not None
    assert model.tracker.predictions[0]["outcome"]["actual_price"] == pytest.approx(108.0)


def test_final_conformal_q_uses_regime_blend(tmp_path):
    model = OMEGAModel(data_dir=str(tmp_path))
    model._omega_final_q_global = 0.10
    model._omega_final_q_by_regime = {"Sideways": 0.20}

    q = model._omega_final_q({"regime_sideways": 1.0, "time_to_expiry": 7.0 / 365.0})
    assert q is not None
    assert q > 0.10

