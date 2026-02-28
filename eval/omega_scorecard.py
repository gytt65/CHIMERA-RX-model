#!/usr/bin/env python3
"""
omega_scorecard.py
==================

Deterministic evaluation harness for OMEGA "9.5/10" evidence claims.
Computes a 100-point score across:
  - Predictive quality (35)
  - Economic quality (30)
  - Calibration quality (20)
  - Robustness (10)
  - Regression integrity (5)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import sys
from collections import defaultdict

import numpy as np

# Ensure repo root is importable when running as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backtester import SyntheticNiftyGenerator, NirvBacktester
from omega_features import set_best_mode_max_accuracy, set_features
from omega_model import OMEGAModel


def _safe_float(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _load_predictions(predictions_path: str):
    if not os.path.exists(predictions_path):
        return []
    try:
        with open(predictions_path, "r") as f:
            payload = json.load(f)
        return payload.get("predictions", []) if isinstance(payload, dict) else []
    except Exception:
        return []


def _directional_rows(predictions):
    rows = []
    for p in predictions:
        pred = p.get("pred") or {}
        out = p.get("outcome") or {}
        sig = str(pred.get("signal", "")).upper()
        if sig not in ("BUY", "STRONG BUY", "SELL", "STRONG SELL"):
            continue
        if "actual_return" not in out:
            continue
        ret = _safe_float(out.get("actual_return"), 0.0)
        signed = ret if "BUY" in sig else -ret
        conf = _safe_float(pred.get("confidence_level"), np.nan)
        feat = p.get("features") or {}
        regime = "Unknown"
        if _safe_float(feat.get("regime_bull_low"), 0.0) > 0.5:
            regime = "Bull-Low Vol"
        elif _safe_float(feat.get("regime_bear_high"), 0.0) > 0.5:
            regime = "Bear-High Vol"
        elif _safe_float(feat.get("regime_bull_high"), 0.0) > 0.5:
            regime = "Bull-High Vol"
        elif _safe_float(feat.get("regime_sideways"), 0.0) > 0.5:
            regime = "Sideways"
        rows.append(
            {
                "signal": sig,
                "signed_return_pct": signed * 100.0,
                "correct": 1.0 if signed > 0 else 0.0,
                "confidence_level": conf,
                "regime": regime,
            }
        )
    return rows


def compute_calibration_ece(rows):
    conf = np.array([r["confidence_level"] for r in rows if np.isfinite(r.get("confidence_level", np.nan))], dtype=float)
    hit = np.array([r["correct"] for r in rows if np.isfinite(r.get("confidence_level", np.nan))], dtype=float)
    if len(conf) < 40:
        return None
    conf01 = np.clip(conf / 100.0, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf01 >= lo) & (conf01 < hi if i < 9 else conf01 <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(hit[mask]))
        cal = float(np.mean(conf01[mask]))
        frac = float(np.mean(mask))
        ece += frac * abs(acc - cal)
    return float(ece)


def _predictive_quality(rows):
    if not rows:
        return {
            "score": 0.0,
            "global_accuracy_pct": 0.0,
            "side_accuracy_min_pct": 0.0,
            "regime_accuracy_min_pct": 0.0,
        }
    acc = 100.0 * float(np.mean([r["correct"] for r in rows]))
    by_side = defaultdict(list)
    by_regime = defaultdict(list)
    for r in rows:
        side = "BUY" if "BUY" in r["signal"] else "SELL"
        by_side[side].append(r["correct"])
        by_regime[r["regime"]].append(r["correct"])
    side_min = 100.0 * min((float(np.mean(v)) for v in by_side.values()), default=0.0)
    reg_min = 100.0 * min((float(np.mean(v)) for v in by_regime.values() if len(v) >= 10), default=acc / 100.0) if by_regime else 0.0

    score_acc = 25.0 * np.clip((acc - 50.0) / 15.0, 0.0, 1.0)
    score_side = 5.0 * np.clip((side_min - 48.0) / 12.0, 0.0, 1.0)
    score_reg = 5.0 * np.clip((reg_min - 48.0) / 12.0, 0.0, 1.0)
    return {
        "score": float(score_acc + score_side + score_reg),
        "global_accuracy_pct": round(float(acc), 2),
        "side_accuracy_min_pct": round(float(side_min), 2),
        "regime_accuracy_min_pct": round(float(reg_min), 2),
    }


def _economic_quality(days: int, seeds: list[int], initial_capital: float):
    metrics = []
    for seed in seeds:
        gen = SyntheticNiftyGenerator(seed=int(seed), initial_spot=23500)
        snaps = gen.generate(n_days=int(days))
        bt = NirvBacktester(
            initial_capital=float(initial_capital),
            model_type="omega",
            regime_filter=False,
            signal_threshold=3.0,
            n_paths=6000,
            omega_min_conviction=8,
            omega_require_cost_cover=True,
        )
        out = bt.run(snaps)
        metrics.append(out.get("metrics", {}) or {})

    sharpe = float(np.mean([_safe_float(m.get("sharpe"), 0.0) for m in metrics])) if metrics else 0.0
    pf = float(np.mean([_safe_float(m.get("profit_factor"), 0.0) for m in metrics])) if metrics else 0.0
    win = float(np.mean([_safe_float(m.get("win_rate"), 0.0) for m in metrics])) if metrics else 0.0
    mdd = float(np.mean([_safe_float(m.get("max_drawdown_pct"), 100.0) for m in metrics])) if metrics else 100.0
    pnl = float(np.mean([_safe_float(m.get("total_pnl"), 0.0) for m in metrics])) if metrics else 0.0
    pnl_pct = 100.0 * pnl / max(float(initial_capital), 1.0)

    score = 0.0
    score += 12.0 * np.clip((sharpe - 0.50) / 1.50, 0.0, 1.0)
    score += 8.0 * np.clip((pf - 1.00) / 0.80, 0.0, 1.0)
    score += 5.0 * np.clip((15.0 - mdd) / 12.0, 0.0, 1.0)
    score += 5.0 * np.clip(pnl_pct / 15.0, 0.0, 1.0)
    return {
        "score": float(score),
        "mean_sharpe": round(sharpe, 3),
        "mean_profit_factor": round(pf, 3),
        "mean_win_rate_pct": round(win, 2),
        "mean_max_drawdown_pct": round(mdd, 2),
        "mean_total_pnl": round(pnl, 2),
        "mean_total_pnl_pct": round(pnl_pct, 2),
    }


def _robustness_quality(n_cases: int = 40):
    rng = np.random.default_rng(123)
    finite_ok = 0
    runs = 0
    with tempfile.TemporaryDirectory() as td:
        model = OMEGAModel(data_dir=td)
        for _ in range(int(n_cases)):
            runs += 1
            spot = float(rng.uniform(21000.0, 25500.0))
            strike = float(round(spot / 50.0) * 50.0 + rng.integers(-500, 550))
            T = float(rng.uniform(2.0 / 365.0, 30.0 / 365.0))
            ivx = float(rng.uniform(10.0, 30.0))
            mp = float(rng.uniform(20.0, 450.0))
            rets = rng.normal(0.0, 0.01, 30)
            try:
                out = model.price_option(
                    spot=spot,
                    strike=strike,
                    T=T,
                    r=0.065,
                    q=0.012,
                    option_type='CE',
                    market_price=mp,
                    india_vix=ivx,
                    fii_net_flow=0.0,
                    dii_net_flow=0.0,
                    days_to_rbi=20,
                    pcr_oi=1.0,
                    returns_30d=rets,
                    bid=max(mp * 0.97, 0.5),
                    ask=max(mp * 1.03, 0.6),
                    volume_oi_ratio=0.1,
                )
                if np.isfinite(float(getattr(out, "fair_value", np.nan))) and np.isfinite(float(getattr(out, "confidence_level", np.nan))):
                    finite_ok += 1
            except Exception:
                pass
    ratio = finite_ok / max(runs, 1)
    return {"score": float(10.0 * ratio), "finite_ratio": round(float(ratio), 4), "cases": int(runs)}


def build_scorecard(
    predictions_path: str,
    seeds: list[int],
    days: int,
    initial_capital: float,
    pytest_passed: bool = False,
    golden_passed: bool = False,
):
    preds = _load_predictions(predictions_path)
    rows = _directional_rows(preds)
    ece = compute_calibration_ece(rows)

    predictive = _predictive_quality(rows)
    economic = _economic_quality(days=days, seeds=seeds, initial_capital=initial_capital)
    if ece is None:
        calibration_score = 0.0
    else:
        calibration_score = 20.0 * np.clip((0.18 - float(ece)) / 0.18, 0.0, 1.0)
    robustness = _robustness_quality()
    regression_score = 0.0
    regression_score += 2.5 if bool(pytest_passed) else 0.0
    regression_score += 2.5 if bool(golden_passed) else 0.0

    total = float(predictive["score"] + economic["score"] + calibration_score + robustness["score"] + regression_score)
    directional_samples = int(len(rows))
    eligible = bool(total >= 95.0 and directional_samples >= 300)
    hard_gates = {
        "total_score_100 >= 90": {"value": round(total, 3), "target": 90.0, "pass": bool(total >= 90.0)},
        "predictive_quality_35 >= 28": {"value": round(float(predictive["score"]), 3), "target": 28.0, "pass": bool(float(predictive["score"]) >= 28.0)},
        "economic_quality_30 >= 22": {"value": round(float(economic["score"]), 3), "target": 22.0, "pass": bool(float(economic["score"]) >= 22.0)},
        "calibration_quality_20 >= 14": {"value": round(float(calibration_score), 3), "target": 14.0, "pass": bool(float(calibration_score) >= 14.0)},
        "robustness_10 >= 9": {"value": round(float(robustness["score"]), 3), "target": 9.0, "pass": bool(float(robustness["score"]) >= 9.0)},
        "regression_integrity_5 == 5": {"value": round(float(regression_score), 3), "target": 5.0, "pass": bool(abs(float(regression_score) - 5.0) <= 1e-9)},
    }
    hard_passed = bool(all(bool(v.get("pass")) for v in hard_gates.values()))

    return {
        "total_score_100": round(total, 3),
        "claim_eligible_95plus": eligible,
        "hard_gate_target_90plus_passed": hard_passed,
        "hard_gates": hard_gates,
        "directional_oos_samples": directional_samples,
        "minimum_samples_required": 300,
        "components": {
            "predictive_quality_35": predictive,
            "economic_quality_30": economic,
            "calibration_quality_20": {
                "score": round(float(calibration_score), 3),
                "ece": None if ece is None else round(float(ece), 5),
            },
            "robustness_10": robustness,
            "regression_integrity_5": {
                "score": round(float(regression_score), 3),
                "pytest_passed": bool(pytest_passed),
                "golden_passed": bool(golden_passed),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="OMEGA 9.5/10 scorecard harness")
    parser.add_argument("--predictions", default="omega_data/predictions.json", help="Path to OMEGA predictions tracker JSON")
    parser.add_argument("--days", type=int, default=120, help="Synthetic backtest horizon per seed")
    parser.add_argument("--seeds", default="41,42,43,44,45", help="Comma-separated deterministic seeds")
    parser.add_argument("--capital", type=float, default=500000.0, help="Initial capital for economic scoring")
    parser.add_argument("--pytest-passed", action="store_true", help="Set when full pytest suite passed")
    parser.add_argument("--golden-passed", action="store_true", help="Set when golden suite passed")
    parser.add_argument("--output", default="", help="Optional JSON file output path")
    args = parser.parse_args()

    # Scorecard is intended for max-accuracy profile comparison runs.
    set_best_mode_max_accuracy()
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if str(s).strip()]
    report = build_scorecard(
        predictions_path=str(args.predictions),
        seeds=seeds,
        days=int(args.days),
        initial_capital=float(args.capital),
        pytest_passed=bool(args.pytest_passed),
        golden_passed=bool(args.golden_passed),
    )
    print(json.dumps(report, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    # Reset to explicit baseline singleton after run.
    set_features()


if __name__ == "__main__":
    main()
