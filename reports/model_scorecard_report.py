#!/usr/bin/env python3
"""
model_scorecard_report.py
=========================

Report writer for model scorecards (JSON + Markdown).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict


def _render_md(report: Dict) -> str:
    ts = report.get("generated_at", "")
    metrics = report.get("model_metrics", {})
    ranks = report.get("rankings", [])
    lines = [
        "# Model Scorecard",
        "",
        f"Generated at: `{ts}`",
        f"Snapshots evaluated: `{report.get('snapshot_count', 0)}`",
        "",
        "## Ranking",
        "",
        "| Rank | Model | Score (/10) | MAE | MAPE% | Edge Hit Rate | Mean Latency (ms) | Coverage | Samples |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranks:
        lines.append(
            "| {rank} | {model} | {score:.2f} | {mae:.3f} | {mape:.2f} | {ehr:.2f} | {lat:.2f} | {cov} | {n} |".format(
                rank=int(r.get("rank", 0)),
                model=str(r.get("model", "")),
                score=float(r.get("score_0_10", 0.0)),
                mae=float(r.get("mae", 0.0)),
                mape=float(r.get("mape_pct", 0.0)),
                ehr=float(r.get("edge_hit_rate", 0.0)),
                lat=float(r.get("latency_ms_mean", 0.0)),
                cov="-" if r.get("coverage") is None else "{:.2f}".format(float(r.get("coverage", 0.0))),
                n=int(r.get("samples", 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `edge_hit_rate` uses `|model - market| - estimated_roundtrip_cost > 0`.",
            "- `coverage` is interval inclusion rate when model exposes intervals.",
            "- `score` is a weighted objective blend of edge, error, coverage, latency, and direction.",
        ]
    )
    return "\n".join(lines)


def write_scorecard_report(report: Dict, out_dir: str, stem: str = "scorecard") -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(out_dir, f"{stem}_{ts}.json")
    md_path = os.path.join(out_dir, f"{stem}_{ts}.md")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w") as f:
        f.write(_render_md(report))
    latest_path = os.path.join(out_dir, f"{stem}_latest.json")
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)
    return {"json": json_path, "md": md_path, "latest": latest_path}
