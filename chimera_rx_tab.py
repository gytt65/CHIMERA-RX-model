"""CHIMERA-RX Streamlit tab renderer (shadow/advisory only)."""

from __future__ import annotations

from typing import Any, MutableMapping, Optional

import pandas as pd

from chimera_rx_data_adapter import build_chimera_rx_input
from chimera_rx_engine import CHIMERARXEngine


def render(st: Any, session_state: MutableMapping[str, Any], engine: Optional[CHIMERARXEngine] = None) -> None:
    st.markdown("### 🧬 CHIMERA-RX (Shadow Mode)")
    st.caption("Advisory-only output. Existing OMEGA/NIRV/TVR model paths remain unchanged.")

    engine = engine or CHIMERARXEngine()
    x = build_chimera_rx_input(session_state)

    if x.spot <= 0 or x.strike <= 0:
        st.info("Select an option contract and fetch live data to run CHIMERA-RX shadow diagnostics.")
        return

    out = engine.run(x)
    result = out.to_dict()
    session_state["chimera_rx_latest"] = result
    session_state["chimera_rx_ptd_timeline"] = out.ptd_timeline
    session_state["chimera_rx_ccas_history"] = out.diagnostics.get("ccas_history", [])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eq Price", f"₹{out.eq_price:.2f}")
    m2.metric("Tail Price", f"₹{out.tail_price:.2f}")
    m3.metric("Mispricing", f"{out.mispricing:+.2f}%")
    m4.metric("Q Score", f"{out.q_score.Q_t:.2f}/10")

    g1, g2, g3 = st.columns(3)
    g1.metric("Compression Ratio", f"{out.abstention.CR_t:.3f}")
    g2.metric("Action", "YES" if out.abstention.action else "NO")
    g3.metric("Regime Alert", out.regime_alert)

    st.markdown("#### Equilibrium vs Market IV Surface")
    iv_diag = out.iv_surface_diag or {}
    iv_rows = iv_diag.get("rows", []) if isinstance(iv_diag, dict) else []
    iv_summary = iv_diag.get("summary", {}) if isinstance(iv_diag, dict) else {}
    if iv_rows:
        st.dataframe(pd.DataFrame(iv_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No IV surface diagnostics available for current input.")
    if iv_summary:
        s1, s2, s3 = st.columns(3)
        s1.metric("RMSE (bp)", f"{float(iv_summary.get('rmse_bp', 0.0)):.1f}")
        s2.metric("ATM Mean (bp)", f"{float(iv_summary.get('atm_mean_bp', 0.0)):+.1f}")
        s3.metric("OTM Mean (bp)", f"{float(iv_summary.get('otm_mean_bp', 0.0)):+.1f}")

    st.markdown("#### India Friction Decomposition")
    fr_df = pd.DataFrame(
        [
            {"component": "Margin", "value": out.friction.margin},
            {"component": "STT ITM", "value": out.friction.stt_itm},
            {"component": "OI Cluster", "value": out.friction.oi_cluster},
            {"component": "GEX", "value": out.friction.gex},
            {"component": "Overnight Gap", "value": out.friction.overnight_gap},
            {"component": "Total ψ", "value": out.friction.total},
        ]
    )
    st.dataframe(fr_df, hide_index=True, use_container_width=True)

    st.markdown("#### Abstention Frontier (Coverage vs Accuracy)")
    if out.frontier:
        fdf = pd.DataFrame(out.frontier)
        st.dataframe(fdf, hide_index=True, use_container_width=True)
    else:
        st.info("Frontier data is not available yet.")
    f1, f2 = st.columns(2)
    f1.metric("Coverage @ τ*", f"{float(out.diagnostics.get('coverage', 0.0)):.2f}")
    f2.metric("Actionable Accuracy @ τ*", f"{float(out.diagnostics.get('actionable_accuracy', 0.0)):.2f}")

    st.markdown("#### Regime Alerts Timeline")
    if out.ptd_timeline:
        tdf = pd.DataFrame(out.ptd_timeline[-12:])
        st.dataframe(tdf, hide_index=True, use_container_width=True)
    else:
        st.info("No regime timeline points yet.")

    st.markdown("#### Deployment Readiness")
    readiness = out.readiness or {}
    st.caption(f"Status: **{readiness.get('status', 'HOLD')}** | Action: **{readiness.get('action_label', 'ABSTAIN')}**")
    checks = readiness.get("checks", [])
    if checks:
        st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)
    else:
        st.info("No readiness checks available.")

    st.markdown("#### Quality Components")
    q_df = pd.DataFrame(
        [
            {"metric": "Accuracy A_t", "value": out.q_score.A_t},
            {"metric": "Robustness R_t", "value": out.q_score.R_t},
            {"metric": "No-Arb N_t", "value": out.q_score.N_t},
            {"metric": "Regime Lead G_t", "value": out.q_score.G_t},
            {"metric": "Selective S_t", "value": out.q_score.S_t},
        ]
    )
    st.dataframe(q_df, hide_index=True, use_container_width=True)

    st.markdown("#### Output Contract")
    st.json(
        {
            "action": out.abstention.action,
            "abstain": out.abstention.abstain,
            "confidence": round(out.confidence, 4),
            "mispricing": round(out.mispricing, 4),
            "q_score": round(out.q_score.Q_t, 4),
            "regime_alert": out.regime_alert,
            "coverage": round(float(out.diagnostics.get("coverage", 0.0)), 4),
            "actionable_accuracy": round(float(out.diagnostics.get("actionable_accuracy", 0.0)), 4),
            "readiness_status": readiness.get("status", "HOLD"),
        }
    )
