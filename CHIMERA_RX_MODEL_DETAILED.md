# CHIMERA-RX Model Documentation (Implementation-Aligned)

## 1. Executive Summary

CHIMERA-RX is an additive, shadow-mode option analytics engine designed for Indian index/options microstructure. It does **not** replace or mutate OMEGA/NIRV/TVR pricing paths. It runs in parallel, produces advisory diagnostics, and focuses on:

1. Endogenous equilibrium-style pricing (`q_t*`) from heterogeneous agent beliefs.
2. Reflexive regime adjustment using roughness and Hawkes-like clustering proxies.
3. Indian friction deformation (margin/STT/OI cluster/GEX/overnight gap).
4. Tail-aware correction for OTM and deep OTM behavior.
5. Causal-compression abstention (MDL-like gating) with dynamic thresholding.
6. Regime transition diagnostics (PTD) with a rolling alert timeline.
7. Research-grade model quality score `Q_t ∈ [0,10]` and deployment readiness checks.

CHIMERA-RX is currently engineered as a **decision-support layer**, not an execution authority.

### 1.1 Current Audit Snapshot (27 February 2026)

This section reflects the latest read-only repository scan and test run.

1. **Role in system**: CHIMERA-RX is additive and shadow/advisory only (`chimera_rx_engine.py`, `chimera_rx_tab.py`).
2. **Model score (current)**: `7.0 / 10` for practical deployment value in this repo state.
3. **Primary strengths**:
   1. Strong governance framing (`abstain`, `readiness`, `quality score`).
   2. Isolation from OMEGA/NIRV/TVR paths.
   3. Clear operator-facing diagnostics for uncertainty and regime stress.
4. **Primary constraints**:
   1. Proxy-heavy behavioral/friction parameterization (heuristic constants).
   2. No direct execution authority by design.
   3. Requires replay and calibration evidence for production-grade confidence.
5. **Validation status snapshot**:
   1. Focused CHIMERA tests pass (`12` passing in engine/equilibrium/PTD/scoring/surface/tab smoke).
   2. OMEGA non-regression with CHIMERA imports passes (`tests/test_omega_outputs_unchanged_with_chimera.py`).
   3. One CHIMERA test file currently has a syntax/collection issue (`tests/test_chimera_rx_behavioral_bridge.py`) and must be fixed to restore full-suite collection.

---

## 2. What CHIMERA-RX Is

CHIMERA-RX is a structured hybrid between:

1. A compact equilibrium model over discrete latent states.
2. A reflexive dynamics adjustment layer.
3. A market-friction deformation operator.
4. A selective prediction gate.
5. A model governance/risk dashboard.

It is implemented through the following modules:

1. [chimera_rx_state.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_state.py)
2. [chimera_rx_equilibrium.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_equilibrium.py)
3. [chimera_rx_reflexive.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_reflexive.py)
4. [chimera_rx_ptd.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_ptd.py)
5. [chimera_rx_ccas.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_ccas.py)
6. [chimera_rx_scoring.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_scoring.py)
7. [chimera_rx_engine.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_engine.py)
8. [chimera_rx_data_adapter.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_data_adapter.py)
9. [chimera_rx_tab.py](/Users/caaniketh/Optionpricingmodel Ai/OPM-with-TVR-NIRV-OMEGA-Models-/chimera_rx_tab.py)

---

## 3. What CHIMERA-RX Does

At every inference call (`CHIMERARXEngine.run`), CHIMERA-RX:

1. Builds a normalized market state tensor.
2. Constructs five agent belief distributions and solves a constrained equilibrium objective.
3. Applies reflexive rough-Hawkes style adjustment.
4. Applies explicit Indian market friction deformation.
5. Applies tail correction.
6. Compares equilibrium-vs-market IV surfaces (ATM/OTM/deep OTM across maturities).
7. Computes MDL-style compression diagnostics and abstention threshold `tau`.
8. Computes PTD (phase-transition diagnostics) and updates timeline.
9. Computes `Q_t` from five quality components.
10. Computes readiness status (`READY`/`HOLD`) from hard checks.
11. Returns full advisory output payload for UI and downstream analysis.

---

## 4. Model Philosophy

CHIMERA-RX addresses the practical issue that direct calibration-only pipelines can conflate:

1. True mispricing.
2. Regime shift.
3. Model misspecification.

It introduces an explicit “model applicability” signal using compression ratio and selective actioning:

\[
CR_t = \frac{L_0}{L_M}, \quad \text{action if } CR_t > \tau_t
\]

This makes the model intentionally selective rather than always-on.

---

## 5. Mathematical Structure

### 5.1 State Tensor

CHIMERA-RX maps raw inputs into bounded factors:

1. `macro`
2. `geopolitical`
3. `climate_monsoon`
4. `behavioral_flow`
5. `microstructure_friction`
6. `overnight_gap`
7. `vix_norm`
8. `pcr_norm`
9. `fii_flow_norm`
10. `gex_norm`
11. `stress_index`

Each is clipped to stable ranges for robust numerical behavior.

### 5.2 Agent Belief Equilibrium

Five agent classes are built:

1. Retail
2. Institutional
3. Market maker
4. FII
5. Arbitrageur

Each has a probability simplex belief over three latent return states (`down`, `mid`, `up`) and a dynamic capital weight.

The solved objective is:

\[
J(q) = \sum_i w_i \, d_{FI}^2(q,p_i) + \lambda_{ent}\,KL(q\|q_{prev}) + \lambda_{arb}\,\Phi_{NA}(q)
\]

Where:

1. \(d_{FI}^2\) is approximated through Hellinger/Fisher-style distance.
2. `KL` regularizes temporal drift.
3. \(\Phi_{NA}\) penalizes no-arbitrage violations across strikes.

Optimization uses mirror-descent style simplex updates.

### 5.3 Reflexive Dynamics

Reflexive update uses:

1. Roughness proxy from Hurst (`rough_component`).
2. Hawkes-like intensity recursion (`hawkes_intensity`).
3. Regime intensity synthesis.
4. Drift/volatility adjustment terms.

This produces a reflexive multiplier applied to equilibrium price.

### 5.4 Indian Friction Deformation

Friction decomposition:

\[
\psi_t = b_1\,\text{Margin} + b_2\,\text{STT}_{ITM} + b_3\,\text{OICluster} + b_4\,\text{GEX} + b_5\,\text{OvernightGap}
\]

\[
V_{deformed} = V_{eq}\exp(\psi_t)
\]

### 5.5 Tail Correction

Tail adjustment:

\[
tail\_mult = (1 + m/\kappa)^{-\alpha_t}\exp(-\xi_t m), \quad m=\left|\log\left(\frac{K}{S}\right)\right|
\]

\[
V_{tail} = V_{deformed} \cdot (1 + 0.35\cdot clip(tail\_mult, 0, 1))
\]

### 5.6 Compression and Abstention

CHIMERA-RX computes:

1. Model code length \(L_M\)
2. Null code length \(L_0\)
3. Compression ratio \(CR_t=L_0/L_M\)

It calibrates `tau` through a frontier sweep to target:

1. Accuracy >= `0.80`
2. Coverage in `[0.35, 0.55]`

### 5.7 PTD (Regime Diagnostics)

PTD components:

1. Curvature proxy
2. Critical slowing proxy
3. Susceptibility proxy
4. Symmetry break proxy

Aggregated score:

\[
score = 0.33C + 0.25S + 0.22U + 0.20B
\]

Alert thresholds:

1. `NORMAL`: `score <= 0.52`
2. `ELEVATED`: `0.52 < score <= 0.72`
3. `STRESS`: `score > 0.72`

### 5.8 Quality Score

\[
Q_t = 10 \cdot (0.30A_t + 0.20R_t + 0.20N_t + 0.15G_t + 0.15S_t)
\]

Where:

1. `A_t`: accuracy
2. `R_t`: robustness
3. `N_t`: no-arbitrage integrity
4. `G_t`: regime lead
5. `S_t`: selective reliability

---

## 6. Inputs and Outputs

### 6.1 `CHIMERARXInput`

Required pricing fields:

1. `spot`
2. `strike`
3. `T`
4. `r`
5. `q`
6. `sigma`
7. `option_type`
8. `market_price`

State/context fields:

1. `vix`
2. `pcr`
3. `vrp_30d`
4. `hawkes_cluster`
5. `hurst`
6. `fii_net_flow_cr`
7. `geopolitical_proxy`
8. `climate_monsoon_proxy`

Friction proxies:

1. `margin_proxy`
2. `stt_itm_proxy`
3. `oi_cluster_proxy`
4. `gex_proxy`
5. `overnight_gap_proxy`

Optional references:

1. `nirv_price`
2. `omega_price`

Optional histories:

1. `surface_observations`
2. `ccas_history`
3. `ptd_timeline`

### 6.2 `CHIMERARXOutput`

Core pricing outputs:

1. `eq_price`
2. `deformed_price`
3. `tail_price`
4. `market_price`
5. `mispricing`

Governance outputs:

1. `abstention` (`CR_t`, `tau`, `action`, `abstain`)
2. `q_score` (`A_t,R_t,N_t,G_t,S_t,Q_t`)
3. `confidence`
4. `regime_alert`
5. `readiness`

Diagnostics outputs:

1. `friction`
2. `iv_surface_diag`
3. `frontier`
4. `ptd_timeline`
5. `diagnostics` map

---

## 7. Parameter Reference (Current Defaults)

### 7.1 Engine-Level Hyperparameters

| Parameter | Default | Meaning |
|---|---:|---|
| `tau` | `1.15` | Initial compression threshold input (final gate uses dynamically calibrated tau). |
| `alpha_t` | `3.0` | Tail power-law exponent control. |
| `xi_t` | `0.15` | Tail tempering coefficient. |
| `kappa` | `1.0` | Tail scaling denominator for moneyness term. |
| `lambda_ent` | `0.15` | KL regularization strength in equilibrium solver. |
| `lambda_arb` | `2.0` | No-arbitrage penalty strength in equilibrium solver. |

### 7.2 Equilibrium Solver Defaults

| Parameter | Default | Meaning |
|---|---:|---|
| `lr` | `0.12` | Mirror descent learning rate. |
| `max_iter` | `120` | Iteration cap. |
| `tol` | `1e-7` | Convergence threshold on simplex movement. |

### 7.3 Friction Coefficients

| Coefficient | Value |
|---|---:|
| `b1` Margin | `0.06` |
| `b2` STT ITM | `0.04` |
| `b3` OI Cluster | `0.05` |
| `b4` GEX | `0.03` |
| `b5` Overnight Gap | `0.08` |

### 7.4 PTD Thresholds

| Alert | Rule |
|---|---|
| `NORMAL` | `score <= 0.52` |
| `ELEVATED` | `0.52 < score <= 0.72` |
| `STRESS` | `score > 0.72` |

### 7.5 Readiness Gates

| Check | Condition |
|---|---|
| Quality gate | `Q_t >= 8.5` |
| No-arb gate | `eq_no_arb_term <= 0.02` |
| Selective accuracy gate | `actionable_accuracy >= 0.80` |
| Coverage gate | `0.35 <= coverage <= 0.55` |

---

## 8. How Accuracy Is Handled

### 8.1 Important Distinction

CHIMERA-RX currently provides two kinds of “accuracy”:

1. **In-run proxy accuracy** via current fit quality and selective gating.
2. **Actionable selective accuracy estimate** from historical `ccas_history` frontier.

It does **not yet constitute a full production-grade out-of-sample walk-forward proof** across all required stress windows in current code alone.

### 8.2 Current Quantitative Validation Status

Implementation/unit/non-regression status is validated:

1. `12` focused CHIMERA tests pass across:
   1. `tests/test_chimera_rx_engine.py`
   2. `tests/test_chimera_rx_ccas.py`
   3. `tests/test_chimera_rx_equilibrium.py`
   4. `tests/test_chimera_rx_ptd.py`
   5. `tests/test_chimera_rx_scoring.py`
   6. `tests/test_chimera_rx_surface_diag.py`
   7. `tests/test_chimera_rx_tab_smoke.py`
2. `1` CHIMERA isolation/non-regression test passes:
   1. `tests/test_omega_outputs_unchanged_with_chimera.py`
3. Full test collection is currently blocked by a syntax issue in:
   1. `tests/test_chimera_rx_behavioral_bridge.py`
4. Broader repo failures (outside CHIMERA core path) currently include:
   1. NIRV golden theta mismatches.
   2. NOVA `pipeline_status()` contract/type mismatch.

Predictive deployment-grade validation status:

1. Requires dedicated historical replay/backtest pipeline over specified event windows and transaction-cost assumptions.

### 8.3 Accuracy Metrics Used

CHIMERA-RX tracks:

1. `rel_error` (price fit)
2. `iv_surface_diag.summary.rmse_bp`
3. `frontier` (`tau`, `coverage`, `accuracy`)
4. `Q_t`
5. `readiness`

### 8.4 Practical Interpretation

1. High `CR_t` with high `Q_t` and `READY` status implies model claims strong structural fit.
2. High `Q_t` but poor coverage placement means gating policy needs recalibration.
3. Good fit with rising PTD score implies caution due transition risk.

---

## 9. Underlying Assumptions

CHIMERA-RX assumes:

1. A 3-state latent return simplex can approximate short-horizon option belief geometry.
2. Agent behavior can be summarized into five classes with weighted skewed beliefs.
3. Selected Indian friction proxies are meaningful and directionally consistent.
4. Compression ratio is a useful proxy for model applicability.
5. Selective prediction improves practical action quality over always-on prediction.
6. Current proxy mappings (state tensor and PTD components) are stable enough for shadow decision-support.

---

## 10. Limitations and Risks

1. The equilibrium is a compact proxy, not a full continuous manifold estimator.
2. PTD signals are engineered proxies, not fully structural topology estimators.
3. CCAS correctness depends on history quality and labeling of `correct`.
4. Surface diagnostics currently use synthesized strike/maturity scenarios around current point.
5. Real execution slippage and market-impact are not part of current CHIMERA-RX core.
6. Stress-window proof of the 80%/35-55% mandate is pending full replay framework.
7. Several coefficients are static defaults; without periodic calibration they can drift from live microstructure reality.
8. CHIMERA-RX is currently best used as a filter/governance layer over existing pricing outputs, not as a standalone alpha executor.

---

## 11. End-to-End Runtime Flow

1. Adapter reads session state and builds `CHIMERARXInput`.
2. Engine creates `StateTensor`.
3. Agent beliefs are generated and equilibrium solved.
4. Reflexive state updates from Hurst and Hawkes cluster.
5. Frictions deform price.
6. Tail correction is applied.
7. Surface diagnostics are computed.
8. Compression diagnostics and frontier are updated.
9. PTD signals and timeline are updated.
10. Quality score and readiness gates are computed.
11. Output payload is persisted in session state and rendered in tab.

---

## 12. UI and Monitoring Behavior

The CHIMERA-RX tab renders:

1. Core metrics (`Eq Price`, `Tail Price`, `Mispricing`, `Q Score`).
2. Gate metrics (`Compression Ratio`, `Action`, `Regime Alert`).
3. IV diagnostics table and summary.
4. Friction decomposition table.
5. Frontier table and effective coverage/accuracy metrics.
6. PTD timeline table.
7. Readiness checklist.
8. Output JSON contract.

Session state persistence:

1. `chimera_rx_latest`
2. `chimera_rx_ptd_timeline`
3. `chimera_rx_ccas_history`

---

## 13. Isolation and Safety Guarantees

Design guarantee:

1. CHIMERA-RX is additive and advisory.
2. No overwrite of OMEGA/NIRV/TVR outputs.
3. OMEGA non-regression test ensures imports do not alter OMEGA class behavior or feature flags.

---

## 14. Testing Coverage (Current)

Covered by tests:

1. Engine output contract and behavior consistency (`tests/test_chimera_rx_engine.py`).
2. Compression sensitivity and frontier monotonicity (`tests/test_chimera_rx_ccas.py`).
3. Equilibrium solver validity and no-arb penalty sanity (`tests/test_chimera_rx_equilibrium.py`).
4. PTD escalation and timeline rolling behavior (`tests/test_chimera_rx_ptd.py`).
5. Quality-score component behavior (`tests/test_chimera_rx_scoring.py`).
6. Surface diagnostics bucket/maturity coverage (`tests/test_chimera_rx_surface_diag.py`).
7. Tab smoke rendering and CHIMERA session-state persistence (`tests/test_chimera_rx_tab_smoke.py`).
8. OMEGA untouched behavior after CHIMERA imports (`tests/test_omega_outputs_unchanged_with_chimera.py`).

Known test friction:

1. `tests/test_chimera_rx_behavioral_bridge.py` currently contains trailing artifact text and fails collection with a syntax error.
2. Until fixed, treat full-suite CHIMERA coverage as partially blocked at collection stage.

---

## 15. Recommended Next Steps for Production-Grade Confidence

1. Fix `tests/test_chimera_rx_behavioral_bridge.py` so full-suite collection is restored.
2. Add historical replay datasets for the specified stress windows.
3. Compute rolling out-of-sample coverage/accuracy with transaction-cost assumptions.
4. Validate no-arb diagnostics across full chain snapshots, not only local diagnostics.
5. Add scenario-level reports: crisis vs calm vs transition.
6. Add calibration governance: hyperparameter drift bounds and scheduled recalibration.
7. Add model risk controls: alert fatigue controls and human-override workflow logs.

---

## 16. Glossary

1. **CCAS**: Causal Compression Abstention System.
2. **PTD**: Phase Transition Diagnostics.
3. **GEX**: Gamma Exposure proxy.
4. **Selective accuracy**: Accuracy measured only where model chooses to act.
5. **Coverage**: Fraction of opportunities where model acts.
6. **No-arb term**: Penalty representing monotonicity/convexity consistency pressure.

---

## 17. Final Practical Read

CHIMERA-RX is best understood as:

1. A **research-grade shadow decision system**.
2. A **structurally richer mispricing filter** than single-point calibration deltas.
3. A **selective, uncertainty-aware advisory layer** with explicit readiness gates.

In the current repository state, CHIMERA-RX is valuable for risk filtering, model-governance diagnostics, and regime-stress awareness. It becomes production-candidate only after full replay validation confirms target selective accuracy and coverage stability in real event windows.
