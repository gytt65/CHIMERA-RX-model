# Exhaustive Repository Analysis Report

Repository: `OPM-with-TVR-NIRV-OMEGA-Models-`
Generated on: 2026-02-28
Scope: Source + Tests + Docs + Configs + policy profile YAML, with runtime artifacts cataloged separately.

## Chapter 1: Repository Orientation

### 1.1 Mission and System Positioning
This repository is a quantitative decision-support system for Indian index options (primarily Nifty 50), centered on a layered architecture:
1. Physics-driven baseline pricing (NIRV and Quant Engine components).
2. Statistical residual correction and execution framing (OMEGA).
3. Frontier multi-stage orchestration (NOVA unified pipeline).
4. Additive advisory overlays (CHIMERA-RX and AI Council) designed to avoid mutating baseline pricing outputs.

The codebase is explicit about non-autonomous trading intent: outputs are research and manual execution support, with multiple safety gates around confidence, actionability, and data quality ([Repo-README], [Repo-Textbook]).

### 1.2 Runtime Modes and Major Operational Paths
The system supports a practical split between:
- Live/manual decision support through `opmAI_app.py`.
- Backtesting and scorecarding through `backtester.py` and `eval/*`.
- Historical-learning retraining through `historical_learning.py` plus `unified_pipeline.py` and `omega_model.py` training hooks.

Feature flags in `omega_features.py` provide a stable compatibility surface for enabling frontier modules gradually and safely ([Repo-Features]).

### 1.3 High-Level Stack (Plain Language)
- `opmAI_app.py`: user-facing control center and orchestration shell.
- `nirv_model.py`: regime-aware stochastic-volatility core pricer.
- `omega_model.py`: NIRV plus ML correction, behavioral/sentiment adjustments, and trade-planning layer.
- `unified_pipeline.py`: NOVA orchestrator chaining surface completion, no-arb fitting, jump/roughness modeling, neural correction, ensembles, conformal intervals, and hedging.
- `quant_engine.py`: reusable math toolkit (SABR, Heston COS, GARCH, Bayesian confidence, Kelly sizing, PDE solvers, flow analytics, and additional research modules).

### 1.4 File Taxonomy
- Core source modules: 75 non-test `.py` files.
- Validation and regression safety: 49 test `.py` files.
- Documentation: 12 docs markdown + 5 root markdown files.
- Runtime/config control: `requirements.txt`, `runtime.txt`, `devcontainer.json`, `config.env.example`, plus local-secret `config.env`.
- Policy profile: `ai_council/senate_profiles.yaml`.

## Chapter 2: System Architecture and Execution Flows

### 2.1 End-to-End User Flow
Primary runtime starts in `opmAI_app.py`, which wires data ingest, option context selection, pricing model calls, and tab-based workflows for TVR, NIRV, OMEGA, NOVA, historical learning, and evaluations.

Core pricing paths:
1. NIRV path: `opmAI_app.py` -> `NIRVModel.price_option()` in `nirv_model.py`.
2. OMEGA path: `opmAI_app.py` -> `OMEGAModel.price_option()` in `omega_model.py` -> NIRV baseline + ML and behavioral/sentiment layers.
3. NOVA path: `opmAI_app.py` -> `UnifiedPricingPipeline.price()` in `unified_pipeline.py`.
4. CHIMERA path (advisory): `opmAI_app.py` -> `chimera_rx_tab.py` -> `CHIMERARXEngine.run()`.
5. TVR path: implemented as embedded PDE pricer classes/functions inside `opmAI_app.py`.

### 2.2 Dependency Spine
The repository-level dependency map is coherent and intentionally additive:
- App routes into OMEGA, NIRV, Quant Engine, and feature flags.
- OMEGA depends on NIRV plus feature engineering and behavioral state.
- NIRV depends on surface and variance modules (`essvi_surface.py`, `model_free_variance.py`, `surface_checks.py`, router, VRP filters).
- Quant Engine provides lower-level model components used by NIRV and broader pipelines.
- Tests target all major layers, including regression and no-arbitrage behavior ([Repo-DependencyMap]).

### 2.3 Training Flow
Historical learning flow for OMEGA/NOVA:
1. Pull historical option and underlying data via `upstox_api_clients.py` into `historical_learning.py`.
2. Engineer features and residual labels against NIRV baseline.
3. Train/update OMEGA ML corrector artifacts.
4. Feed structured report into `UnifiedPricingPipeline.train_from_historical_report()` and `train_all()` to calibrate NOVA components (PINN, Neural J-SDE, KAN, Hawkes, Heston COS calibration state).

This flow exists to close cold-start gaps and align model corrections with real observed residuals, not synthetic assumptions ([Repo-HL], [Repo-NOVA]).

### 2.4 Evaluation Flow
Two main evaluation lanes:
- `backtester.py`: synthetic and real-data backtesting framework for model and ablation performance.
- `eval/*`: deterministic scorecards and model ranking harnesses (including OMEGA-specific claim checks and multi-model comparisons).

Golden snapshot tests in `tests/golden/*` protect numerical baseline behavior for NIRV under controlled settings.

### 2.5 Feature-Flag Control Plane
`omega_features.py` centralizes risk-controlled activation of upgrades. This enables:
- strict baseline reproducibility,
- progressive hardening (v5/v6/v7 transitions),
- and profile-level runtime tuning (macbook, accuracy-first, etc.) without refactoring core interfaces.

## Chapter 3: Core Model Deep Dives

### 3.1 NIRV (`nirv_model.py`)

#### Purpose
NIRV is the core physics model: a regime-aware, volatility-surface-informed pricer for Indian index options. It combines market-state features, regime detection, stochastic-volatility/jump pricing, confidence estimation, and chain scanning in a single orchestrator.

#### Internal Components
Major classes include:
- `IndiaFeatureEngine`: normalizes market inputs and builds feature context.
- `RegimeDetector`: discrete/continuous regime inference with optional HMM support.
- `VolatilitySurface`: SVI/eSSVI-aware IV surface logic with no-arbitrage checks.
- `HestonJumpDiffusionPricer`: Monte Carlo path pricer with regime-conditioned Heston and jumps.
- `BayesianConfidenceEngine`: risk-neutral and physical profitability/confidence estimation.
- `GreeksCalculator`, `MispricingSignal`, `EntropyEnsemble`, and top-level `NIRVModel`.

#### Interaction Map
- Upstream callers: `opmAI_app.py`, `omega_model.py`, `backtester.py`, `eval/run_all_model_scorecards.py`, tests.
- Downstream dependencies: `quant_engine.py`, `model_free_variance.py`, `essvi_surface.py`, `surface_checks.py`, `pricer_router.py`, `vrp_state.py`, `vrr_state.py`, `arbfree_surface.py`, `calibration/regime_conformal.py`, and optional meta modules.

#### Mathematical Core
NIRV combines:
- Heston stochastic volatility and jump diffusion extensions ([Lit-Heston-1993], [Lit-Merton-1976]).
- SVI/eSSVI smile parameterization and no-static-arbitrage controls ([Lit-SVI-Gatheral], [Lit-eSSVI]).
- Bayesian/bootstrap confidence logic for actionability.
- Optional roughness and entropy-based blending of method outputs.

#### Uniqueness and Rationale
The model is not "one formula"; it is a risk-managed pricing stack for non-stationary, microstructure-heavy market behavior. Its design rationale is to separate:
- physical pricing backbone,
- state adaptation,
- and post-pricing confidence/actionability gates.
This improves failure isolation compared with end-to-end black-box alternatives.

#### Strengths and Limits
Strengths:
- robust integration of regime, surface, and confidence layers,
- explicit fallback paths when optional dependencies are missing,
- strong regression test coverage.

Limits:
- complex state interactions raise calibration burden,
- optional paths can produce behavior divergence across feature profiles if not tightly governed.

### 3.2 OMEGA (`omega_model.py`)

#### Purpose
OMEGA is a layered intelligence wrapper on top of NIRV:
`NIRV base -> ML residual corrector -> anomaly/behavioral/sentiment adjustments -> actionability/trade plan`.

#### Internal Components
Key classes:
- `FeatureFactory`: feature extraction from market and NIRV context.
- `MLPricingCorrector`: residual learner with calibration and conformal interval support.
- `EfficiencyHunter`: anomaly/inefficiency diagnostics.
- `SentimentIntelligence`: LLM-driven sentiment scoring (provider-optional).
- `BehavioralEngine`, `PredictionTracker`, `TradePlanGenerator`.
- `ProspectTheoryKernel`, `DispositionFlowPredictor`, `EventRiskPricer`, `BehavioralLiquidityFeedback`, `ShadowHedger`.
- `OMEGAModel`: orchestrator.

#### Interaction Map
- Upstream callers: app tab workflows, eval harnesses, backtester paths.
- Downstream: `NIRVModel`, `behavioral_state_engine.py`, `historical_learning.py`, optional calibrators.

#### Mathematical Core
- Supervised residual modeling (tree ensembles or calibrated alternatives).
- Conformal interval estimation for distribution-free actionability bounds ([Lit-Conformal-2005]).
- Behavioral finance kernels (prospect-theory style asymmetry).
- Risk-aware position framing (including Kelly-style sizing ties via quant layer).

#### Uniqueness and Rationale
OMEGA explicitly uses a "physics-first, ML-second" principle: ML corrects systematic errors around a structural baseline rather than replacing it. This limits catastrophic extrapolation risk on sparse option data.

#### Strengths and Limits
Strengths:
- clear modular layering and tracker-based feedback,
- practical actionability gating,
- strong integration with historical-learning pipeline.

Limits:
- data quality dependence is high,
- sentiment layer quality can degrade under weak citation/provider reliability.

### 3.3 NOVA Unified Pipeline (`unified_pipeline.py`)

#### Purpose
NOVA is a multi-stage orchestrator that chains frontier modules into one pricing-and-uncertainty workflow.

#### Stages (as implemented)
- Surface completion (SGM).
- No-arbitrage surface fitting (PINN-style RBF constraints).
- Jump clustering (Hawkes estimator).
- Neural jump-SDE estimation.
- Fast/semi-analytical pricing augmentation (Heston COS, rough-vol routing support).
- KAN residual correction.
- Adaptive ensemble weighting.
- Conformal uncertainty bounds.
- Deep hedging outputs.
- Optional BSE validation and shadow-feedback gates.

#### Interaction Map
- Upstream callers: NOVA tab in app, scorecard harness.
- Downstream modules: `sgm_surface.py`, `pinn_vol_surface.py`, `hawkes_jump.py`, `neural_jsde.py`, `kan_corrector.py`, `ensemble_pricer.py`, `heston_cos.py`, `pricer_router.py`, `deep_hedging.py`, `vrp_state.py`, `behavioral_state_engine.py`, `cross_exchange_validator.py`.

#### Mathematical Core
- Score matching and Langevin dynamics for sparse IV completion.
- Constrained surface fitting with no-arbitrage and wing constraints.
- Hawkes self-exciting jump intensity ([Lit-Hawkes-1971]).
- State-conditional neural SDE parameterization.
- Online multiplicative-weights ensembles.
- Split/conformal uncertainty scaling.
- CVaR-aware hedging objective.

#### Uniqueness and Rationale
NOVA is notable for operationally wiring many research modules under a common interface instead of treating them as isolated notebooks.

#### Strengths and Limits
Strengths:
- explicit stage diagnostics and fallback logic,
- training hooks from historical reports,
- uncertainty-aware output.

Limits:
- high dimensional calibration and data-synchronization demands,
- potential runtime cost if multiple heavy modules are simultaneously enabled.

### 3.4 Quant Engine (`quant_engine.py`)

#### Purpose
Quant Engine is the shared mathematical toolkit used across NIRV/OMEGA/NOVA and research modules.

#### Major Families
- Volatility/smile: Dynamic SABR, GJR-GARCH, Heston COS.
- Risk and confidence: Bayesian posterior confidence, Kelly sizing.
- Flow and microstructure: GEX, inventory and execution modules.
- PDE and process modules: adaptive mesh PDE, Levy process pricer, contagion/coupling modules, copula/regime utilities.

#### Interaction Map
- Upstream callers: NIRV and app workflows, tests, backtester.
- Downstream dependencies: SciPy optimization/statistics stack, optional ML libs.

#### Mathematical Core
- SABR approximation and calibration ([Lit-SABR-Hagan-2002]).
- GJR-GARCH leverage-aware volatility forecasting ([Lit-GJR-1993], [Lit-Bollerslev-1986]).
- COS method for fast Fourier-based valuation ([Lit-COS-2008], [Lit-Albrecher-2007]).
- Bayesian updating and probability calibration.
- PDE finite-difference numerical methods.

#### Uniqueness and Rationale
Rather than embedding math directly into all higher-level models, the repository centralizes reusable quant methods here, reducing duplication and preserving comparability across pipelines.

### 3.5 Tiered Router and Rough Vol (`pricer_router.py`)

#### Purpose
CPU-budget-aware tiered routing among:
- very fast surface-IV BSM baseline,
- Heston COS refinement,
- selective MC/rough-vol routes where justified.

#### Mathematical Core
- Rough Bergomi-style volatility handling and live Hurst-aware adjustments ([Lit-rBergomi-2016], [Lit-RoughVol-2018]).
- Budget-constrained routing policy to balance speed and fidelity.

#### Uniqueness and Rationale
Practical tradeoff engine: it treats pricing-route choice as an engineering decision under latency constraints, not just a pure model-preference question.

### 3.6 Additional Model Engines and Their Roles

#### `heston_cos.py`
Fast Heston valuation using COS expansion with numerical-stability protections (Little Heston Trap form), calibration support, and Greek calculations.

#### `hawkes_jump.py`
Self-exciting jump-intensity estimation with MLE and fallback behavior when jump samples are sparse.

#### `neural_jsde.py`
State-conditioned neural parameterization of drift/diffusion/jump quantities, enabling context-adaptive pricing dynamics.

#### `kan_corrector.py`
Interpretable B-spline based residual corrector (KAN) as a structured alternative to opaque residual learners.

#### `pinn_vol_surface.py`
Constrained RBF surface fitting under no-arbitrage, wing, and martingale conditions.

#### `sgm_surface.py`
Score-based surface completion for sparse/illiquid option chains, reducing fragile extrapolation.

#### `model_free_variance.py` and `india_vix_synth.py`
Model-free variance/VIX-style estimators using OTM strips and maturity interpolation with microstructure filters.

#### `essvi_surface.py` and `arbfree_surface.py`
Surface parameterization and post-processing layers to maintain static no-arbitrage behavior.

#### `deep_hedging.py`
Hedging policy learning targeting PnL variance and tail-risk objectives.

#### `martingale_optimal_transport.py`
Linear-programming no-arbitrage bounds via martingale constraints; assumption-light sanity envelope for model outputs.

#### `path_signatures.py`
Rough-path inspired feature extraction for trajectory-level market-state representation.

#### `vrp_state.py` and `vrr_state.py`
Variance risk premium state extraction and compatibility logic for risk-regime modulation.

#### Embedded TVR in `opmAI_app.py`
Contains PDE-oriented TVR implementation used as a pricing path and cross-check framework in UI/backtest flows.

#### CHIMERA-RX suite
A full additive advisory engine (`chimera_rx_*`) with equilibrium, reflexive dynamics, compression-based abstention, transition diagnostics, and scoring. Design intent explicitly avoids mutating OMEGA/NIRV/TVR outputs ([Repo-CHIMERA]).

## Chapter 4: File-by-File Catalog (All Scoped Files)

Catalog format per file:
- File Name
- Purpose
- Interactions with other files
- Mathematical principles involved
- Unique features
- Key classes/functions
- Risk or maintenance notes

### 4.1 Source Catalog (75 non-test Python files)
- File Name: `ai_council/__init__.py` | Purpose: package export surface for AI Council modules. | Interactions: re-exports agents, alerts, orchestrator, worker types to app/import callers. | Mathematical principles involved: none directly; interface aggregation only. | Unique features: stable import facade to decouple internals. | Key classes/functions: module exports only. | Risk notes: low risk; keep exports synced with module evolution.
- File Name: `ai_council/agents.py` | Purpose: define persona agents and profile loading for council seats. | Interactions: uses `ai_council/types.py` and provider abstractions; consumed by orchestrator. | Mathematical principles involved: weighted evidence/opinion heuristics. | Unique features: senate profile loading with high-impact context handling. | Key classes/functions: `BaseAgent`, `PersonaAgent`, `build_senate_agents`, `load_senate_profiles`. | Risk notes: profile/schema drift can silently degrade opinion quality.
- File Name: `ai_council/alerts.py` | Purpose: publish council verdict alerts with priority thresholds. | Interactions: used by orchestrator/worker and optional Telegram routing. | Mathematical principles involved: rule-based priority gating. | Unique features: channel abstraction with fail-soft behavior. | Key classes/functions: `AlertConfig`, `CouncilAlertPublisher`. | Risk notes: external API failures should remain non-fatal.
- File Name: `ai_council/event_ingestion.py` | Purpose: normalize events/news and compute event shock summaries. | Interactions: consumed by council context construction and orchestrator cycles. | Mathematical principles involved: weighted scoring, sentiment-impact mapping. | Unique features: source-quality and staleness weighting. | Key classes/functions: `EventShock`, `compute_event_shock`, normalization helpers. | Risk notes: weak normalization rules can bias council direction.
- File Name: `ai_council/memory_store.py` | Purpose: SQLite persistence for council cycles, reliability, and outcomes. | Interactions: used by orchestrator and worker for calibration updates. | Mathematical principles involved: reliability calibration statistics storage. | Unique features: incremental reliability/evidence telemetry tables. | Key classes/functions: `CouncilMemoryStore`. | Risk notes: schema migration management is important for continuity.
- File Name: `ai_council/model_bridge.py` | Purpose: translate OMEGA/NIRV outputs into council-readable health payloads. | Interactions: ingests app/session model outputs; consumed by orchestrator. | Mathematical principles involved: disagreement scoring and health finding rules. | Unique features: model conflict and execution-quality diagnostics. | Key classes/functions: `ModelBridge`, `ModelHealthResult`. | Risk notes: contract mismatches can break council gating.
- File Name: `ai_council/orchestrator.py` | Purpose: run multi-agent deliberation, critiques, and final verdict assembly. | Interactions: depends on agents/types/providers/memory/alerts; called by worker and app paths. | Mathematical principles involved: weighted voting, probabilistic aggregation, reliability priors. | Unique features: hard-veto and reliability-aware arbitration. | Key classes/functions: `CouncilConfig`, `AICouncilOrchestrator`. | Risk notes: behavioral policy complexity requires tight regression tests.
- File Name: `ai_council/providers.py` | Purpose: provider wrappers, routing policy, payload validation, and citation parsing. | Interactions: used by agents/orchestrator for LLM calls. | Mathematical principles involved: probability normalization and routing heuristics. | Unique features: cost-tier aware provider routing with strict JSON parsing utilities. | Key classes/functions: `ProviderRoutingPolicy`, `LLMProvider`, `validate_payload`, `parse_first_json_object`. | Risk notes: provider response drift is a recurring maintenance vector.
- File Name: `ai_council/types.py` | Purpose: canonical dataclasses and literals for council contracts. | Interactions: shared by all AI Council modules and tests. | Mathematical principles involved: typed probabilistic forecast structures. | Unique features: strongly typed decision schema. | Key classes/functions: `CouncilContext`, `AgentOpinion`, `CouncilVerdict`, related dataclasses. | Risk notes: contract changes require synchronized updates across modules/tests.
- File Name: `ai_council/worker.py` | Purpose: background worker loop for periodic council cycles. | Interactions: wraps orchestrator, memory, alerting; integrated from app runtime. | Mathematical principles involved: trigger thresholds and dominant-signal heuristics. | Unique features: event-trigger logic on news/volatility state deltas. | Key classes/functions: `CouncilWorker`, `CouncilWorkerConfig`, worker registry helpers. | Risk notes: concurrency and shutdown behavior need careful handling.

- File Name: `arbfree_surface.py` | Purpose: arbitrage-free surface state container and repair checks. | Interactions: imported by NIRV and surface tests. | Mathematical principles involved: butterfly/calendar no-arbitrage constraints, interpolation. | Unique features: post-processing layer to preserve valid surface geometry. | Key classes/functions: `ArbFreeSurfaceState`. | Risk notes: calibration fallback behavior should avoid over-smoothing.
- File Name: `backtester.py` | Purpose: realistic simulation framework for model PnL and ablation tests. | Interactions: loads NIRV/OMEGA/TVR paths and synthetic/real market generators. | Mathematical principles involved: stochastic simulation, transaction-cost aware performance metrics. | Unique features: upgrade-level marginal alpha measurement. | Key classes/functions: `SyntheticNiftyGenerator`, `NirvBacktester`, `AblationAnalyzer`, `PerformanceReport`. | Risk notes: simulation assumptions can mislead if treated as live guarantees.
- File Name: `behavioral_agents.py` | Purpose: explicit behavioral-demand distortion model for IV shape effects. | Interactions: consumed by CHIMERA adapter and behavioral paths. | Mathematical principles involved: agent-mixture demand distortions and skew pressure. | Unique features: separate retail lottery vs institutional hedger channels. | Key classes/functions: `BehavioralAgents`. | Risk notes: requires empirical calibration; hardcoded multipliers can drift.
- File Name: `behavioral_state_engine.py` | Purpose: additive behavioral state builder for market microstructure context. | Interactions: used by OMEGA, NOVA, CHIMERA bridges. | Mathematical principles involved: normalized factor aggregation and clipping. | Unique features: backward-compatible optional payload design. | Key classes/functions: `BehavioralInputs`, `BehavioralStateEngine`. | Risk notes: input scaling consistency is important for downstream models.
- File Name: `bench/bench_pricing.py` | Purpose: latency benchmark script for pricing tiers. | Interactions: instantiates NIRV and measures throughput. | Mathematical principles involved: none new; performance instrumentation. | Unique features: simple reproducible CPU benchmark harness. | Key classes/functions: `run_bench`. | Risk notes: benchmark comparability depends on fixed environment settings.

- File Name: `calibration/__init__.py` | Purpose: calibration package marker. | Interactions: import namespace support. | Mathematical principles involved: none directly. | Unique features: lightweight package boundary. | Key classes/functions: package init only. | Risk notes: low.
- File Name: `calibration/confidence_calibrator.py` | Purpose: confidence calibration utilities for prediction reliability. | Interactions: used by OMEGA/NOVA/NIRV confidence paths. | Mathematical principles involved: probability calibration (logit/Platt-like transforms). | Unique features: reusable calibrator stats object. | Key classes/functions: `ConfidenceCalibrator`, `CalibrationStats`. | Risk notes: calibration data leakage must be avoided.
- File Name: `calibration/regime_conformal.py` | Purpose: split-conformal intervals bucketed by regime/moneyness/DTE. | Interactions: used by NIRV and reliability tests. | Mathematical principles involved: conformal prediction and conditional bucketing. | Unique features: regime-aware residual quantile storage. | Key classes/functions: `RegimeConformalIntervals`, `bucket_key`. | Risk notes: sparse buckets can degrade interval quality.

- File Name: `chimera_rx_ccas.py` | Purpose: causal-compression abstention diagnostics for CHIMERA. | Interactions: called by `chimera_rx_engine.py`. | Mathematical principles involved: MDL-like compression frontier and thresholding. | Unique features: abstain-vs-act gating from compression ratio statistics. | Key classes/functions: `CompressionDiagnostics`, `compute_compression_diagnostics`, `calibrate_tau`. | Risk notes: threshold calibration critical to avoid over-abstention.
- File Name: `chimera_rx_data_adapter.py` | Purpose: read-only adapter from app/session values into CHIMERA input contract. | Interactions: used by `chimera_rx_tab.py`; imports behavioral/structural helper modules. | Mathematical principles involved: deterministic feature extraction and normalization. | Unique features: bridge layer isolates UI schema from engine schema. | Key classes/functions: `build_chimera_rx_input`. | Risk notes: schema drift in session state can break adapter silently.
- File Name: `chimera_rx_engine.py` | Purpose: additive shadow-model orchestration for CHIMERA-RX outputs. | Interactions: composes equilibrium, reflexive, PTD, scoring, and state modules. | Mathematical principles involved: equilibrium pricing proxies, reflexive dynamics, scoring aggregation. | Unique features: explicit non-mutation contract versus OMEGA/NIRV/TVR outputs. | Key classes/functions: `CHIMERARXInput`, `CHIMERARXOutput`, `CHIMERARXEngine`. | Risk notes: maintain strict additive-only behavior.
- File Name: `chimera_rx_equilibrium.py` | Purpose: heterogeneous-belief equilibrium solver and no-arb penalty logic. | Interactions: consumed by CHIMERA engine and tests. | Mathematical principles involved: simplex normalization, divergence measures, no-arbitrage constraints. | Unique features: agent-belief construction tied to state tensor. | Key classes/functions: `AgentBelief`, `solve_equilibrium`, `equilibrium_option_price`. | Risk notes: numerical stability in constrained solves is key.
- File Name: `chimera_rx_ptd.py` | Purpose: phase-transition diagnostic scoring and alert timeline updates. | Interactions: uses reflexive state and tensor inputs; called by CHIMERA engine. | Mathematical principles involved: stress-score aggregation over state trajectories. | Unique features: rolling alert state machine. | Key classes/functions: `PTDSignals`, `compute_ptd_signals`, `update_ptd_timeline`. | Risk notes: threshold tuning can alter alert frequency sharply.
- File Name: `chimera_rx_reflexive.py` | Purpose: reflexive rough-Hawkes style state update and price adjustment proxy. | Interactions: used by PTD and engine. | Mathematical principles involved: recursive state dynamics with bounded transforms. | Unique features: compact reflexive proxy decoupled from primary pricer. | Key classes/functions: `ReflexiveState`, `update_reflexive_state`, `reflexive_price_adjustment`. | Risk notes: proxy interpretation must remain advisory-only.
- File Name: `chimera_rx_scoring.py` | Purpose: CHIMERA quality score aggregation into deployment-oriented score. | Interactions: consumed by CHIMERA engine and tests. | Mathematical principles involved: bounded weighted score composition on [0,10]. | Unique features: explicit dimension-level score decomposition. | Key classes/functions: `ModelQualityScore`, `compute_quality_score`. | Risk notes: weight assumptions should be versioned.
- File Name: `chimera_rx_state.py` | Purpose: state tensor construction from heterogeneous market inputs. | Interactions: used by equilibrium/reflexive/PTD/engine modules. | Mathematical principles involved: scaled transforms and signed-log compression. | Unique features: optional behavioral engine integration fallback-safe. | Key classes/functions: `StateTensor`, `build_state_tensor`. | Risk notes: input preprocessing assumptions influence all downstream scores.
- File Name: `chimera_rx_tab.py` | Purpose: Streamlit rendering layer for CHIMERA advisory outputs. | Interactions: uses data adapter and CHIMERA engine; called from `opmAI_app.py`. | Mathematical principles involved: none new; UI presentation layer. | Unique features: explicit shadow-mode user communication. | Key classes/functions: `render`. | Risk notes: UI contract changes can break adapter expectations.

- File Name: `cross_exchange_validator.py` | Purpose: cross-market signal sanity check (NSE vs BSE proxies). | Interactions: integrated into NOVA optional validation stage. | Mathematical principles involved: correlation-informed cross-validation heuristics. | Unique features: data-artifact guard using parallel market evidence. | Key classes/functions: `CrossExchangeValidator`. | Risk notes: mapping assumptions can become stale with contract changes.
- File Name: `deep_hedging.py` | Purpose: learn hedging policy minimizing variance and tail risk with costs. | Interactions: used by NOVA pipeline; evaluated in backtests/audits. | Mathematical principles involved: supervised optimization of PnL^2 + CVaR-like objective. | Unique features: surface-informed hedge features beyond delta-only hedging. | Key classes/functions: `HedgingNetwork`, `DeepHedger`. | Risk notes: optimizer choice and calibration data quality are critical.
- File Name: `ensemble_pricer.py` | Purpose: online adaptive model-combination engine. | Interactions: used by NOVA pipeline with NIRV/neural/KAN model outputs. | Mathematical principles involved: multiplicative weights/exponential hedging updates. | Unique features: cold-start vs adaptive weighting modes. | Key classes/functions: `EnsemblePricer`. | Risk notes: sparse observation windows can cause unstable weights.
- File Name: `essvi_surface.py` | Purpose: lightweight eSSVI surface implementation with safer bounds. | Interactions: used by NIRV surface path and surface tests. | Mathematical principles involved: eSSVI parameterization and static no-arbitrage checks. | Unique features: robust bounded slices for practical calibration. | Key classes/functions: `ESSVISurface`. | Risk notes: fallback behavior must avoid silent regime distortions.

- File Name: `eval/metrics.py` | Purpose: evaluation metrics library for pricing and surface quality. | Interactions: consumed by eval runners and scorecard builders. | Mathematical principles involved: RMSE/IV RMSE, arbitrage-violation counts, stability metrics. | Unique features: vega-weighted error support. | Key classes/functions: `vega_weighted_rmse`, `arbitrage_violation_counts`. | Risk notes: metric definitions must remain stable across report versions.
- File Name: `eval/omega_scorecard.py` | Purpose: deterministic OMEGA claim-evaluation scorecard (100-point). | Interactions: reads prediction logs and runs robustness/economic checks. | Mathematical principles involved: calibration, directional accuracy, score aggregation. | Unique features: explicit 95+ claim eligibility gate. | Key classes/functions: `build_scorecard`, component scorers. | Risk notes: score inflation risk if assumptions are changed ad hoc.
- File Name: `eval/run.py` | Purpose: CLI runner for evaluation harness on snapshots and features. | Interactions: imports NIRV and eval metrics; called from command line. | Mathematical principles involved: deterministic harness orchestration. | Unique features: lightweight reproducible entrypoint. | Key classes/functions: `run_evaluation`. | Risk notes: CLI defaults should be documented for reproducibility.
- File Name: `eval/run_all_model_scorecards.py` | Purpose: canonical multi-model evaluator and markdown/json scorecard writer. | Interactions: runs TVR/NIRV/OMEGA/NOVA/CHIMERA paths and report writers. | Mathematical principles involved: multi-criteria scoring, cost-adjusted edge checks. | Unique features: unified cross-model ranking output. | Key classes/functions: `ModelEvalRow`, `run_all`, aggregation helpers. | Risk notes: ensure consistent model configs across comparisons.

- File Name: `hawkes_jump.py` | Purpose: Hawkes jump process estimation for clustering jump risk. | Interactions: used by Quant Engine and NOVA pipeline. | Mathematical principles involved: self-exciting point processes, Ogata thinning, MLE. | Unique features: VIX-modulated intensity path and fallback to Poisson. | Key classes/functions: `HawkesProcess`, `HawkesJumpEstimator`. | Risk notes: sparse jump samples can destabilize fit.
- File Name: `heston_cos.py` | Purpose: fast semi-analytical Heston pricer and calibrator. | Interactions: used directly in NOVA and indirectly in quant/NIRV validation paths. | Mathematical principles involved: characteristic function valuation via COS expansion. | Unique features: numerically stable little-Heston-trap formulation and vectorized pricing. | Key classes/functions: `HestonCOSPricer`, calibration and Greeks helpers. | Risk notes: calibration bounds should be checked for regime outliers.
- File Name: `historical_learning.py` | Purpose: pull historical data, engineer labels/features, train/update OMEGA corrector. | Interactions: uses Upstox clients, NIRV baseline, OMEGA feature/corrector utilities, NOVA training interface. | Mathematical principles involved: time-series splits, residual learning, walk-forward evaluation. | Unique features: additive training pipeline integrated with existing runtime models. | Key classes/functions: `HistoricalLearningConfig`, `pull_historical_option_data`, `train_or_update_ml_corrector`, `pull_and_train`. | Risk notes: timestamp alignment and missing-spot handling are high-impact.
- File Name: `india_vix_synth.py` | Purpose: synthetic India VIX computation wrapper/compatibility route. | Interactions: delegates to model-free variance engine when enabled. | Mathematical principles involved: VIX-style strip integration and maturity interpolation. | Unique features: legacy compatibility shim around newer engine. | Key classes/functions: `compute_synthetic_vix`, helper estimators. | Risk notes: keep parity between legacy and new route behavior.
- File Name: `iv_solver.py` | Purpose: high-precision implied volatility inversion (Jaeckel-style). | Interactions: used in pipeline calibration and variance processing helpers. | Mathematical principles involved: rational approximations and Householder iterations. | Unique features: robust convergence across moneyness/tenor regions. | Key classes/functions: `bs_implied_vol`, `implied_total_vol`, derivative helpers. | Risk notes: numerical guardrails should remain strict near expiry/zero prices.
- File Name: `kan_corrector.py` | Purpose: KAN residual-correction network for pricing adjustments. | Interactions: used by NOVA pipeline and optional training flows. | Mathematical principles involved: Kolmogorov-Arnold decomposition with B-spline edges, SPSA/Adam optimization. | Unique features: interpretable edge-level learned functions. | Key classes/functions: `KANLayer`, `KANCorrector`. | Risk notes: sensitivity to feature scaling and knot configuration.
- File Name: `market_conventions.py` | Purpose: NSE-specific trading-time and option convention helpers. | Interactions: used by pricing/time-to-expiry logic and tests. | Mathematical principles involved: calendar-time to year-fraction conversion. | Unique features: India-market convention centralization. | Key classes/functions: `time_to_expiry`, `is_index_option`, `EuropeanOptionRouter`. | Risk notes: calendar assumptions must track exchange rule changes.
- File Name: `martingale_optimal_transport.py` | Purpose: model-free pricing bounds under martingale constraints. | Interactions: optional support to NIRV/research diagnostics. | Mathematical principles involved: linear programming for martingale OT bounds. | Unique features: assumption-light no-arbitrage bound checks. | Key classes/functions: `MartingaleOptimalTransport`. | Risk notes: computational scaling and discretization choices matter.

- File Name: `meta/__init__.py` | Purpose: meta-model package marker. | Interactions: namespace support. | Mathematical principles involved: none directly. | Unique features: package boundary only. | Key classes/functions: none. | Risk notes: low.
- File Name: `meta/residual_stacker.py` | Purpose: regime-aware residual stacking meta-model. | Interactions: optional import in NIRV residual pathways. | Mathematical principles involved: stacked residual correction and regime conditioning. | Unique features: meta-layer around baseline residual errors. | Key classes/functions: `ResidualSample`, `ResidualStacker`. | Risk notes: avoid target leakage in stacked training.

- File Name: `model_free_variance.py` | Purpose: NSE-consistent model-free variance and VIX computation engine. | Interactions: used by NIRV and legacy VIX wrappers; tested extensively. | Mathematical principles involved: variance swap strip approximation, forward extraction, 30-day interpolation. | Unique features: microstructure filters and optional tail correction. | Key classes/functions: `estimate_forward_from_chain`, `compute_variance_for_expiry`, `compute_vix_30d`. | Risk notes: quote-quality filtering thresholds are critical.
- File Name: `neural_jsde.py` | Purpose: neural jump-SDE pricing module with state-dependent parameters. | Interactions: used by NOVA pipeline training/pricing stages. | Mathematical principles involved: stochastic differential equations with neural parameter maps. | Unique features: numpy-only implementation preserving low dependency footprint. | Key classes/functions: `NeuralParameterNetwork`, `NeuralJSDE`. | Risk notes: calibration stability depends on feature normalization.
- File Name: `nirv_model.py` | Purpose: NIRV master pricing model and chain scanner. | Interactions: central dependency for OMEGA, app, backtester, eval, historical learning. | Mathematical principles involved: Heston+jumps, SVI/eSSVI, Bayesian confidence, regime modeling. | Unique features: India-specific regime/market-state architecture with extensive fallbacks. | Key classes/functions: `NIRVModel`, `HestonJumpDiffusionPricer`, `VolatilitySurface`, others. | Risk notes: high-complexity orchestrator; profile-specific behavior must stay tested.
- File Name: `nse_specs.py` | Purpose: contract spec resolver for expiries, lot sizes, TTE computations. | Interactions: used by pricing and contract-resolution tests. | Mathematical principles involved: time/calendar transformations and deterministic rule logic. | Unique features: Tuesday-centric fallback with holiday rollback support. | Key classes/functions: `ContractSpecResolver`, `time_to_expiry_minutes`. | Risk notes: stale contract masters can propagate lot/expiry errors.
- File Name: `omega_features.py` | Purpose: central feature-flag dataclass and profile presets. | Interactions: imported across NIRV/OMEGA/NOVA/QuantEngine/app/tests. | Mathematical principles involved: none directly; governance layer. | Unique features: singleton state + profile convenience methods. | Key classes/functions: `OmegaFeatures`, `get_features`, `set_features`. | Risk notes: global singleton misuse can cause test contamination.
- File Name: `omega_model.py` | Purpose: OMEGA layered intelligence model over NIRV baseline. | Interactions: uses NIRV, behavioral state, historical learning, optional calibration modules. | Mathematical principles involved: residual ML, conformal intervals, behavioral kernels. | Unique features: full trade-plan and actionability orchestration with trackers. | Key classes/functions: `OMEGAModel`, `FeatureFactory`, `MLPricingCorrector`, others. | Risk notes: data quality and calibration cadence dominate reliability.
- File Name: `opmAI_app.py` | Purpose: Streamlit application entrypoint with all tabs and orchestration. | Interactions: wires all major models, APIs, training, evaluation, AI council, and CHIMERA tab. | Mathematical principles involved: hosts TVR PDE implementation and uses outputs from quant engines. | Unique features: single operational cockpit spanning research to live decision support. | Key classes/functions: `TVRAmericanOptionPricer`, cache/data helpers, multiple model runners. | Risk notes: very large file; refactor pressure and UI/runtime coupling risk.
- File Name: `path_signatures.py` | Purpose: path-signature feature extraction utilities (rough-path style). | Interactions: used by frontier modeling paths where available. | Mathematical principles involved: rough path signatures and lead-lag transforms. | Unique features: optional dependency fallback when signature libs absent. | Key classes/functions: `PathSignatureExtractor`. | Risk notes: approximation mode differs from true library-backed signatures.
- File Name: `pinn_vol_surface.py` | Purpose: constrained RBF volatility surface with no-arbitrage physics losses. | Interactions: used by NOVA pipeline and surface diagnostics. | Mathematical principles involved: constrained optimization with butterfly/calendar/wing/martingale penalties. | Unique features: multi-constraint practical PINN-style implementation without heavy DL frameworks. | Key classes/functions: `RBFNetwork`, `PINNVolSurface`. | Risk notes: penalty weights and calibration stability require monitoring.
- File Name: `pricer_router.py` | Purpose: tiered CPU-budgeted routing and rough-vol pricer support. | Interactions: used by NIRV and NOVA paths; tested independently. | Mathematical principles involved: route selection and rough-vol Monte Carlo. | Unique features: latency-aware pricing-route governance. | Key classes/functions: `RBergomiPricer`, `TieredPricerRouter`. | Risk notes: routing thresholds should be profile-specific and tested.
- File Name: `python3 -m py_compile opmAI_app.py` | Purpose: accidental zero-byte artifact with `.py` suffix. | Interactions: none intended. | Mathematical principles involved: none. | Unique features: filename indicates mistaken command redirection/event. | Key classes/functions: none (empty file). | Risk notes: should be removed or ignored to avoid tooling confusion.
- File Name: `quant_engine.py` | Purpose: centralized quantitative-method library and extended research modules. | Interactions: consumed by NIRV/app and evaluation/testing ecosystems. | Mathematical principles involved: SABR, GJR-GARCH, Heston COS, Bayesian confidence, Kelly, PDE, Levy, copula, contagion, transfer entropy, more. | Unique features: broad reusable quant stack in one module. | Key classes/functions: `QuantEngine` and many domain classes. | Risk notes: module size is high; partitioning could reduce maintenance risk.

- File Name: `reports/__init__.py` | Purpose: report package marker. | Interactions: import namespace support for report writers. | Mathematical principles involved: none directly. | Unique features: lightweight package boundary. | Key classes/functions: none. | Risk notes: low.
- File Name: `reports/model_scorecard_report.py` | Purpose: scorecard markdown/json rendering utilities. | Interactions: called by eval runners. | Mathematical principles involved: none new; formatting and persistence layer. | Unique features: deterministic markdown rendering for scorecards. | Key classes/functions: `_render_md`, `write_scorecard_report`. | Risk notes: template changes can break downstream report parsers.
- File Name: `scripts/regenerate_snapshots.py` | Purpose: regenerate golden snapshots for regression baseline updates. | Interactions: imports NIRV and writes test snapshot fixtures. | Mathematical principles involved: deterministic scenario replay. | Unique features: reproducibility instructions embedded. | Key classes/functions: `regenerate`. | Risk notes: should only be run during controlled baseline updates.
- File Name: `sgm_surface.py` | Purpose: score-matching based IV surface completion for sparse quotes. | Interactions: used by NOVA pipeline before PINN fitting. | Mathematical principles involved: KDE score matching and Langevin dynamics. | Unique features: non-parametric infill alternative to fixed functional extrapolation. | Key classes/functions: `ScoreBasedSurfaceCompleter`. | Risk notes: bandwidth/noise schedules influence stability.
- File Name: `structural_frictions.py` | Purpose: deterministic market-structure friction model (fees/margin/tax effects). | Interactions: used in behavioral/advisory contexts including CHIMERA bridge. | Mathematical principles involved: transaction-cost/friction adjustments. | Unique features: explicit Indian regulatory friction placeholders. | Key classes/functions: `StructuralFrictions`. | Risk notes: policy updates can quickly stale constants.
- File Name: `surface_checks.py` | Purpose: no-arbitrage diagnostics for volatility and call-price surfaces. | Interactions: used by NIRV/ESSVI/arbfree modules and tests. | Mathematical principles involved: butterfly, calendar, convexity monotonicity conditions. | Unique features: lightweight reusable diagnostics functions. | Key classes/functions: `check_butterfly_arbitrage_slice`, `check_calendar_arbitrage`, `check_monotonicity_and_convexity_of_call_prices`. | Risk notes: tolerance choices affect false-positive/false-negative balance.
- File Name: `surface_shock.py` | Purpose: scaffold for generative surface-shock modeling. | Interactions: currently placeholder module with tests. | Mathematical principles involved: scenario sampling abstractions for surface shocks. | Unique features: interface-first design for future VAE/GAN-like upgrades. | Key classes/functions: `SurfaceShockModel`. | Risk notes: placeholder behavior should not be mistaken for production model.
- File Name: `svi_fixed_point.py` | Purpose: fixed-point warm-start helper for SVI calibration. | Interactions: optional path used by NIRV surface calibration. | Mathematical principles involved: iterative fixed-point initialization for nonlinear optimization. | Unique features: targeted warm-start to improve convergence stability. | Key classes/functions: `fixed_point_svi_warmstart`. | Risk notes: must remain compatible with primary SVI objective.
- File Name: `test_phase2.py` | Purpose: legacy smoke test script for phase-2 upgrades. | Interactions: imports NIRV/OMEGA/Quant Engine components outside pytest suite. | Mathematical principles involved: synthetic sanity checks for feature modules. | Unique features: standalone script mode. | Key classes/functions: multiple `test_*` helpers in script style. | Risk notes: script-style tests can diverge from pytest truth over time.
- File Name: `test_upgrades.py` | Purpose: legacy smoke test script for upgrade bundles. | Interactions: imports key modules and runs scripted checks. | Mathematical principles involved: broad sanity checks across model upgrades. | Unique features: explicit import fallback logic. | Key classes/functions: scripted `test_*` functions. | Risk notes: duplicates pytest coverage and may become stale.

- File Name: `training/__init__.py` | Purpose: training package marker. | Interactions: namespace support. | Mathematical principles involved: none directly. | Unique features: package boundary only. | Key classes/functions: none. | Risk notes: low.
- File Name: `training/dataset_builder.py` | Purpose: build leak-safe training datasets from chain snapshots. | Interactions: used by training workflows and potential future automation. | Mathematical principles involved: leakage-safe time-series feature assembly. | Unique features: explicit build config for reproducibility. | Key classes/functions: `DatasetBuildConfig`, `build_dataset_from_rows`. | Risk notes: split logic must stay aligned with walk-forward runner.
- File Name: `training/hpo_optuna.py` | Purpose: hyperparameter optimization runner with fallback mode. | Interactions: can optimize training modules where enabled. | Mathematical principles involved: search-based optimization and trial evaluation. | Unique features: optional Optuna dependency with graceful fallback. | Key classes/functions: `TrialResult`, `HPORunner`. | Risk notes: uncontrolled search can overfit without strict validation.
- File Name: `training/walkforward_runner.py` | Purpose: purged walk-forward split and execution helpers. | Interactions: used by training/evaluation workflows for temporal robustness. | Mathematical principles involved: purged/embargoed time-series cross-validation. | Unique features: explicit leakage controls for temporal data. | Key classes/functions: `WalkForwardConfig`, `purged_walkforward_splits`, `run_walkforward`. | Risk notes: incorrect purge/embargo settings can reintroduce leakage.

- File Name: `unified_pipeline.py` | Purpose: NOVA end-to-end orchestrator and training manager. | Interactions: imports frontier modules and integrates with app/eval/training flows. | Mathematical principles involved: multi-stage hybrid modeling, conformal intervals, ensemble adaptation, hedging. | Unique features: unified stage diagnostics and shadow-feedback gating. | Key classes/functions: `UnifiedPricingPipeline`, `_estimate_hurst`, `price`, `train_all`, `train_from_historical_report`. | Risk notes: high integration complexity requires strong regression controls.
- File Name: `upstox_api_clients.py` | Purpose: typed API wrappers for Upstox data ingestion. | Interactions: used by historical learning and app data pull paths. | Mathematical principles involved: none directly; data-contract validation. | Unique features: robust parameter validation and payload normalization. | Key classes/functions: `UpstoxAPIClients`, `UpstoxAPIError`, builder/validators. | Risk notes: upstream API schema changes are a recurring risk.
- File Name: `vrp_state.py` | Purpose: model-free VRP term-structure state estimator. | Interactions: used by NIRV and NOVA for parameter modulation. | Mathematical principles involved: implied vs realized variance spread state extraction. | Unique features: deterministic state labeling and adjustment helpers. | Key classes/functions: `ModelFreeVRPState`. | Risk notes: realized-variance estimation windows affect state stability.
- File Name: `vrr_state.py` | Purpose: legacy VRR/VRP filter retained for compatibility and fallback. | Interactions: optional import in NIRV legacy path and tests. | Mathematical principles involved: variance risk ratio state transitions and parameter multipliers. | Unique features: backward-compatible risk-aversion state abstraction. | Key classes/functions: `VRRStateFilter`. | Risk notes: keep distinction clear vs newer `vrp_state.py` path.

### 4.2 Test Catalog (49 test Python files)

- File Name: `tests/__init__.py` | Purpose: test package marker. | Interactions: enables package imports for tests. | Mathematical principles involved: none. | Unique features: minimal package scaffold. | Key classes/functions: none. | Risk notes: low.
- File Name: `tests/golden/__init__.py` | Purpose: golden-test package marker. | Interactions: namespace support for golden suite. | Mathematical principles involved: none. | Unique features: minimal scaffold. | Key classes/functions: none. | Risk notes: low.
- File Name: `tests/golden/generate_golden_snapshots.py` | Purpose: generate deterministic golden snapshot fixtures. | Interactions: imports NIRV and writes snapshot JSONs. | Mathematical principles involved: reproducible Monte Carlo baseline control. | Unique features: reproducibility strategy documented in script. | Key classes/functions: `regenerate` flow in script. | Risk notes: run only during controlled baseline refresh.
- File Name: `tests/golden/test_golden_nirv_outputs.py` | Purpose: freeze baseline NIRV outputs against golden fixtures. | Interactions: imports NIRV and reads `tests/golden/snapshots/*.json`. | Mathematical principles involved: regression tolerance checks for stochastic-model outputs. | Unique features: baseline-preservation contract when feature flags are OFF. | Key classes/functions: golden pytest tests. | Risk notes: fixture drift management is critical.

- File Name: `tests/test_ai_council_alerts.py` | Purpose: validate alert publishing priority logic. | Interactions: tests `ai_council/alerts.py`. | Mathematical principles involved: threshold gating logic. | Unique features: verifies Telegram optional behavior. | Key classes/functions: alert priority tests. | Risk notes: ensure policy changes are reflected.
- File Name: `tests/test_ai_council_event_ingestion.py` | Purpose: test event normalization and shock scoring. | Interactions: tests `ai_council/event_ingestion.py`. | Mathematical principles involved: weighted shock scoring and staleness effects. | Unique features: source quality sensitivity checks. | Key classes/functions: event-ingestion tests. | Risk notes: taxonomy updates need matching tests.
- File Name: `tests/test_ai_council_memory_store.py` | Purpose: validate extended reliability/outcome SQLite operations. | Interactions: tests `ai_council/memory_store.py`. | Mathematical principles involved: persistence of calibration metrics. | Unique features: temporary DB lifecycle test coverage. | Key classes/functions: memory store table operations tests. | Risk notes: schema migrations should keep tests green.
- File Name: `tests/test_ai_council_model_bridge.py` | Purpose: verify model conflict/execution health detection. | Interactions: tests `ai_council/model_bridge.py`. | Mathematical principles involved: disagreement scoring heuristics. | Unique features: conflict and stale-data finding assertions. | Key classes/functions: bridge health tests. | Risk notes: model-output schema drift can break expectations.
- File Name: `tests/test_ai_council_orchestrator.py` | Purpose: test deliberation orchestration and critique flow. | Interactions: tests `ai_council/orchestrator.py` with static agents. | Mathematical principles involved: weighted voting and critique severity logic. | Unique features: deterministic stub-agent orchestration tests. | Key classes/functions: orchestrator scenario tests. | Risk notes: keep deterministic mocks for reproducibility.
- File Name: `tests/test_ai_council_profiles.py` | Purpose: verify senate profile completeness and required seats. | Interactions: tests profile loading from agents/yaml. | Mathematical principles involved: none. | Unique features: minimum-seat integrity checks. | Key classes/functions: profile load test. | Risk notes: update when seat roster changes.
- File Name: `tests/test_ai_council_provider_utils.py` | Purpose: validate provider payload parsing and normalization. | Interactions: tests `ai_council/providers.py`. | Mathematical principles involved: probability normalization constraints. | Unique features: noisy-text JSON extraction test. | Key classes/functions: provider utility tests. | Risk notes: parser changes require robust regression checks.
- File Name: `tests/test_ai_council_routing.py` | Purpose: test cost-mode provider routing policy. | Interactions: tests `ProviderRoutingPolicy`. | Mathematical principles involved: constrained optimization heuristics. | Unique features: high-impact override behavior coverage. | Key classes/functions: routing selection tests. | Risk notes: cost tiers and priorities must remain consistent.
- File Name: `tests/test_ai_council_worker.py` | Purpose: verify worker trigger logic and event sensitivity. | Interactions: tests `ai_council/worker.py` + orchestrator integration. | Mathematical principles involved: threshold-based state-change triggers. | Unique features: high-impact news trigger scenarios. | Key classes/functions: worker trigger tests. | Risk notes: asynchronous assumptions should stay deterministic.

- File Name: `tests/test_arbfree_integration.py` | Purpose: integration check for arb-free surface path in NIRV. | Interactions: tests NIRV + `arbfree_surface.py`. | Mathematical principles involved: no-arbitrage surface embedding in pricing path. | Unique features: flag-gated integration behavior test. | Key classes/functions: integration tests. | Risk notes: feature-flag defaults must remain explicit.
- File Name: `tests/test_arbfree_surface.py` | Purpose: unit tests for `ArbFreeSurfaceState` behavior. | Interactions: tests interpolation and arbitrage detection logic. | Mathematical principles involved: convexity/calendar constraints. | Unique features: focused surface-state correctness checks. | Key classes/functions: `TestArbFreeSurface` methods. | Risk notes: tolerance tuning can affect pass/fail.
- File Name: `tests/test_backtest_no_lookahead.py` | Purpose: ensure synthetic generator avoids look-ahead leakage. | Interactions: tests `SyntheticNiftyGenerator` in backtester. | Mathematical principles involved: time-series causality constraints. | Unique features: determinism under varying horizon lengths. | Key classes/functions: no-lookahead test. | Risk notes: sequence-building changes should preserve causality.
- File Name: `tests/test_behavioral_state_engine.py` | Purpose: verify shape/bounds of behavioral state output. | Interactions: tests `behavioral_state_engine.py`. | Mathematical principles involved: normalized factor bounds. | Unique features: comprehensive input vector sanity checks. | Key classes/functions: output-shape test. | Risk notes: scaling changes require updating expected bounds.

- File Name: `tests/test_chimera_rx_behavioral_bridge.py` | Purpose: validate CHIMERA behavioral bridge and identifiability controls. | Interactions: tests CHIMERA engine/equilibrium/state modules. | Mathematical principles involved: belief/equilibrium stability under behavioral shifts. | Unique features: multi-input stress assertions. | Key classes/functions: bridge scenario tests. | Risk notes: ensure advisory-only invariants remain true.
- File Name: `tests/test_chimera_rx_ccas.py` | Purpose: test compression diagnostics and threshold monotonicity. | Interactions: tests `chimera_rx_ccas.py`. | Mathematical principles involved: compression ratio frontier behavior. | Unique features: monotone coverage checks over threshold grid. | Key classes/functions: CCAS tests. | Risk notes: calibration logic changes can alter expected monotonicity.
- File Name: `tests/test_chimera_rx_engine.py` | Purpose: smoke-test CHIMERA output contract. | Interactions: tests `CHIMERARXEngine.run`. | Mathematical principles involved: bounded scoring and output consistency. | Unique features: contract-level smoke gate. | Key classes/functions: engine smoke tests. | Risk notes: output schema changes should be versioned.
- File Name: `tests/test_chimera_rx_equilibrium.py` | Purpose: test equilibrium solver validity and simplex behavior. | Interactions: tests `chimera_rx_equilibrium.py` + state builder. | Mathematical principles involved: constrained equilibrium and no-arb penalties. | Unique features: simplex validity assertions. | Key classes/functions: equilibrium tests. | Risk notes: numeric tolerance sensitivity.
- File Name: `tests/test_chimera_rx_ptd.py` | Purpose: verify PTD stress escalation behavior. | Interactions: tests `chimera_rx_ptd.py` + reflexive state updates. | Mathematical principles involved: transition diagnostics under stressed state vectors. | Unique features: low-vs-high stress comparative checks. | Key classes/functions: PTD escalation tests. | Risk notes: threshold adjustments require test updates.
- File Name: `tests/test_chimera_rx_scoring.py` | Purpose: test CHIMERA quality score bounds and monotonicity. | Interactions: tests `chimera_rx_scoring.py`. | Mathematical principles involved: bounded weighted score composition. | Unique features: score-improves-with-accuracy checks. | Key classes/functions: scoring tests. | Risk notes: keep score semantics stable over versions.
- File Name: `tests/test_chimera_rx_surface_diag.py` | Purpose: verify CHIMERA surface diagnostics include expected buckets/maturities. | Interactions: tests `chimera_rx_engine.py`. | Mathematical principles involved: categorical surface diagnostics sanity checks. | Unique features: ATM/OTM/DEEP_OTM coverage check. | Key classes/functions: surface diagnostic test. | Risk notes: diagnostic schema changes must be coordinated.
- File Name: `tests/test_chimera_rx_tab_smoke.py` | Purpose: UI smoke test for CHIMERA tab renderer with dummy Streamlit object. | Interactions: tests `chimera_rx_tab.py`. | Mathematical principles involved: none. | Unique features: UI contract tested without full Streamlit runtime. | Key classes/functions: render smoke test. | Risk notes: dummy API should track real Streamlit usage.

- File Name: `tests/test_confidence_engine_stability.py` | Purpose: ensure confidence engine handles non-finite terminal paths. | Interactions: tests `BayesianConfidenceEngine` in NIRV. | Mathematical principles involved: robust statistics under contaminated samples. | Unique features: NaN/Inf stress inputs. | Key classes/functions: confidence stability test. | Risk notes: maintain non-finite guards in engine code.
- File Name: `tests/test_conformal_intervals.py` | Purpose: validate conformal interval behavior on toy distributions. | Interactions: tests OMEGA `MLPricingCorrector` conformal logic. | Mathematical principles involved: empirical coverage properties of conformal quantiles. | Unique features: synthetic large-sample coverage setup. | Key classes/functions: conformal interval tests. | Risk notes: calibration method changes need careful re-baselining.
- File Name: `tests/test_contract_spec_resolver_lot_and_expiry.py` | Purpose: verify contract resolver lot-size and expiry handling. | Interactions: tests `nse_specs.py`. | Mathematical principles involved: calendar/date rule correctness. | Unique features: toy calendar with holiday edge cases. | Key classes/functions: resolver/time-to-expiry tests. | Risk notes: exchange-rule changes should be reflected quickly.
- File Name: `tests/test_feature_profiles.py` | Purpose: test feature profile presets and singleton reset behavior. | Interactions: tests `omega_features.py`. | Mathematical principles involved: none. | Unique features: automatic fixture-based global reset. | Key classes/functions: profile tests. | Risk notes: prevents cross-test leakage.
- File Name: `tests/test_frontier_upgrades.py` | Purpose: broad frontier-upgrade test pack (Hawkes, rBergomi, conformal, VRP, etc.). | Interactions: tests many modules across stack. | Mathematical principles involved: multi-model frontier behavior validation. | Unique features: large integration-like frontier test suite. | Key classes/functions: multiple class-based sections. | Risk notes: runtime cost and brittleness should be monitored.
- File Name: `tests/test_gex_enhanced_vector.py` | Purpose: verify enhanced GEX vector preserves legacy outputs and new metrics. | Interactions: tests `GEXCalculator` in Quant Engine. | Mathematical principles involved: dealer gamma/charm/vanna exposure aggregation. | Unique features: backward-compatibility assertion. | Key classes/functions: GEX vector test. | Risk notes: maintain schema compatibility for downstream consumers.
- File Name: `tests/test_hardening_integrations.py` | Purpose: verify integration of hardening modules and gating controls. | Interactions: touches OMEGA/NIRV/NOVA/Quant Engine modules. | Mathematical principles involved: safety gate and confidence behavior checks. | Unique features: multi-module hardening regression coverage. | Key classes/functions: hardening integration tests. | Risk notes: keep aligned with release hardening commitments.
- File Name: `tests/test_historical_learning.py` | Purpose: test historical-learning pull/feature/update pipeline behavior. | Interactions: tests `historical_learning.py` with mock client/data. | Mathematical principles involved: feature engineering and residual-label pipeline consistency. | Unique features: API mock with end-to-end training path checks. | Key classes/functions: historical learning tests. | Risk notes: API schema changes require test fixture updates.
- File Name: `tests/test_india_vix_synth.py` | Purpose: verify synthetic VIX against Black-Scholes-generated chains. | Interactions: tests `india_vix_synth.py`. | Mathematical principles involved: model-free variance replication sanity under known flat vol. | Unique features: controlled synthetic pricing environment. | Key classes/functions: synthetic VIX tests. | Risk notes: ensure tolerance matches interpolation design.
- File Name: `tests/test_market_conventions.py` | Purpose: test NSE convention utilities and time-to-expiry logic. | Interactions: tests `market_conventions.py`. | Mathematical principles involved: date-time annualization and market-hour mapping. | Unique features: fixture-based temporal edge-case coverage. | Key classes/functions: convention tests. | Risk notes: calendar assumptions can age quickly.
- File Name: `tests/test_model_free_forward_estimator.py` | Purpose: verify forward estimator behavior and improved estimator paths. | Interactions: tests `model_free_variance.py`. | Mathematical principles involved: put-call parity-based forward inference and strip integration. | Unique features: parity-consistent synthetic chain generation. | Key classes/functions: forward-estimator tests. | Risk notes: maintain compatibility between legacy and improved paths.
- File Name: `tests/test_model_free_variance_engine.py` | Purpose: test model-free variance and 30-day VIX computations. | Interactions: tests `model_free_variance.py`. | Mathematical principles involved: variance-swap strip and interpolation mechanics. | Unique features: detailed engine-level functional checks. | Key classes/functions: variance-engine tests. | Risk notes: filter/tail-correction changes should be guarded by tests.
- File Name: `tests/test_omega_95_hardening.py` | Purpose: hardening tests around OMEGA reliability and scoring behavior. | Interactions: tests `OMEGAModel` and `PredictionTracker`. | Mathematical principles involved: directional outcome tracking and actionability gating. | Unique features: "95"-claim hardening scenarios. | Key classes/functions: OMEGA hardening tests. | Risk notes: claim criteria changes must update tests and docs together.
- File Name: `tests/test_omega_auto_learning.py` | Purpose: validate OMEGA auto-learning from tracked outcomes. | Interactions: tests tracker-driven model update paths. | Mathematical principles involved: online residual learning feedback loop checks. | Unique features: prediction-record fixture modeling. | Key classes/functions: auto-learning tests. | Risk notes: persistence and retrain triggers should remain deterministic.
- File Name: `tests/test_omega_outputs_unchanged_with_chimera.py` | Purpose: non-regression guard that CHIMERA imports do not alter OMEGA behavior. | Interactions: tests OMEGA and CHIMERA import side effects. | Mathematical principles involved: none; behavioral invariance check. | Unique features: explicit additive-only contract enforcement. | Key classes/functions: class identity/flag equality assertions. | Risk notes: critical for architectural isolation guarantees.
- File Name: `tests/test_oos_reliability_gate.py` | Purpose: test out-of-sample reliability gate logic. | Interactions: tests OMEGA tracker and gate computation. | Mathematical principles involved: accuracy/edge threshold gating by sample size/regime/side. | Unique features: granular gate-failure reason checks. | Key classes/functions: OOS gate tests. | Risk notes: threshold defaults should be version-controlled.
- File Name: `tests/test_pricer_router.py` | Purpose: validate tiered router behavior with dummy engines. | Interactions: tests `TieredPricerRouter` in `pricer_router.py`. | Mathematical principles involved: route selection and blended output logic. | Unique features: controlled deterministic dummy pricer integration. | Key classes/functions: router tests. | Risk notes: keep routing contracts stable.
- File Name: `tests/test_pricer_router_rbergomi.py` | Purpose: validate rBergomi route activation and outputs. | Interactions: tests rough-vol branch in router. | Mathematical principles involved: rough-vol route sanity checks. | Unique features: explicit tier tag assertion (`tier_rbergomi`). | Key classes/functions: rBergomi router test. | Risk notes: ensure config keys remain backward compatible.
- File Name: `tests/test_research_high_conviction.py` | Purpose: test high-conviction scoring buckets for research mode. | Interactions: tests OMEGA conviction helpers and filtering behavior. | Mathematical principles involved: score bucketing from multiple confidence/mispricing factors. | Unique features: explicit 0-10 bucket expectations. | Key classes/functions: conviction scoring tests. | Risk notes: scoring-criterion changes require docs alignment.
- File Name: `tests/test_surface_no_arbitrage_properties.py` | Purpose: verify no-arbitrage properties on synthetic surfaces/prices. | Interactions: tests `arbfree_surface.py` and `surface_checks.py`. | Mathematical principles involved: convexity, monotonicity, butterfly/calendar constraints. | Unique features: synthetic mathematically-valid constructions. | Key classes/functions: no-arbitrage property tests. | Risk notes: numeric tolerance settings are sensitive.
- File Name: `tests/test_surface_shock.py` | Purpose: unit tests for placeholder surface-shock model API behavior. | Interactions: tests `surface_shock.py`. | Mathematical principles involved: none advanced (stub behavior). | Unique features: confirms scaffold contract before full model implementation. | Key classes/functions: `TestSurfaceShockModel`. | Risk notes: expected behavior will change when model is fully implemented.
- File Name: `tests/test_upstox_api_clients.py` | Purpose: test Upstox client parameter and payload handling. | Interactions: tests `upstox_api_clients.py` with dummy session responses. | Mathematical principles involved: none. | Unique features: response-shape and error propagation checks. | Key classes/functions: API client tests. | Risk notes: external API evolution requires fixture updates.
- File Name: `tests/test_vrp_state.py` | Purpose: deterministic and missing-data robustness tests for VRP state estimator. | Interactions: tests `vrp_state.py`. | Mathematical principles involved: term-structure state extraction consistency. | Unique features: repeatability and sparse-input checks. | Key classes/functions: VRP state tests. | Risk notes: state formula changes should retain deterministic behavior.
- File Name: `tests/test_vrr_integration.py` | Purpose: integration tests for legacy VRR state path in NIRV. | Interactions: tests NIRV + `vrr_state.py` flag behavior. | Mathematical principles involved: variance-risk-ratio-driven adjustment checks. | Unique features: legacy-path compatibility assurance. | Key classes/functions: VRR integration tests. | Risk notes: keep legacy support explicit and test-backed.

### 4.3 Documentation, Config, and Profile Catalog (22 files)

- File Name: `docs/DEPENDENCY_MAP.md` | Purpose: architecture dependency graph and insertion points. | Interactions: documents app/OMEGA/NIRV/Quant dependencies. | Mathematical principles involved: none directly. | Unique features: concise mermaid dependency map. | Key classes/functions: N/A. | Risk notes: must stay synchronized with real imports.
- File Name: `docs/NOVA_TRAINING_GUIDE.md` | Purpose: procedural guide for NOVA data prep and training workflow. | Interactions: references `historical_learning.py` and `unified_pipeline.py`. | Mathematical principles involved: practical implications of IV/spot alignment and training quality. | Unique features: operations-first workflow narrative. | Key classes/functions: N/A. | Risk notes: ensure guidance matches current code paths.
- File Name: `docs/NOVA_UI_TRAINING_GUIDE.md` | Purpose: UI-driven NOVA training instructions using broker APIs. | Interactions: references `opmAI_app.py` historical-learning tab flow. | Mathematical principles involved: data-quality and synchronization rationale. | Unique features: no-code training execution guide. | Key classes/functions: N/A. | Risk notes: UI changes can stale step numbering.
- File Name: `docs/OMEGA_TEXTBOOK_CHAPTERS_1_TO_12.md` | Purpose: large textbook-style system documentation including Chapter 13 NOVA material. | Interactions: anchors many module references and formulas. | Mathematical principles involved: broad foundational and implementation math references. | Unique features: comprehensive pedagogical narrative. | Key classes/functions: N/A. | Risk notes: long-form docs can drift without periodic audits.
- File Name: `docs/OMEGA_v5_upgrade_plan.md` | Purpose: v5 design/implementation migration documentation. | Interactions: references core module upgrade insertions and flags. | Mathematical principles involved: architecture correctness and no-arbitrage rationale. | Unique features: explicit implementation status mapping. | Key classes/functions: N/A. | Risk notes: historical doc should be labeled if superseded.
- File Name: `docs/V6_UPGRADE.md` | Purpose: v6 upgrade notes (correctness, performance, regression safety). | Interactions: references `nse_specs.py`, `model_free_variance.py`, routing upgrades. | Mathematical principles involved: contract specs and variance-engine improvements. | Unique features: focused release delta summary. | Key classes/functions: N/A. | Risk notes: keep release chronology clear.
- File Name: `docs/bugs-roadmap.md` | Purpose: bug and inaccuracy roadmap with prioritized fixes. | Interactions: references many module line locations and remediation plans. | Mathematical principles involved: identifies model/data handling inaccuracies. | Unique features: actionable risk register style. | Key classes/functions: N/A. | Risk notes: can become outdated after fixes land.
- File Name: `docs/empirical_validation_plan.md` | Purpose: practical path for empirical OOS validation of OMEGA/NOVA. | Interactions: ties app tabs, backtester, and scorecards into operational plan. | Mathematical principles involved: out-of-sample performance validation methodology. | Unique features: operations timeline and criteria checklist. | Key classes/functions: N/A. | Risk notes: metrics/thresholds should reflect latest runtime.
- File Name: `docs/full_audit.md` | Purpose: repository audit narrative with strengths/issues. | Interactions: references major module findings. | Mathematical principles involved: model correctness and optimization diagnostics. | Unique features: broad health verdict table. | Key classes/functions: N/A. | Risk notes: audit findings should be dated and revisited.
- File Name: `docs/model_rankings.md` | Purpose: expert model ranking and scoring. | Interactions: covers modules across stack. | Mathematical principles involved: multi-factor model quality scoring. | Unique features: alpha-potential perspective ranking. | Key classes/functions: N/A. | Risk notes: scoring subjectivity should be contextualized.
- File Name: `docs/model_rankings_definitive.md` | Purpose: post-hardening expanded ranking narrative. | Interactions: references all major modules and council hardening. | Mathematical principles involved: architecture and reliability evaluation criteria. | Unique features: broad system-level comparative narrative. | Key classes/functions: N/A. | Risk notes: should track subsequent revisions.
- File Name: `docs/tab_model_evaluation.md` | Purpose: tab-by-tab model evaluation for UI-facing models. | Interactions: maps app tabs to model engines. | Mathematical principles involved: comparative framework for model class strengths. | Unique features: user-facing ranking orientation. | Key classes/functions: N/A. | Risk notes: UI/tab changes can stale mapping.

- File Name: `CHANGELOG.md` | Purpose: versioned change tracking and hardening notes. | Interactions: references many modified modules and flags. | Mathematical principles involved: none directly. | Unique features: structured release deltas. | Key classes/functions: N/A. | Risk notes: keep entries atomic and dated.
- File Name: `CHIMERA_RX_MODEL_DETAILED.md` | Purpose: detailed CHIMERA-RX architecture and implementation-aligned doc. | Interactions: maps each `chimera_rx_*` module and invariants. | Mathematical principles involved: equilibrium/reflexive/compression diagnostics summary. | Unique features: explicit additive-only and readiness criteria. | Key classes/functions: N/A. | Risk notes: must stay aligned with code and tests.
- File Name: `CLAUDE.md` | Purpose: project-level AI assistant guidance and collaboration notes. | Interactions: shapes how assistants operate on repository tasks. | Mathematical principles involved: none. | Unique features: local operational instruction context. | Key classes/functions: N/A. | Risk notes: instruction drift can affect future automation quality.
- File Name: `README.md` | Purpose: primary handbook and system overview for v7 runtime. | Interactions: references core modules, flags, workflows, and formulas. | Mathematical principles involved: high-level derivations and model summaries. | Unique features: central entrypoint for users and contributors. | Key classes/functions: N/A. | Risk notes: must remain synchronized with implementation reality.
- File Name: `README_new.md` | Purpose: legacy/newer baseline handbook snapshot (v5 context). | Interactions: documents earlier runtime status and hardening details. | Mathematical principles involved: model summaries and architecture rationale. | Unique features: historical reference for evolution comparisons. | Key classes/functions: N/A. | Risk notes: clarify status to avoid confusion with main README.

- File Name: `config.env.example` | Purpose: template for required/optional environment variables. | Interactions: used by API clients, AI council, and runtime settings. | Mathematical principles involved: none. | Unique features: includes provider routing and budget keys. | Key classes/functions: N/A. | Risk notes: keep keys aligned with actual code consumption.
- File Name: `requirements.txt` | Purpose: Python dependency manifest. | Interactions: governs runtime/test environment setup. | Mathematical principles involved: none directly. | Unique features: includes optional quant/LLM packages in one manifest. | Key classes/functions: N/A. | Risk notes: dependency pinning strategy impacts reproducibility.
- File Name: `runtime.txt` | Purpose: target runtime Python version declaration. | Interactions: deployment/runtime environment coordination. | Mathematical principles involved: none. | Unique features: simple explicit python version pin. | Key classes/functions: N/A. | Risk notes: update with compatibility testing.
- File Name: `devcontainer.json` | Purpose: development container configuration. | Interactions: defines reproducible dev environment scaffolding. | Mathematical principles involved: none. | Unique features: portable workspace bootstrap. | Key classes/functions: N/A. | Risk notes: container config should track dependency updates.

- File Name: `ai_council/senate_profiles.yaml` | Purpose: seat configuration, reliability priors, provider priorities, and policy limits for AI Council. | Interactions: loaded by `ai_council/agents.py` and used by orchestrator logic. | Mathematical principles involved: weighted voting priors and confidence caps. | Unique features: profile-driven council behavior with explicit domain seats. | Key classes/functions: N/A (data file). | Risk notes: policy edits materially change council behavior; review required.

### 4.4 Catalog Completeness Notes
- Scoped file counts targeted by this catalog:
  - 75 non-test Python source files.
  - 49 test Python files.
  - 12 docs markdown files.
  - 5 root markdown files.
  - 4 config/runtime files.
  - 1 profile YAML file.
- `config.env` is intentionally documented as a sensitive local-secret runtime file in Chapter 7 and is not expanded with values.

## Chapter 5: Mathematical Foundations and Why They Exist Here

This chapter explains the major mathematical ideas in plain language and maps each idea to implementation modules.

### 5.1 Option Pricing Baselines

#### Black-Scholes-Merton (BSM)
Used as a fast baseline, implied-vol inversion reference, and sanity check. In practice:
- appears in router tier-0 logic,
- underpins IV inversion,
- and acts as a control variate reference.

Core equation concept (European call):
`C = S*exp(-qT)*N(d1) - K*exp(-rT)*N(d2)`.

Why used here: speed, interpretability, and as a benchmark baseline even when richer dynamics are required ([Lit-BSM-1973], [Lit-Merton-1973], [Repo-Router], [Repo-IVSolver]).

#### Heston stochastic volatility
Heston lets volatility move stochastically instead of staying constant.
Why used here: Indian index options often exhibit time-varying vol clustering and skew dynamics that constant-vol BSM misses.
Implemented in `heston_cos.py`, `quant_engine.py`, and NIRV Monte Carlo logic ([Lit-Heston-1993], [Repo-HestonCOS], [Repo-NIRV], [Repo-Quant]).

#### Jump diffusion
Jumps represent abrupt gap moves beyond diffusive motion.
Why used here: crash and event clusters are practically important in index options.
Implemented as Merton-style jump extensions in NIRV and upgraded by Hawkes intensity estimation in frontier paths ([Lit-Merton-1976], [Repo-NIRV], [Repo-Hawkes]).

### 5.2 Volatility Surface Mathematics

#### SVI/eSSVI
SVI/eSSVI are parsimonious smile parameterizations with arbitrage-aware constraints.
Why used here: robust fit quality across strikes/expiries while preserving no-static-arbitrage requirements.
Implemented in `nirv_model.py`, `essvi_surface.py`, with checks in `surface_checks.py` and repairs in `arbfree_surface.py` ([Lit-SVI-Gatheral], [Lit-eSSVI], [Repo-ESSVI], [Repo-SurfaceChecks]).

#### No-arbitrage constraints
Key principles:
- Butterfly condition (non-negative risk-neutral density proxy).
- Calendar monotonicity in total variance.
- Monotonic/convex call-price behavior.

Why used here: invalid surfaces can produce impossible prices and unstable hedges.

### 5.3 Fast Numerical Methods and Routing

#### COS method
Characteristic-function expansion gives very fast European pricing with strong numerical behavior.
Why used here: chain-level scanning under tight latency constraints.
Implemented in `heston_cos.py` and Quant Engine `HestonCOS` class ([Lit-COS-2008], [Lit-Albrecher-2007], [Repo-HestonCOS], [Repo-Quant]).

#### Tiered routing and rough-vol branch
Router chooses pricing path by budget and need.
Why used here: practical balance between latency and realism.
Rough-vol route improves short-dated behavior in rough regimes.
Implemented in `pricer_router.py` and invoked from NIRV/NOVA ([Lit-rBergomi-2016], [Lit-RoughVol-2018], [Repo-Router], [Repo-NOVA], [Repo-NIRV]).

### 5.4 Time-Series and Regime Mathematics

#### HMM/continuous regime mapping and GARCH
Used to detect and quantify regime shifts and volatility asymmetry.
Why used here: regime transitions materially alter option risk/reward profiles.
Implemented in NIRV/Quant components (`RegimeDetector`, `ContinuousRegimeDetector`, `GJRGarch`) ([Lit-Hamilton-1989], [Lit-Bollerslev-1986], [Lit-GJR-1993], [Repo-NIRV], [Repo-Quant]).

#### VRP state
`VRP(T) = implied_variance(T) - expected_realized_variance(T)`.
Why used here: VRP sign and slope help detect fear/complacency and calibrate model aggressiveness.
Implemented in `vrp_state.py` and legacy `vrr_state.py`, fed into NIRV/NOVA paths ([Repo-VRP], [Repo-VRR], [Repo-NOVA], [Repo-NIRV]).

### 5.5 Uncertainty Quantification and Decision Theory

#### Bayesian confidence and posterior updates
Used to avoid over-trusting point estimates, especially under sparse/illiquid data.
Implemented in NIRV and Quant Engine confidence classes ([Repo-NIRV], [Repo-Quant]).

#### Conformal prediction
Distribution-free interval construction from residual behavior.
Why used here: actionability gates should include uncertainty, not only point mispricing.
Implemented in NIRV/OMEGA/NOVA calibration and interval paths ([Lit-Conformal-2005], [Repo-RegimeConformal], [Repo-OMEGA], [Repo-NOVA]).

#### Kelly criterion
Position sizing principle for maximizing expected log-growth under probabilistic edge assumptions.
Why used here: risk-aware sizing with practical fractional Kelly adjustments.
Implemented in `quant_engine.py` and used in planning contexts ([Lit-Kelly-1956], [Repo-Quant], [Repo-OMEGA]).

### 5.6 Frontier Methods Present in NOVA/Research Paths

#### Hawkes self-exciting jumps
Captures jump clustering where one jump increases short-term probability of more jumps.
Implemented in `hawkes_jump.py` and integrated into Quant/NOVA paths ([Lit-Hawkes-1971], [Repo-Hawkes], [Repo-NOVA]).

#### Neural jump-SDE
Learns state-conditional drift/diffusion/jump parameters.
Implemented in `neural_jsde.py` and orchestrated by NOVA ([Repo-NeuralJSDE], [Repo-NOVA]).

#### KAN residual correction
Interpretable spline-edge network for residual adjustments.
Implemented in `kan_corrector.py` ([Lit-KAN-2024], [Repo-KAN]).

#### Score matching and PINN-style constraints
- `sgm_surface.py`: score-based completion from historical shape statistics.
- `pinn_vol_surface.py`: constrained RBF fitting under no-arbitrage and wing/martingale penalties.

#### Deep hedging objective
Uses learned hedge policy minimizing variance and tail-risk proxy terms including costs.
Implemented in `deep_hedging.py` ([Repo-DeepHedging]).

#### Martingale optimal transport
Model-free no-arbitrage bounds from linear programming under martingale constraints.
Implemented in `martingale_optimal_transport.py` ([Lit-MOT-Beiglbock-2013], [Repo-MOT]).

## Chapter 6: Testing and Reliability Framework

### 6.1 Test Strategy Layers
1. Golden baseline tests:
- Freeze baseline outputs to catch unintended behavior drift.

2. Unit tests:
- Verify individual modules (surface checks, API wrappers, features, states, scoring utilities).

3. Integration tests:
- Verify cross-module paths (NIRV+surface, VRP integration, hardening insertions, CHIMERA additive behavior).

4. Frontier tests:
- Validate advanced module behavior (Hawkes, rough-vol routes, conformal intervals, VRP paths).

5. Governance tests:
- AI Council provider routing, memory persistence, profile integrity, worker triggers, model-bridge conflict detection.

### 6.2 Safety Mechanisms Enforced by Tests
- No-lookahead safeguards in synthetic data generation.
- No-arbitrage properties for surfaces and call-price geometry.
- Feature profile reset discipline to prevent test contamination.
- Additive-only guarantee for CHIMERA import side effects.
- OOS reliability gate behavior and downgraded signal contracts.

### 6.3 Residual Gaps and Operational Cautions
- Very large orchestrator files (notably `opmAI_app.py`, `quant_engine.py`, `omega_model.py`, `nirv_model.py`) increase regression surface area.
- Some legacy script-style tests (`test_phase2.py`, `test_upgrades.py`) overlap with pytest suites and can diverge over time.
- Evaluation scorecards should always be interpreted with explicit data/regime context, not as invariant rank truth.

## Chapter 7: Runtime Artifacts and Operational Data

This chapter catalogs runtime files separately from source/test/docs scope.

### 7.1 Sensitive Local Runtime File
- `config.env`: local secret-bearing runtime config file. Purpose: store actual API credentials and private settings. Handling: never commit secrets; document structure only.

### 7.2 Artifact Catalog (47 files)

- Artifact File: `eval/reports/omega_scorecard_latest.json` | Type: scorecard output | Provenance: `eval/omega_scorecard.py` and related runners | Operational role: latest OMEGA scoring snapshot.
- Artifact File: `eval/reports/scorecard_20260227T164046Z.json` | Type: scorecard output | Provenance: `eval/run_all_model_scorecards.py` | Operational role: timestamped model score snapshot.
- Artifact File: `eval/reports/scorecard_20260227T164046Z.md` | Type: scorecard report | Provenance: report writer utilities | Operational role: human-readable ranking summary.
- Artifact File: `eval/reports/scorecard_20260227T164130Z.json` | Type: scorecard output | Provenance: multi-model evaluator | Operational role: timestamped evaluation artifact.
- Artifact File: `eval/reports/scorecard_20260227T164130Z.md` | Type: scorecard report | Provenance: report writer | Operational role: markdown scorecard.
- Artifact File: `eval/reports/scorecard_20260227T164220Z.json` | Type: scorecard output | Provenance: evaluator run | Operational role: comparison artifact.
- Artifact File: `eval/reports/scorecard_20260227T164220Z.md` | Type: scorecard report | Provenance: evaluator run | Operational role: markdown companion.
- Artifact File: `eval/reports/scorecard_20260227T164255Z.json` | Type: scorecard output | Provenance: evaluator run | Operational role: model ranking record.
- Artifact File: `eval/reports/scorecard_20260227T164255Z.md` | Type: scorecard report | Provenance: evaluator run | Operational role: human-readable run summary.
- Artifact File: `eval/reports/scorecard_20260227T164349Z.json` | Type: scorecard output | Provenance: evaluator run | Operational role: timestamped benchmark artifact.
- Artifact File: `eval/reports/scorecard_20260227T164349Z.md` | Type: scorecard report | Provenance: evaluator run | Operational role: markdown benchmark summary.

- Artifact File: `logs/.DS_Store` | Type: macOS metadata | Provenance: OS filesystem behavior | Operational role: none for model logic.
- Artifact File: `logs/2026-02-19/app.log` | Type: runtime log | Provenance: app execution | Operational role: diagnostics and incident review.
- Artifact File: `logs/2026-02-20/app.log` | Type: runtime log | Provenance: app execution | Operational role: diagnostics.
- Artifact File: `logs/2026-02-27/app.log` | Type: runtime log | Provenance: app execution | Operational role: diagnostics.
- Artifact File: `logs/2026-02-28/app.log` | Type: runtime log | Provenance: app execution | Operational role: latest diagnostics.

- Artifact File: `omega_data/.DS_Store` | Type: macOS metadata | Provenance: OS behavior | Operational role: none.
- Artifact File: `omega_data/historical/.DS_Store` | Type: macOS metadata | Provenance: OS behavior | Operational role: none.
- Artifact File: `omega_data/historical/processed/features.parquet` | Type: engineered feature dataset | Provenance: `historical_learning.py` | Operational role: ML residual training input.
- Artifact File: `omega_data/historical/processed/training_report_20260220T101943Z.json` | Type: training report | Provenance: historical-learning run | Operational role: audit trail of training pass.
- Artifact File: `omega_data/historical/processed/training_report_20260220T102012Z.json` | Type: training report | Provenance: historical-learning run | Operational role: calibration record.
- Artifact File: `omega_data/historical/raw/candles_20260220T101910Z.parquet` | Type: raw market candles | Provenance: API pull | Operational role: historical source data.
- Artifact File: `omega_data/historical/raw/candles_20260220T101943Z.parquet` | Type: raw market candles | Provenance: API pull | Operational role: source data.
- Artifact File: `omega_data/historical/raw/candles_20260220T101951Z.parquet` | Type: raw market candles | Provenance: API pull | Operational role: source data.
- Artifact File: `omega_data/historical/raw/candles_20260220T102012Z.parquet` | Type: raw market candles | Provenance: API pull | Operational role: source data.
- Artifact File: `omega_data/historical/raw/candles_20260220T103957Z.parquet` | Type: raw market candles | Provenance: API pull | Operational role: source data.
- Artifact File: `omega_data/historical/raw/candles_master.parquet` | Type: merged candle dataset | Provenance: historical-learning consolidation | Operational role: unified raw store.
- Artifact File: `omega_data/historical/raw/contracts_20260220T101910Z.json` | Type: raw contract payload | Provenance: API pull | Operational role: option metadata source.
- Artifact File: `omega_data/historical/raw/contracts_20260220T101943Z.json` | Type: raw contract payload | Provenance: API pull | Operational role: source contract data.
- Artifact File: `omega_data/historical/raw/contracts_20260220T101951Z.json` | Type: raw contract payload | Provenance: API pull | Operational role: source contract data.
- Artifact File: `omega_data/historical/raw/contracts_20260220T102012Z.json` | Type: raw contract payload | Provenance: API pull | Operational role: source contract data.
- Artifact File: `omega_data/historical/raw/contracts_20260220T103957Z.json` | Type: raw contract payload | Provenance: API pull | Operational role: source contract data.
- Artifact File: `omega_data/models/.DS_Store` | Type: macOS metadata | Provenance: OS behavior | Operational role: none.
- Artifact File: `omega_data/models/backups/pricing_model_20260220T102012Z.joblib` | Type: model checkpoint | Provenance: OMEGA learning pipeline | Operational role: rollback/reproducibility snapshot.
- Artifact File: `omega_data/models/backups/pricing_model_20260220T103958Z.joblib` | Type: model checkpoint | Provenance: OMEGA learning pipeline | Operational role: backup checkpoint.
- Artifact File: `omega_data/predictions.json` | Type: prediction/outcome tracker store | Provenance: OMEGA tracker logic | Operational role: feedback-loop learning source.
- Artifact File: `omega_data/pricing_model.joblib` | Type: active ML corrector artifact | Provenance: OMEGA learning | Operational role: live residual correction model.
- Artifact File: `omega_data/rollback_demo/models/backups/pricing_model_20260220T102112Z.joblib` | Type: rollback-demo checkpoint | Provenance: rollback demonstration path | Operational role: rollback validation sample.
- Artifact File: `omega_data/rollback_demo/pricing_model.joblib` | Type: rollback-demo active model | Provenance: rollback demo path | Operational role: demo model artifact.

- Artifact File: `tests/golden/snapshots/01_atm_call_normal.json` | Type: golden fixture | Provenance: golden snapshot generator | Operational role: regression baseline.
- Artifact File: `tests/golden/snapshots/02_otm_put_high_vix.json` | Type: golden fixture | Provenance: golden snapshot generator | Operational role: regression baseline.
- Artifact File: `tests/golden/snapshots/03_itm_call_low_vix.json` | Type: golden fixture | Provenance: golden snapshot generator | Operational role: regression baseline.
- Artifact File: `tests/golden/snapshots/04_near_expiry_atm.json` | Type: golden fixture | Provenance: golden snapshot generator | Operational role: regression baseline.
- Artifact File: `tests/golden/snapshots/05_otm_call_low_vol.json` | Type: golden fixture | Provenance: golden snapshot generator | Operational role: regression baseline.

- Artifact File: `trading_data/ai_council.db` | Type: SQLite state store | Provenance: AI Council runtime persistence | Operational role: council cycle memory/reliability data.
- Artifact File: `trading_data/trade_journal 2.db` | Type: SQLite journal store | Provenance: runtime/backtest journaling | Operational role: trade history persistence.
- Artifact File: `trading_data/trade_journal.db` | Type: SQLite journal store | Provenance: runtime/backtest journaling | Operational role: primary trade journal persistence.

### 7.3 Artifact Governance Guidance
- Treat `joblib`, `parquet`, and `.db` files as mutable runtime state, not immutable source.
- Preserve provenance metadata (timestamps and run parameters) for reproducibility.
- Keep logs and scorecards pruned or archived to avoid stale-performance interpretation.

## Chapter 8: Synthesis

### 8.1 Comparative Model Roles
- NIRV: structural pricing backbone with regime and surface intelligence.
- OMEGA: residual intelligence and decision framing on top of NIRV.
- NOVA: research-grade multi-stage orchestration adding richer dynamics and uncertainty layers.
- TVR: embedded PDE-based alternative/cross-check path in app workflows.
- CHIMERA-RX: additive advisory diagnostics to complement but not override core outputs.

### 8.2 Design Philosophy
The repository consistently favors:
1. Physics-first baseline.
2. ML correction second.
3. Explicit uncertainty and reliability gates.
4. Additive overlays instead of destructive rewrites.

This design is robust for high-noise derivative markets where purely black-box learning can be brittle under regime shifts.

### 8.3 Practical Extension Points
1. Break up large orchestrator files into package modules while preserving APIs.
2. Expand calibration and backtest provenance tracking for stronger reproducibility.
3. Promote artifact lifecycle policy (retention, naming, immutable eval snapshots).
4. Continue strengthening profile-aware regression tests for feature-flag combinations.

## References

### Repository References
- [Repo-README] `README.md`
- [Repo-Textbook] `docs/OMEGA_TEXTBOOK_CHAPTERS_1_TO_12.md`
- [Repo-DependencyMap] `docs/DEPENDENCY_MAP.md`
- [Repo-CHIMERA] `CHIMERA_RX_MODEL_DETAILED.md`
- [Repo-Features] `omega_features.py`
- [Repo-NIRV] `nirv_model.py`
- [Repo-OMEGA] `omega_model.py`
- [Repo-NOVA] `unified_pipeline.py`
- [Repo-Quant] `quant_engine.py`
- [Repo-Router] `pricer_router.py`
- [Repo-Hawkes] `hawkes_jump.py`
- [Repo-HestonCOS] `heston_cos.py`
- [Repo-HL] `historical_learning.py`
- [Repo-RegimeConformal] `calibration/regime_conformal.py`
- [Repo-VRP] `vrp_state.py`
- [Repo-VRR] `vrr_state.py`
- [Repo-ESSVI] `essvi_surface.py`
- [Repo-SurfaceChecks] `surface_checks.py`
- [Repo-NeuralJSDE] `neural_jsde.py`
- [Repo-KAN] `kan_corrector.py`
- [Repo-DeepHedging] `deep_hedging.py`
- [Repo-MOT] `martingale_optimal_transport.py`
- [Repo-IVSolver] `iv_solver.py`

### Canonical Literature References
- [Lit-BSM-1973] Black, F., and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities.
- [Lit-Merton-1973] Merton, R.C. (1973). Theory of Rational Option Pricing.
- [Lit-Merton-1976] Merton, R.C. (1976). Option Pricing When Underlying Stock Returns Are Discontinuous.
- [Lit-Heston-1993] Heston, S.L. (1993). A Closed-Form Solution for Options with Stochastic Volatility.
- [Lit-SABR-Hagan-2002] Hagan, P. et al. (2002). Managing Smile Risk.
- [Lit-Bollerslev-1986] Bollerslev, T. (1986). Generalized ARCH.
- [Lit-GJR-1993] Glosten, Jagannathan, and Runkle (1993). On the Relation Between the Expected Value and Volatility.
- [Lit-COS-2008] Fang, F., and Oosterlee, C.W. (2008). A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions.
- [Lit-Albrecher-2007] Albrecher, H. et al. (2007). The Little Heston Trap.
- [Lit-Hawkes-1971] Hawkes, A.G. (1971). Spectra of Some Self-Exciting and Mutually Exciting Point Processes.
- [Lit-rBergomi-2016] Bayer, C., Friz, P., and Gatheral, J. (2016). Pricing Under Rough Volatility.
- [Lit-RoughVol-2018] Gatheral, J., Jaisson, T., and Rosenbaum, M. (2018). Volatility is Rough.
- [Lit-SVI-Gatheral] Gatheral, J. (SVI/volatility-surface work; standard references in practical smile modeling).
- [Lit-eSSVI] Extended SSVI literature (eSSVI static-arbitrage parameterizations).
- [Lit-Kelly-1956] Kelly, J.L. (1956). A New Interpretation of Information Rate.
- [Lit-Conformal-2005] Shafer, G., and Vovk, V. (2005). Algorithmic Learning in a Random World.
- [Lit-LongstaffSchwartz-2001] Longstaff, F.A., and Schwartz, E.S. (2001). Valuing American Options by Simulation.
- [Lit-MOT-Beiglbock-2013] Beiglbock, M., Henry-Labordere, P., and Penkner, F. (2013). Model-Independent Bounds for Option Prices.
- [Lit-KAN-2024] Liu, Z. et al. (2024). KAN: Kolmogorov-Arnold Networks.

