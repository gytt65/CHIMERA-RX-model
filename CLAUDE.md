# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run opmAI_app.py

# Run all tests
python -m pytest tests/ -x -q

# Run a single test file
python -m pytest tests/test_chimera_rx_engine.py -x -q

# Run golden master regression tests
python -m pytest tests/golden/ -x -q

# Regenerate golden snapshots (after intentional pricing changes)
python tests/golden/generate_golden_snapshots.py

# Run CPU benchmark (100 strikes x 2 expiries)
python bench/bench_pricing.py

# Run evaluation harness against golden snapshots
python -m eval.run --snapshots tests/golden/snapshots --features '{"arb_free_surface":true}'

# Syntax check without running
python3 -m py_compile opmAI_app.py
```

## Architecture

OMEGA (Options Market Efficiency & Generative Analysis) is an option pricing and trading system for NSE Nifty 50 index options. The pricing formula is:
```
OMEGA_price = NIRV_base × (1 + ML_correction) + sentiment_adjustment
```

### Core Pricing Stack (bottom-up)

```
iv_solver.py          Jaeckel "Let's Be Rational" IV solver
    ↑
nirv_model.py         Mathematical core: HMM regime detection, SVI vol surface,
                      Heston MC + Sobol QMC, Bayesian confidence, CRN Greeks
    ↑
quant_engine.py       15 institutional quant methods (SABR, GARCH, COS, GEX, Kelly...)
    ↑
omega_model.py        6-layer ML/AI orchestrator:
                        L0: NIRV base, L1: ML correction (GBM), L2: Anomaly (IsoForest),
                        L3: Sentiment (Gemini/Perplexity), L4: Behavioral actors,
                        L5: Adaptive learning
    ↑
opmAI_app.py          Streamlit dashboard (~25k lines — read in chunks with offset+limit)
```

### Frontier Pricing Tier

These modules are gated behind feature flags in `omega_features.py` (all default OFF):

- **Surfaces:** `arbfree_surface.py` (no-arb checks), `essvi_surface.py` (parametric eSSVI), `svi_fixed_point.py` (warm-start), `sgm_surface.py` (score-matching completion), `pinn_vol_surface.py` (5-constraint PINN)
- **Pricers:** `heston_cos.py` (semi-analytical COS, 4000x faster than MC), `pricer_router.py` (tiered CPU-budgeted routing including rBergomi), `ensemble_pricer.py` (adaptive weighting across NIRV/rBergomi/KAN/PINN)
- **Neural/ML:** `neural_jsde.py` (learned drift/diffusion/jump), `kan_corrector.py` (KAN B-spline corrector), `deep_hedging.py` (surface-informed hedge ratio)
- **Microstructure:** `hawkes_jump.py` (self-exciting jumps), `path_signatures.py` (rough path features), `behavioral_agents.py` (retail lottery/institutional hedging), `structural_frictions.py` (STT, margin, overnight gap)
- **Variance:** `model_free_variance.py` (model-free VIX), `vrp_state.py` (variance risk premium), `vrr_state.py` (risk aversion filter)
- **Pipeline:** `unified_pipeline.py` orchestrates: PINN Surface → Neural J-SDE → KAN Corrector → Conformal → Deep Hedge

### Decision Support Systems

- **CHIMERA-RX** (`chimera_rx_*.py`, 9 modules): Additive, non-mutating shadow system. Agent equilibrium solver (5 agent classes × 3 latent states), reflexive dynamics, friction deformation, quality score Q_t ∈ [0,10], phase transition diagnostics, causal compression abstention. UI in `chimera_rx_tab.py`.
- **AI Council** (`ai_council/`): 14-seat expert senate with weighted voting, LLM provider routing (Gemini/OpenAI/Anthropic/Perplexity), background worker cycles, Telegram alerts. Profiles in `senate_profiles.yaml`.

### Feature Flag System

`omega_features.py` controls all optional modules via an immutable singleton. Flags span v5 legacy, v6, v7 frontier, and hardening guards. Override via:
```python
from omega_features import set_features
set_features(USE_HAWKES_JUMPS=True, USE_PINN_SURFACE=True)
```
Or via env: `OMEGA_FEATURES_JSON='{"USE_HAWKES_JUMPS": true}'`

Built-in profiles: `OmegaFeatures.best_mode_macbook()`, `OmegaFeatures.best_mode_max_accuracy()`, `OmegaFeatures.all_on()`.

## Conventions

- **Option types:** `'CE'` / `'PE'` (Indian convention), never `'call'`/`'put'`
- **Underlying:** Nifty 50 index options on NSE; lot size = 65 (see `nse_specs.py` for BANKNIFTY:30, FINNIFTY:60)
- **VIX:** India VIX as percentage (e.g. `14.0` = 14%)
- **Flows:** FII/DII in ₹ crores (positive = net buy)
- **Time:** `T` in years (e.g. `7/365` for 1 week)
- **Rates:** `r` ≈ 0.065–0.07, `q` ≈ 0.012–0.013
- **ML cold start:** All ML layers return zero correction until ≥30 training samples
- **Graceful degradation:** Heavy ML libs (sklearn, arch, hmmlearn, xgboost, lightgbm) are optional — system falls back automatically

## Sensitive Files

- `config.env` — real API credentials; **never commit** (use `config.env.example` as template)
- `trading_data/*.db` — live SQLite trade journals
- `omega_data/*.joblib` / `omega_data/*.json` — ML model state and prediction logs

## Test Patterns

Tests use plain pytest (no fixtures framework). Common pattern:
1. Construct input dataclass directly (e.g. `CHIMERARXInput(spot=22500, ...)`)
2. Run the engine/model
3. Assert output contract (bounds, types, required keys)

Golden master tests pin NIRV outputs with fixed seeds (`use_sobol=False` + explicit seed, small `n_paths`). After intentional pricing changes, regenerate snapshots before committing.

## Adding a New Market Factor

1. Register in `FactorRegistry.FACTORS` in `omega_model.py`
2. Add fetching logic in `opmAI_app.py` (the `_fetch_*` methods)
3. Include in `FeatureFactory.extract()` if it should be an ML feature

## Adding a New Feature-Flagged Module

1. Add flag to `OmegaFeatures._DEFAULTS` in `omega_features.py` (default OFF)
2. Add lowercase alias in `OmegaFeatures._ALIASES`
3. Guard all call sites with `if get_features().YOUR_FLAG:`
4. Add to appropriate profile (`best_mode_macbook`, `best_mode_max_accuracy`) once stable
5. Write tests that work with the flag both on and off
