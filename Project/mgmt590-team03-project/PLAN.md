# NYC Yellow-Taxi Prescriptive Optimization Agent — Implementation Plan

## Context

We are building an AI agent for **prescriptive demand optimization** of NYC yellow taxis. The
end goal: management supplies new operating parameters, constraints, or data (via uploaded
document, typed instruction, or revealed insight); the agent parses the request, tweaks the
underlying optimization model accordingly, re-runs it **quickly and correctly with full
observability/traceability**, and recommends an allocation that maximizes served trips.

The work is split into two phases:
- **Phase 1 — Optimization Engine:** a solid, validated baseline that, for a given
  (time bucket, day, month), outputs the optimal taxi allocation `q*` (a 263×1 vector — one
  per zone) by minimizing a **newsvendor** loss with asymmetric under/over-supply costs.
- **Phase 2 — Agent Layer:** wraps the engine as a callable tool behind a LangGraph agent
  (Groq/Gemini) that ingests management changes, maps them to a typed config, re-runs, and
  explains the result — with every run logged for traceability.

Why newsvendor (not MSE forecasting): under-serving (lost fares) and over-serving (idle
cabs) have **different costs**; newsvendor encodes that asymmetry directly, and its optimum is
the **critical-fractile quantile** of demand, `q*_z = F_z^{-1}(τ_z)` with `τ_z = Cu_z/(Cu_z+Co_z)`.

## Dataset reality (verified from `zone_time_bucket_dow_month_agg_2025.parquet`)

- 109,378 rows × 19 cols; grid is ~99% dense over (month × day_of_week × time_bucket × zone).
- **263 zones** (`PULocationID`, IDs 1–265 minus 103/104) → `q*` is **263×1**, not 260.
- **5 time buckets**, not 6: `00:00–05:59`, `06:00–08:59`, `09:00–15:59`, `16:00–18:59`,
  `19:00–23:59`.
- Trip counts: `yellow_pickups`, `fhv_pickups` (Uber/Lyft proxy), `pickup_count` (= sum, verified).
  `yellow_pickups = 0` in 8.65% of cells (yellow fares NaN there); `fhv_pickups = 0` in 0.77%.
- Fares: `yellow_avg_total_fare / base_fare / tip`, `fhv_avg_*` (no distance/duration — costs
  must come from fares). Weather: `temperature_2m`, `precipitation`, `relative_humidity_2m`,
  `apparent_temperature`. Exposure: `zone_hours`.
- Data quality is high (no dup keys; NaNs only where the corresponding pickup count is 0).
  One outlier to clip: `yellow_avg_base_fare` max 2092.5.

## Key design decisions (locked with user)

1. **Decision-focused learning: run BOTH tracks (Option C).** Predict-then-optimize *and*
   end-to-end DFL, compared head-to-head on out-of-sample realized newsvendor cost.
2. **Constraints: single-period, per-bucket, convex.** Budget `sum(q) ≤ 13000` + data-derived
   per-zone caps + floors. **Multi-period spatial rebalancing is explicitly OUT of scope.**
3. **Demand model:** empirical quantiles (reference) + quantile regression (core) + NegBinomial
   (stretch).
4. **Demand proxy:** `mul_factor` computed at **borough / zone-cluster** granularity (not one
   global scalar) to remove outer-borough bias; global scalar kept as a sensitivity check.
5. **Optimizers = class methods (scipy):** **SLSQP** for the constrained inner allocation;
   **L-BFGS-B / BFGS** for the outer DFL parameter learning. No PyTorch/cvxpylayers.
6. **Agent stack: LangGraph/LangChain + Groq/Gemini** (Phase 2).

### Why this stays convex (and why dimension/per-zone costs don't change that)
Each zone's cost `Cz(q) = Co_z·E[(q−D)+] + Cu_z·E[(D−q)+]` is convex in `q` (its 2nd derivative
`(Cu_z+Co_z)·f_z(q) ≥ 0`). Heterogeneous per-zone `Cu_z, Co_z` are just constants → they move
each zone's target fractile, not the curvature. The total `Σ Cz` is convex; budget + box bounds
are affine → a convex polytope. A *linear* coupling constraint keeps the set convex, so the
budget-coupled problem is a single-global-optimum convex program. Non-convexity would only come
from integrality, multi-period flow, or fare-elasticity — all out of the baseline scope.

---

## Phase 1 — Optimization Engine

### Stage 1.1 — Demand realization (proxy)
No ground-truth demand exists, so realize latent yellow demand from observed activity:
```
mul_factor_b = avg_yellow_b / (avg_yellow_b + avg_fhv_b)          # per borough/cluster b
demand_{z,t,d,m} = mul_factor_{b(z)} * (yellow_pickups + fhv_pickups)
```
- `b(z)` = borough of zone z (map via TLC zone lookup) or a KMeans cluster on zone activity
  profiles if borough mapping is unavailable.
- Rationale: a *more-aggregated-than-cell* market share creates genuine over- and under-supply
  signal (a cell where yellow is locally over/under-represented deviates from its group share);
  doing it per borough avoids the global scalar's structural over-statement of yellow demand in
  FHV-dominated outer boroughs.
- Also compute the **global-scalar** version for the sensitivity report.
- **Documented limitation:** observed yellow trips are supply-censored (`served = min(demand,
  supply)`), and FHV is only an approximate latent-demand signal. The proxy is the single
  largest assumption; the pipeline can only be validated *self-consistently* against it, not
  against true demand.

### Stage 1.2 — Cost parameters (per zone × time bucket, year-averaged)
- **Underage `Cu_{z,t}`** = lost contribution per unserved trip ≈ `yellow_avg_total_fare`
  (incl. tip) for that zone×bucket (year mean).
- **Overage `Co_t`** = opportunity cost of an idle cab ≈ **all-zones** average fare for that
  bucket (the citywide earning forgone). Bucket-level (same across zones within a bucket).
- ⇒ critical fractile `τ_{z,t} = Cu_{z,t}/(Cu_{z,t}+Co_t)`: high-fare zones (airports) get a
  higher τ → stocked more; low-fare zones get less. Matches the intended economics.
- Guards: enforce `Cu, Co > 0`; clip fare outliers; impute `Cu` for yellow-zero cells from the
  borough×bucket mean.

### Stage 1.3 — Constraints (caps & floors, data-derived, agent-tunable)
- **Cap:** `cap_z = ceil(1.5 * D_hat_p95(z,t,d))` (also optional hard ceiling = X% of fleet).
- **Floor:** `floor_z = ceil(α * D_hat_median(z,t,d))`, α ≈ 0.1–0.25; floor = 0 for
  negligible-demand zones. Equity / minimum-coverage lever.
- **Feasibility invariants:** `Σ floor_z ≤ 13000` (keep ≲ 20–30% of fleet for headroom) and
  `floor_z ≤ cap_z`. All bounds affine ⇒ problem stays convex.

### Stage 1.4 — Demand-uncertainty models (need a distribution F_z, not a point)
- **(a) Empirical quantiles — reference baseline to beat.** Pool demand across months (and dow)
  per (zone,bucket); read off the τ-quantile. Caveat: only ~12 monthly points per cell → thin;
  pooling across dow/borough borrows strength.
- **(b) Quantile regression — core.** LightGBM with pinball loss at `τ_{z,t}`; features =
  zone/borough/cluster, time_bucket, dow, month (cyclical), weather, `zone_hours` offset.
  Borrows strength across cells; directly predicts the decision quantile.
- **(c) Negative Binomial count model — stretch.** Fit `D ~ NegBin(μ_z, α_z)` for overdispersed
  counts; take analytic `F_z^{-1}(τ)`. Gives a full distribution for a third bake-off entry.

### Stage 1.5 — Optimization (inner allocation), shared by both tracks
Solve, per (bucket, dow, month):
```
min_q   Σ_z [ Co_t·E[(q_z − D_z)+] + Cu_{z,t}·E[(D_z − q_z)+] ]
s.t.    Σ_z q_z ≤ 13000 ;  floor_z ≤ q_z ≤ cap_z
```
- **Primary solver: SLSQP** (`scipy.optimize.minimize`, method="SLSQP") — handles the linear
  budget inequality + box bounds on the smooth convex objective.
- **Cross-checks (must agree):** analytic **clipped water-filling** (bisection on the budget
  multiplier λ; `F_z(q_z)=(Cu_z−λ)/(Cu_z+Co_z)` clipped to `[floor_z,cap_z]`) and a **cvxpy**
  convex solve. Three independent confirmations of the optimum.
- Continuous `q*` then integer-rounded with a greedy budget-repair pass (continuous optimum is
  convex; rounding is near-optimal here).

### Stage 1.6 — Track A: Predict-then-Optimize
LightGBM quantile predictor (1.4b) trained **decoupled** (standalone pinball), then the
Stage-1.5 SLSQP/water-filling allocation. For the *unconstrained* newsvendor this is provably
decision-optimal; the budget+floors are what make it differ from Track B.

### Stage 1.7 — Track B: End-to-End Decision-Focused Learning (DFL)
Learn predictor params so the *resulting allocation's realized cost* is minimized:
1. Model `f_θ(x)` → predicted demand `D_hat` (start linear/GLM — smooth, moderate-dim, ideal for
   quasi-Newton; avoids fighting L-BFGS with a deep net).
2. Inner solve `q*(D_hat)` via **SLSQP** (Stage 1.5).
3. Realized regret of `q*` vs true held demand; use the **SPO+ convex surrogate**
   (Elmachtoub–Grigas) for well-behaved subgradients ∂loss/∂θ.
4. **L-BFGS-B** (primary) / **BFGS** (alt) minimize total SPO+ loss over θ. Iterate.

Tracks A and B share the identical inner optimizer and differ only in predictor training → a
clean A/B. Empirical allocator (1.4a) is the reference both must beat.

### Stage 1.8 — Validation & metrics
- **Split:** temporal — train months 1–10, validate 11, test 12 (note Dec holiday shift); plus a
  rolling-month backtest. No leakage across the split.
- **Primary metric:** out-of-sample **realized newsvendor cost** on held demand
  (`Σ Cu·(D−q)+ + Co·(q−D)+`).
- **Secondary:** pinball loss at τ; achieved fill-rate / service level vs target τ; over/under
  split; total realized profit; fleet utilization; constraint satisfaction; **SPO regret vs
  oracle** (allocation using true demand).
- **Bake-off:** {empirical, QR-PtO, NegBin, DFL} × allocation, ranked on primary metric;
  plus the proxy sensitivity report (global vs borough/cluster) showing how `q*` shifts.

### Stage 1.9 — Limitations (state explicitly)
- Demand is a synthetic proxy → no true-demand validation; supply-censoring and FHV-as-demand
  bias remain. - Year-stationarity assumption. - Static fleet snapshot (no intra-bucket
  dynamics). - Fares (no distance/duration) approximate margins. - Thin per-cell samples for the
  empirical model. - Cu/Co exogenous (no fare elasticity) — required to keep the baseline convex.

### Phase-1 deliverables / files (new, under `engine/`)
- `data.py` (load parquet, borough/cluster map, splits), `proxy.py` (demand realization +
  sensitivity), `costs.py` (Cu/Co, τ), `bounds.py` (caps/floors), `models/` (`empirical.py`,
  `qr_lgbm.py`, `negbin.py`, `dfl_spo.py`), `optimize.py` (SLSQP + water-filling + cvxpy +
  rounding), `evaluate.py` (metrics, backtest, bake-off), `config.py` (Pydantic
  `OptimizationConfig`). A `notebooks/` EDA + results notebook.

---

## Phase 2 — Agent Layer (LangGraph + Groq/Gemini)

**Goal:** management uploads a document / types an instruction / reveals data → agent maps it to
a validated config diff → re-runs the engine → returns `q*` + plain-English rationale, fully logged.

### Components
- **Typed contract:** `OptimizationConfig` (Pydantic) = the single source of truth — fleet size,
  per-zone cap/floor overrides, Cu/Co multipliers, weather scenario, target bucket/dow/month,
  demand multipliers, proxy aggregation level. The engine is a pure function of this config.
- **Engine tool:** `solve(config) -> {q*, objective, diagnostics}` registered as a LangChain tool.
- **LangGraph flow:** `ingest` (PDF/text/NL) → `extract` (LLM → config **diff**, structured
  output) → `validate` (Pydantic + feasibility: `Σfloor ≤ fleet`, `floor ≤ cap`) → `solve` →
  `explain` (before/after deltas, which zones gained/lost, why). Invalid/infeasible → repair loop
  back to `extract`.
- **LLM:** Groq or Gemini for the NL→structured-config extraction (the LLM does parsing, **not
  math**). Model is swappable behind the LangChain interface.

### Speed (the "fast + correct" requirement)
- Predictor trained **once and cached**; a parameter/constraint change is just a re-solve
  (SLSQP/water-filling ⇒ milliseconds). Only **new data** triggers a re-fit. Two distinct paths:
  `reparametrize` (instant) vs `refit` (background).

### Observability / traceability (hard requirement)
- **Run registry** (one record per solve): input config, parsed diff from prior config, solver
  status, objective value, runtime, `q*`-hash, model/data version. Append-only (JSONL or SQLite).
- **Tracing:** LangSmith (or OpenTelemetry) over the LangGraph nodes; deterministic seeds for
  reproducibility; every recommendation links back to the config + run id that produced it.
- **Guardrails:** schema validation, feasibility check, and infeasibility reported (never a
  silent bad allocation).

### Phase-2 deliverables / files (new, under `agent/`)
- `config.py` (shared Pydantic schema), `tools.py` (engine tool), `graph.py` (LangGraph nodes),
  `extract.py` (LLM prompt + structured output), `registry.py` (run log), a thin CLI/Streamlit
  entrypoint, and a sample management document for the end-to-end demo.

---

## 3-Day Timeline
- **Day 1 — Engine core:** EDA, borough/cluster map, proxy (+ sensitivity), Cu/Co/τ, caps/floors,
  SLSQP + water-filling + cvxpy agreement, empirical + QR models, temporal split + primary metric.
- **Day 2 — DFL + bake-off:** SPO+ DFL with L-BFGS-B/BFGS, Track-A vs Track-B vs empirical
  comparison, backtest, NegBin (stretch), results notebook + limitations writeup.
- **Day 3 — Agent layer:** Pydantic config, engine tool, LangGraph flow, Groq/Gemini extraction,
  run registry + tracing, end-to-end demo on a sample management doc, README.

## Verification (end-to-end)
1. **Optimizer correctness:** SLSQP, water-filling, and cvxpy return the same `q*`
   (within tolerance) on the same instance; `Σ q* ≤ 13000`, `floor ≤ q* ≤ cap` all hold.
2. **Decision quality:** on the held-out month, realized newsvendor cost is reported for
   {empirical, QR-PtO, NegBin, DFL}; DFL ≤ QR-PtO ≤ empirical expected when the budget binds —
   verify and explain where it doesn't.
3. **Proxy sensitivity:** global vs borough/cluster `q*` deltas reported.
4. **Agent loop:** feed a sample doc ("cut fleet to 11,000; guarantee ≥3 cabs in every Bronx
   zone; rainy-day scenario") → agent emits a valid config diff, re-solves, returns `q*` + a
   correct before/after explanation; the run appears in the registry with full metadata.
