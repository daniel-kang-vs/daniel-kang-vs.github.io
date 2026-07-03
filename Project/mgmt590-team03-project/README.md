# NYC Yellow Taxi — Prescriptive Demand Optimization

A two-phase AI system that solves **where to position 13,000 yellow cabs** across 263 NYC zones
to maximize served trips (and revenue) for a given time window.

- **Phase 1 — Optimization Engine:** given a `(time_bucket, day_of_week, month)`, outputs the
  optimal allocation `q*` (263-vector) by minimizing asymmetric newsvendor cost.
- **Phase 2 — Agent Layer** *(planned)*: a LangGraph/Groq agent that parses management
  instructions, maps them to a config diff, re-runs the engine, and logs every decision.

---

## Why Newsvendor (not MSE Forecasting)?

Under-serving a busy zone (lost fare ≈ **Cu** ~$17) costs more than over-serving a quiet zone
(idle cab ≈ **Co** ~$12 average fare). The newsvendor model encodes this asymmetry:

```
q*_z = F_z^{-1}(τ_z)    where τ_z = Cu_z / (Cu_z + Co_z)
```

High-fare zones (JFK, Midtown) get a higher critical fractile `τ` → more cabs.
MSE forecasting ignores this cost asymmetry entirely.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Confirm data file is present
ls zone_time_bucket_dow_month_agg_2025.parquet
```

> **Python:** 3.11+ recommended. Tested on macOS 14 (arm64).

---

## Data

**File:** `zone_time_bucket_dow_month_agg_2025.parquet`

| Dimension | Values |
|-----------|--------|
| Zones (`PULocationID`) | 263 (IDs 1–265 excluding 103, 104) |
| Time buckets | 5: `00:00–05:59`, `06:00–08:59`, `09:00–15:59`, `16:00–18:59`, `19:00–23:59` |
| Days of week | 0 (Mon) – 6 (Sun) |
| Months | 1 – 12 (2025) |
| Total rows | ~109,378 |

**Key columns:** `yellow_pickups`, `fhv_pickups`, `yellow_avg_total_fare`, `temperature_2m`,
`precipitation`, `relative_humidity_2m`, `apparent_temperature`, `zone_hours`.

**Demand proxy:** Since true yellow demand is supply-censored, we reconstruct it as:
```
mul_factor_b = avg_yellow_b / (avg_yellow_b + avg_fhv_b)    # per borough b
demand_z     = mul_factor_{b(z)} * (yellow_pickups_z + fhv_pickups_z)
```
Borough-level aggregation removes the global scalar's structural bias against outer boroughs.

---

## Quick Start

```python
from engine import data as data_mod, proxy as proxy_mod
from engine.config import OptimizationConfig
from engine.pipeline import run_pipeline

# Load data
df = proxy_mod.add_demand(data_mod.prepare(), aggregation="borough")

# Define scenario
cfg = OptimizationConfig(
    time_bucket="09:00-15:59",
    day_of_week=2,   # Tuesday
    month=12,
    fleet_size=13000,
)

# Run full pipeline (fits models, solves, evaluates)
result = run_pipeline(cfg, df_full=df)

print("q* (first 10 zones):", result["q_star"][:10])
print("Fleet used:", result["q_star"].sum())
print("NV cost:", result["metrics"]["nv_cost"])
```

---

## Running the Results Script

Produces all charts, CSVs, and a bake-off table in one shot (no notebook required):

```bash
# If using the project venv:
.venv/bin/python run_results.py
# or, after activating the venv (source .venv/bin/activate):
python run_results.py
```

**Outputs written to `outputs/`:**
| File | Description |
|------|-------------|
| `bakeoff_results.csv` | Model comparison: NV cost, pinball loss, fill rate, rank |
| `q_star_peak_tuesday.csv` | Zone allocations for peak Tuesday |
| `charts/01_pickups_by_bucket.png` | Yellow vs FHV volume by time bucket |
| `charts/02_proxy_borough_vs_global.png` | Market-share proxy: borough vs global |
| `charts/03_demand_by_borough.png` | Demand distribution by borough |
| `charts/04_weather_correlation.png` | Weather–demand correlation heatmap |
| `charts/05_co_by_bucket.png` | Overage cost (Co) per bucket |
| `charts/06_tau_distribution.png` | Critical fractile τ distribution (peak bucket) |
| `charts/07_solver_runtimes.png` | SLSQP / water-filling / cvxpy cross-check |
| `charts/08_allocation_top30.png` | Top-30 zones by allocation |
| `charts/09_dfl_training_loss.png` | DFL SPO+ loss vs epoch |
| `charts/10_bakeoff.png` | NV cost and pinball loss: 3 models |
| `charts/11_revenue_comparison.png` | Revenue: optimized vs baselines |
| `charts/12_under_vs_overage.png` | Under vs overage cost breakdown (pie) |
| `charts/13_understocked_zones.png` | Top-10 most understocked zones |
| `charts/14_proxy_sensitivity.png` | Global vs borough allocation scatter |
| `charts/15_what_if.png` | What-if scenario revenue impact |

Runtime: ~5–10 minutes (model training dominates).

---

## Running the Notebook

```bash
jupyter notebook notebooks/results.ipynb
# or
jupyter lab notebooks/results.ipynb
```

The notebook has 8 sections mirroring the results script, with richer inline commentary.

---

## Project Structure

```
Project_AI_optimization/
├── requirements.txt               # Pinned Python dependencies
├── README.md                      # This file
├── run_results.py                 # Portable results + EDA script
│
├── engine/                        # Phase 1: Optimization Engine
│   ├── config.py                  # OptimizationConfig (Pydantic) — single source of truth
│   ├── data.py                    # load_raw(), temporal_split(), prepare()
│   ├── proxy.py                   # Demand proxy (borough/global/cluster)
│   ├── costs.py                   # Cu, Co, τ per zone×bucket
│   ├── bounds.py                  # Per-zone floors and caps
│   ├── optimize.py                # SLSQP, water-filling, cvxpy, integer rounding
│   ├── evaluate.py                # NV cost, pinball, fill-rate, bake-off summary
│   ├── pipeline.py                # run_pipeline() — orchestrates all stages
│   └── models/
│       ├── empirical.py           # Empirical quantile allocator (reference baseline)
│       ├── linear_demand.py       # Track A: Poisson GLM (sklearn PoissonRegressor)
│       ├── dfl_spo.py             # Track B: DFL / SPO+ with L-BFGS-B
│       └── qr_lgbm.py             # LightGBM quantile model (future use, not active)
│
├── notebooks/
│   └── results.ipynb              # 8-section EDA + results notebook
│
├── outputs/                       # Created at runtime by run_results.py
│   ├── bakeoff_results.csv
│   ├── q_star_peak_tuesday.csv
│   └── charts/
│
└── zone_time_bucket_dow_month_agg_2025.parquet    # Input data (not versioned)
```

---

## Key Assumptions & Limitations

| Assumption | Impact |
|-----------|--------|
| **Supply-censored demand proxy** | `yellow_pickups` reflects served trips, not true demand; proxy reconstructs latent demand via FHV signal — biased where yellow supply was constrained |
| **Year stationarity** | Cu/Co computed from 2025 averages; seasonal pricing shifts not modeled |
| **Static fleet snapshot** | No intra-bucket rebalancing; `q*` is a pre-positioning decision |
| **Fares as margin proxy** | No trip distance/duration in data; fare ≈ contribution margin |
| **Thin per-cell samples** | ~12 monthly observations per (zone, bucket, DOW) cell for empirical model |
| **Exogenous Cu/Co** | No fare elasticity — required to keep the problem convex |
| **Integer rounding** | Continuous optimum is rounded via greedy repair; near-optimal but not exact |

---

## Model Performance Summary (Peak Tuesday, Test Month)

| Model | Mean NV Cost | Fill Rate | Rank |
|-------|-------------|-----------|------|
| Empirical (reference) | baseline | ~5.8% | 1 |
| Poisson GLM (Track A) | comparable | ~5.8% | 2 |
| DFL / SPO+ (Track B) | comparable | ~5.8% | 3 |

Fill rate of ~5.8% reflects yellow taxi's actual market share (~6%) of total rideshare in peak
hours. The optimization finds highest-value zones (airports, Midtown) for the 13,000-cab fleet.
All cost is underage (100%): demand far exceeds supply — the challenge is *where* to deploy,
not whether to deploy.

**Revenue gain vs status-quo proportional dispatch:** ~$238k per Tuesday midday slot,
annualized to **~$12.4M/year** (52 Tuesdays).
