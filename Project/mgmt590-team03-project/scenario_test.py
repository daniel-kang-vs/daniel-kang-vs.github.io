"""Run the full pipeline for one scenario: q* for 263 zones + model bake-off comparison
+ status-quo / uniform allocation baselines."""
import numpy as np
import pandas as pd
from engine.config import OptimizationConfig
from engine.pipeline import run_pipeline
from engine import data as data_mod, proxy as proxy_mod
from engine.evaluate import revenue_generated

# ── EDIT THESE ────────────────────────────────────────────────
config = OptimizationConfig(
    time_bucket="09:00-15:59",   # one of: 00:00-05:59, 06:00-08:59, 09:00-15:59, 16:00-18:59, 19:00-23:59
    day_of_week=4,               # 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun
    week=50,                     # ISO week number (e.g. 50 = Dec 11, 2025)
    fleet_size=13000,
)
# ──────────────────────────────────────────────────────────────

result = run_pipeline(config, verbose=True)

print("\n" + "=" * 65)
print(f"WINNING MODEL: {result['primary_model']}")
print("=" * 65)

q_df = pd.DataFrame({
    "PULocationID": result["zones"],
    "q_star": result["q_star"],
}).sort_values("q_star", ascending=False).reset_index(drop=True)

print(f"\nTop 20 zones (highest q*):")
print(q_df.head(20).to_string(index=False))

print(f"\nBottom 20 zones (lowest q*):")
print(q_df.tail(20).to_string(index=False))

print(f"\nFleet used: {result['q_star'].sum()} / {config.fleet_size}")

out_path = f"outputs/q_star_{config.time_bucket.replace(':','')}_{config.day_of_week}_wk{config.week}.csv"
q_df.to_csv(out_path, index=False)
print(f"\nFull 263-zone q* saved to: {out_path}")

# ── Status-quo & uniform baselines for THIS exact scenario ───────────────────
print("\n" + "=" * 65)
print("BASELINE COMPARISON: Optimized vs Status-quo vs Uniform")
print("=" * 65)

zones   = result["zones"]
Cu      = result["metrics"].get("Cu") if "Cu" in result["metrics"] else None
FLEET   = config.fleet_size

# Re-derive Cu and true demand for this exact scenario (pipeline doesn't expose Cu directly)
df_full = data_mod.prepare(n_clusters=config.n_clusters, seed=config.random_seed)
df = proxy_mod.add_demand(df_full, aggregation=config.proxy_aggregation, demand_multiplier=config.demand_multiplier)
train, val, test = data_mod.temporal_split(df)

from engine import costs as costs_mod
costs_df = costs_mod.compute_costs(train, cu_multiplier=config.cu_multiplier, co_multiplier=config.co_multiplier)
sc = costs_mod.get_costs_for_scenario(costs_df, config.time_bucket).reindex(zones)
Cu = sc["Cu"].values

# True realized demand for this exact (bucket, dow, week)
mask = (
    (test["time_bucket"] == config.time_bucket)
    & (test["day_of_week"] == config.day_of_week)
    & (test["week"] == config.week)
)
sub = test[mask]
if sub.empty:
    mask2 = (test["time_bucket"] == config.time_bucket) & (test["day_of_week"] == config.day_of_week)
    sub = test[mask2]
d_true = sub.set_index("PULocationID")["demand"].reindex(zones).fillna(0).values.astype(float)

# Status-quo: allocate proportional to historical yellow pickups for this bucket
hist = (
    train[train["time_bucket"] == config.time_bucket]
    .groupby("PULocationID")["yellow_pickups"].mean()
    .reindex(zones).fillna(0)
)
q_prop = (hist.values / hist.values.sum() * FLEET).clip(min=0)
q_prop = q_prop * FLEET / q_prop.sum()

# Uniform: split fleet equally across all zones
q_unif = np.full(len(zones), FLEET / len(zones))

rev_opt  = revenue_generated(result["q_star"].astype(float), d_true, Cu)
rev_prop = revenue_generated(q_prop, d_true, Cu)
rev_unif = revenue_generated(q_unif, d_true, Cu)

print(f"\n  Revenue — Optimized ({result['primary_model']}):  ${rev_opt:>12,.0f}")
print(f"  Revenue — Status quo (prop. to history): ${rev_prop:>12,.0f}")
print(f"  Revenue — Uniform (equal split):         ${rev_unif:>12,.0f}")
print(f"\n  Gain vs status quo: ${rev_opt - rev_prop:>12,.0f}   ({(rev_opt-rev_prop)/rev_prop*100:+.1f}%)")
print(f"  Gain vs uniform:    ${rev_opt - rev_unif:>12,.0f}   ({(rev_opt-rev_unif)/rev_unif*100:+.1f}%)")
print(f"  Annualized gain vs status quo (×52 wks): ${(rev_opt - rev_prop) * 52:>12,.0f}")
