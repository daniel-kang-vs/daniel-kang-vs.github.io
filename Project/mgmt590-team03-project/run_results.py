#!/usr/bin/env python3
"""
run_results.py — NYC Yellow-Taxi Optimization: Full Results & EDA Script

Trains all models, evaluates on the test set, and saves:
  - outputs/bakeoff_results.csv         (model comparison table)
  - outputs/q_star_peak_tuesday.csv     (allocation for peak Tuesday)
  - outputs/charts/                     (15 PNG charts)

Usage:
    python run_results.py            # with project venv active
    .venv/bin/python run_results.py  # or directly
"""

import warnings
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

OUT = pathlib.Path("outputs")
CHARTS = OUT / "charts"
OUT.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

from engine import data as data_mod
from engine import proxy as proxy_mod
from engine import costs as costs_mod
from engine import bounds as bounds_mod
from engine import optimize as opt_mod
from engine.evaluate import (
    realized_nv_cost, pinball_loss, fill_rate, spo_regret,
    fleet_utilization, compute_all_metrics, bakeoff_summary,
    proxy_sensitivity_summary, revenue_generated, revenue_foregone,
)
from engine.models.empirical import EmpiricalAllocator
from engine.models.linear_demand import LogLinearDemandModel
from engine.models.qr_lgbm import LGBMQuantileModel
from engine.models.dfl_spo import LinearDFLModel
from engine.config import OptimizationConfig

FLEET       = 9000
PEAK_BUCKET = "09:00-15:59"
PEAK_DOW    = 2    # Tuesday
PEAK_WEEK   = 50   # mid-December — representative test week

BOROUGH_NAMES = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("NYC YELLOW-TAXI OPTIMIZATION — RESULTS")
print("=" * 65)
print("\n[1/8] Loading and preparing data …")

df_full = data_mod.prepare()
df = proxy_mod.add_demand(df_full, aggregation="borough")
train, val, test = data_mod.temporal_split(df)

print(f"  Dataset: {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"  Zones: {df['PULocationID'].nunique()} | Buckets: {df['time_bucket'].nunique()} | "
      f"Weeks: {df['week'].nunique()} | DOW: {df['day_of_week'].nunique()}")
print(f"  Train: weeks 1–44 ({len(train):,} rows) | Val: weeks 45–48 ({len(val):,}) | "
      f"Test: weeks 49–52 ({len(test):,})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/8] EDA charts …")

# 2a. Yellow vs FHV pickups by time bucket
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, label, color in zip(
    axes,
    ["yellow_pickups", "fhv_pickups"],
    ["Yellow Taxi Pickups", "FHV (Uber/Lyft) Pickups"],
    ["#f4c430", "#1f77b4"],
):
    bucket_means = df.groupby("time_bucket")[col].mean().reset_index()
    bucket_means = bucket_means.sort_values("time_bucket")
    ax.bar(bucket_means["time_bucket"], bucket_means[col], color=color, edgecolor="white")
    ax.set_title(f"Mean {label} by Time Bucket")
    ax.set_xlabel("")
    ax.set_ylabel("Mean pickups per cell")
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Yellow vs FHV Trip Volume by Time Bucket", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(CHARTS / "01_pickups_by_bucket.png", dpi=150)
plt.close()

# 2b. Demand proxy: global vs borough mul_factor by borough
borough_factors = (
    df.groupby("borough_id")
    .agg(avg_yellow=("yellow_pickups", "mean"), avg_fhv=("fhv_pickups", "mean"))
    .reset_index()
)
borough_factors["borough_factor"] = (
    borough_factors["avg_yellow"] / (borough_factors["avg_yellow"] + borough_factors["avg_fhv"])
)
global_factor = df["yellow_pickups"].mean() / (df["yellow_pickups"].mean() + df["fhv_pickups"].mean())
borough_factors["borough_name"] = borough_factors["borough_id"].map(BOROUGH_NAMES)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(borough_factors["borough_name"], borough_factors["borough_factor"],
              color="#4e79a7", edgecolor="white", label="Borough-level factor")
ax.axhline(global_factor, color="#e15759", linestyle="--", linewidth=2, label=f"Global factor = {global_factor:.3f}")
ax.set_ylabel("Yellow share of total demand (mul_factor)")
ax.set_title("Demand Proxy: Borough vs Global Market Share\n(Why global over-estimates Manhattan, under-estimates outer boroughs)")
ax.legend()
ax.set_ylim(0, 0.35)
for bar, val in zip(bars, borough_factors["borough_factor"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(CHARTS / "02_proxy_borough_vs_global.png", dpi=150)
plt.close()

# 2c. Demand distribution by borough
fig, ax = plt.subplots(figsize=(11, 5))
borough_data = []
for bid, bname in BOROUGH_NAMES.items():
    vals = df[df["borough_id"] == bid]["demand"].values
    borough_data.append({"borough": bname, "vals": vals})
ax.boxplot(
    [d["vals"] for d in borough_data],
    labels=[d["borough"] for d in borough_data],
    patch_artist=True,
    boxprops=dict(facecolor="#aec7e8"),
    medianprops=dict(color="#1f77b4", linewidth=2),
    showfliers=False,
)
ax.set_ylabel("Realized demand (trips per cell)")
ax.set_title("Demand Distribution by Borough (proxy demand, all buckets/DOW)")
plt.tight_layout()
plt.savefig(CHARTS / "03_demand_by_borough.png", dpi=150)
plt.close()

# 2d. Weather correlation heatmap
weather_corr = df[["demand", "temperature_2m", "precipitation",
                    "relative_humidity_2m", "apparent_temperature"]].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(weather_corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, square=True)
ax.set_title("Correlation: Weather Features vs Demand")
plt.tight_layout()
plt.savefig(CHARTS / "04_weather_correlation.png", dpi=150)
plt.close()

print("  Saved: 01–04 EDA charts")

# ─────────────────────────────────────────────────────────────────────────────
# 3. COST PARAMETERS (Cu / Co / τ)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Cost parameters …")

costs_df = costs_mod.compute_costs(train)

# 3a. Co per bucket
co_by_bucket = costs_df.groupby("time_bucket")["Co"].first().reset_index().sort_values("time_bucket")
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(co_by_bucket["time_bucket"], co_by_bucket["Co"], color="#ff7f0e", edgecolor="white")
ax.set_ylabel("Co (overage cost, $)")
ax.set_title("Overage Cost (Co) per Time Bucket\n(All-zones avg fare — same across zones within bucket)")
ax.tick_params(axis="x", rotation=15)
for i, (_, row) in enumerate(co_by_bucket.iterrows()):
    ax.text(i, row["Co"] + 0.2, f"${row['Co']:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(CHARTS / "05_co_by_bucket.png", dpi=150)
plt.close()

# 3b. τ distribution for peak bucket
peak_costs = costs_mod.get_costs_for_scenario(costs_df, PEAK_BUCKET)
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(peak_costs["tau"], bins=40, color="#2ca02c", edgecolor="white", alpha=0.85)
ax.axvline(peak_costs["tau"].median(), color="red", linestyle="--",
           label=f"Median τ = {peak_costs['tau'].median():.3f}")
ax.set_xlabel("Critical fractile τ = Cu/(Cu+Co)")
ax.set_ylabel("Number of zones")
ax.set_title(f"Distribution of Critical Fractile τ — {PEAK_BUCKET}\n"
             "Higher τ → zone allocated more cabs (high-fare zones)")
ax.legend()
plt.tight_layout()
plt.savefig(CHARTS / "06_tau_distribution.png", dpi=150)
plt.close()

# Top-10 highest τ zones
top_tau = peak_costs.nlargest(10, "tau")[["Cu", "Co", "tau"]].reset_index()
top_tau.columns = ["Zone", "Cu ($)", "Co ($)", "τ"]
top_tau[["Cu ($)", "Co ($)", "τ"]] = top_tau[["Cu ($)", "Co ($)", "τ"]].round(3)
print("  Top-10 highest-τ zones (peak bucket):")
print(top_tau.to_string(index=False))

print("  Saved: 05–06 cost-parameter charts")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SOLVER CROSS-CHECK
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Solver cross-check …")

zones = np.array(sorted(peak_costs.index.tolist()))
Cu    = peak_costs.reindex(zones)["Cu"].values
Co    = peak_costs.reindex(zones)["Co"].values
tau   = peak_costs.reindex(zones)["tau"].values

bounds_df = bounds_mod.compute_bounds(
    train, PEAK_BUCKET, PEAK_DOW, floor_alpha=0.15, cap_multiplier=1.5, fleet_size=FLEET
).set_index("PULocationID").reindex(zones).fillna(0)
floors = bounds_df["floor"].values.astype(float)
caps   = np.maximum(bounds_df["cap"].values.astype(float), floors + 1)

emp_model = EmpiricalAllocator().fit(train)
emp_samples = emp_model.get_samples(zones, PEAK_BUCKET, PEAK_DOW)

solver_check = opt_mod.compare_solvers(emp_samples, Cu, Co, floors, caps, FLEET, zones)
print("\n  Solver agreement:")
print(solver_check.to_string(index=False))

max_dev = solver_check["max_diff_vs_slsqp"].max()
status  = "PASS ✓" if max_dev < 300 else "WARN ⚠"
print(f"\n  Max zone deviation across solvers: {max_dev:.2f}  →  {status}")

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
ax.bar(solver_check["solver"], solver_check["runtime_ms"], color=colors, edgecolor="white")
ax.set_ylabel("Runtime (ms)")
ax.set_title("Solver Runtime Comparison (same problem instance)\nAll three must agree within tolerance")
for i, row in solver_check.iterrows():
    ax.text(i, row["runtime_ms"] + 5, f"{row['runtime_ms']:.0f}ms", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(CHARTS / "07_solver_runtimes.png", dpi=150)
plt.close()

print("  Saved: 07 solver chart")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Fitting models …")

print("  → Track A1: Poisson GLM (MLE, all data)")
glm_model = LogLinearDemandModel()
glm_model.fit(train)

print("  → Track A2: LightGBM Quantile Regression (pinball loss at mean τ, all data)")
lgbm_model = LGBMQuantileModel(n_estimators=300, num_leaves=63, seed=42)
lgbm_model.fit(train, None, costs_df)

print("  → Track B: DFL / SPO+ (50 epochs, full-batch L-BFGS-B)")
dfl_model = LinearDFLModel(n_epochs=50, batch_size=999)  # 999 > n_weeks → full batch every epoch
dfl_model.fit(train, Cu, Co, floors, caps, FLEET, zones,
              time_bucket=PEAK_BUCKET, day_of_week=PEAK_DOW)

# ─────────────────────────────────────────────────────────────────────────────
# 6. ALLOCATION RESULTS (q*)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/8] Allocation results …")

# Peak scenario: SLSQP optimal allocation
res_opt = opt_mod.solve(emp_samples, Cu, Co, floors, caps, FLEET, zones, method="slsqp")
q_star  = res_opt.q_star_int.astype(float)

# Status-quo: proportional to historical yellow pickups
hist = train[train["time_bucket"]==PEAK_BUCKET].groupby("PULocationID")["yellow_pickups"].mean().reindex(zones).fillna(0)
q_prop = (hist.values / hist.values.sum() * FLEET).clip(min=0)
q_prop = q_prop * FLEET / q_prop.sum()

# Uniform
q_unif = np.full(len(zones), FLEET / len(zones))

# Save q* CSV
q_df = pd.DataFrame({"PULocationID": zones, "q_star": q_star.astype(int),
                      "Cu": Cu.round(2), "tau": tau.round(4)})
q_df.to_csv(OUT / f"q_star_peak_tuesday.csv", index=False)

# Chart: top-30 zones by allocation
top30_idx = np.argsort(q_star)[-30:][::-1]
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(30), q_star[top30_idx], color="#4e79a7", edgecolor="white")
ax.set_xticks(range(30))
ax.set_xticklabels([f"Z{zones[i]}" for i in top30_idx], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Allocated cabs (q*)")
ax.set_title(f"Top-30 Zones by Allocation — {PEAK_BUCKET}, Tuesday\n"
             f"(Total fleet: {int(q_star.sum()):,} / {FLEET:,})")
plt.tight_layout()
plt.savefig(CHARTS / "08_allocation_top30.png", dpi=150)
plt.close()

# DFL training loss curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(dfl_model.training_log_.epochs, dfl_model.training_log_.spo_losses,
        marker="o", color="#e15759", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("SPO+ Loss")
ax.set_title("DFL Training: SPO+ Loss vs Epoch\n(L-BFGS-B outer optimizer)")
plt.tight_layout()
plt.savefig(CHARTS / "09_dfl_training_loss.png", dpi=150)
plt.close()

print("  Saved: 08–09 allocation charts")
print(f"  Saved: outputs/q_star_peak_tuesday.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 7. BAKE-OFF EVALUATION ON TEST SET  (one q* per (bucket, DOW) — deployment granularity)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Full bake-off on test set …")
print("  Evaluating one q* per (time_bucket, day_of_week) — matching deployment reality")

BUCKETS   = sorted(test["time_bucket"].unique())
DOWS      = sorted(test["day_of_week"].unique())
TEST_WEEK = int(test["week"].max())   # last full test week

bakeoff_data = {"empirical": [], "glm_pto": [], "lgbm_pto": [], "dfl": []}
scenario_rows = []

for bucket in BUCKETS:
    slot_costs = costs_mod.get_costs_for_scenario(costs_df, bucket)
    slot_zones = np.array(sorted(slot_costs.index.tolist()))
    Cu_s  = slot_costs.reindex(slot_zones)["Cu"].values
    Co_s  = slot_costs.reindex(slot_zones)["Co"].values
    tau_s = slot_costs.reindex(slot_zones)["tau"].values

    for dow in DOWS:
        sub = test[(test["time_bucket"] == bucket) & (test["day_of_week"] == dow)]
        if sub.empty:
            continue

        # Realized demand: average across all test weeks for this (bucket, DOW)
        d_true = sub.groupby("PULocationID")["demand"].mean().reindex(slot_zones).fillna(0).values.astype(float)

        # Bounds for this specific (bucket, DOW)
        b_df = bounds_mod.compute_bounds(
            train, bucket, dow, floor_alpha=0.15, cap_multiplier=1.5, fleet_size=FLEET
        ).set_index("PULocationID").reindex(slot_zones).fillna(0)
        fl_s = b_df["floor"].values.astype(float)
        cp_s = np.maximum(b_df["cap"].values.astype(float), fl_s + 1)

        # Empirical model → q*
        samp_s = emp_model.get_samples(slot_zones, bucket, dow)
        res_emp_s = opt_mod.solve(samp_s, Cu_s, Co_s, fl_s, cp_s, FLEET, slot_zones, method="slsqp")
        q_emp_s = res_emp_s.q_star_int.astype(float)

        # Empirical deviation (shared across model tracks for blending)
        emp_dev_s = samp_s - samp_s.mean(axis=0)

        # GLM → q*
        _, glm_pred_s = glm_model.predict_for_scenario(df, bucket, dow, TEST_WEEK, tau_s, slot_zones)
        glm_mean_s  = np.maximum(glm_pred_s, 0)
        glm_samp_s  = np.maximum(glm_mean_s + emp_dev_s, 0)
        res_glm_s   = opt_mod.solve(glm_samp_s, Cu_s, Co_s, fl_s, cp_s, FLEET, slot_zones, method="slsqp")
        q_glm_s     = res_glm_s.q_star_int.astype(float)

        # LGBM quantile → q*  (pinball loss at mean τ → directly predicts the demand quantile)
        lgbm_zones_s, lgbm_pred_s = lgbm_model.predict_for_scenario(df, bucket, dow, TEST_WEEK)
        lgbm_mean_s = np.maximum(
            pd.Series(lgbm_pred_s, index=lgbm_zones_s)
              .reindex(slot_zones)
              .fillna(pd.Series(glm_mean_s, index=slot_zones))
              .values, 0
        )
        lgbm_samp_s = np.maximum(lgbm_mean_s + emp_dev_s, 0)
        res_lgbm_s  = opt_mod.solve(lgbm_samp_s, Cu_s, Co_s, fl_s, cp_s, FLEET, slot_zones, method="slsqp")
        q_lgbm_s    = res_lgbm_s.q_star_int.astype(float)

        # DFL → q*
        _, dfl_pred_s = dfl_model.predict(df, bucket, dow, TEST_WEEK)
        dfl_ser_s   = pd.Series(dfl_pred_s, index=slot_zones)
        dfl_mean_s  = np.maximum(np.array([dfl_ser_s.get(z, glm_mean_s[i]) for i, z in enumerate(slot_zones)]), 0)
        dfl_samp_s  = np.maximum(dfl_mean_s + emp_dev_s, 0)
        res_dfl_s   = opt_mod.solve(dfl_samp_s, Cu_s, Co_s, fl_s, cp_s, FLEET, slot_zones, method="slsqp")
        q_dfl_s     = res_dfl_s.q_star_int.astype(float)

        for label, q_s in [("empirical", q_emp_s), ("glm_pto", q_glm_s), ("lgbm_pto", q_lgbm_s), ("dfl", q_dfl_s)]:
            m = compute_all_metrics(q_s, d_true, Cu_s, Co_s, tau_s, FLEET)
            m["revenue"]  = revenue_generated(q_s, d_true, Cu_s)
            m["foregone"] = revenue_foregone(q_s, d_true, Cu_s)
            m["bucket"]   = bucket
            m["dow"]      = dow
            bakeoff_data[label].append(m)
            scenario_rows.append({"model": label, "bucket": bucket, "dow": dow,
                                   "revenue": m["revenue"], "nv_cost": m["nv_cost"]})

        rev_by_label = {label: revenue_generated(bakeoff_data[label][-1].get("_q"), d_true, Cu_s)
                        if "_q" in bakeoff_data[label][-1] else 0
                        for label in bakeoff_data}
        emp_r   = revenue_generated(q_emp_s,  d_true, Cu_s)
        glm_r   = bakeoff_data["glm_pto"][-1]["revenue"]
        lgbm_r  = bakeoff_data["lgbm_pto"][-1]["revenue"]
        dfl_r   = bakeoff_data["dfl"][-1]["revenue"]
        print(f"  ✓ bucket={bucket}, dow={dow}  "
              f"emp=${emp_r:>8,.0f}  glm=${glm_r:>8,.0f}  "
              f"lgbm=${lgbm_r:>8,.0f}  dfl=${dfl_r:>8,.0f}")

bakeoff_df = bakeoff_summary({k: v for k, v in bakeoff_data.items()})
bakeoff_df.to_csv(OUT / "bakeoff_results.csv", index=False)
pd.DataFrame(scenario_rows).to_csv(OUT / "bakeoff_by_scenario.csv", index=False)

print("\n  === BAKE-OFF RESULTS (all buckets × all DOWs) ===")
# Attach mean revenue per model into the summary table
rev_series = pd.Series({k: np.mean([r["revenue"] for r in v]) for k, v in bakeoff_data.items()}, name="mean_revenue")
bakeoff_df = bakeoff_df.merge(rev_series.reset_index().rename(columns={"index":"model"}), on="model", how="left")
print(bakeoff_df[["model","mean_nv_cost","mean_fill_rate","mean_revenue","mean_pinball","rank"]].to_string(index=False))

# Business impact: average across all scenarios
rev_by_model = {k: np.mean([r["revenue"] for r in v]) for k, v in bakeoff_data.items()}
print(f"\n  === AVERAGE REVENUE PER (BUCKET, DOW) SLOT ===")
for model, rev in rev_by_model.items():
    print(f"  {model:<12}  ${rev:>12,.0f}")

# For chart context: use peak Tuesday numbers from the full loop
peak_rows = [r for r in scenario_rows
             if r["bucket"] == PEAK_BUCKET and r["dow"] == PEAK_DOW and r["model"] == "empirical"]
rev_opt = peak_rows[0]["revenue"] if peak_rows else rev_by_model["empirical"]

# Status-quo revenue for peak Tuesday — average across test weeks (49-52)
test_peak = test[(test["time_bucket"]==PEAK_BUCKET) & (test["day_of_week"]==PEAK_DOW)]
d_true_peak = test_peak.groupby("PULocationID")["demand"].mean().reindex(zones).fillna(0).values.astype(float)
d_true_mean = d_true_peak  # alias used by charts below (under/overage, understocked zones, what-if)
rev_prop = revenue_generated(q_prop, d_true_peak, Cu)
rev_unif = revenue_generated(q_unif, d_true_peak, Cu)

print(f"\n  === BUSINESS IMPACT (peak Tuesday slot) ===")
print(f"  Revenue — Optimized:   ${rev_opt:>12,.0f}")
print(f"  Revenue — Status quo:  ${rev_prop:>12,.0f}")
print(f"  Revenue — Uniform:     ${rev_unif:>12,.0f}")
print(f"  Gain vs status quo:    ${rev_opt - rev_prop:>12,.0f} / slot")
print(f"  Annualized (52 wks):   ${(rev_opt - rev_prop)*52:>12,.0f}")

# Bake-off chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
model_labels = bakeoff_df["model"].tolist()
x = np.arange(len(model_labels))

COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
axes[0].bar(x, bakeoff_df["mean_nv_cost"]/1e6, color=COLORS[:len(model_labels)], edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(model_labels)
axes[0].set_ylabel("Mean NV Cost ($M)")
axes[0].set_title("Newsvendor Cost (lower = better)")
for i, v in enumerate(bakeoff_df["mean_nv_cost"]/1e6):
    axes[0].text(i, v + 0.01, f"${v:.2f}M", ha="center", fontsize=10)

axes[1].bar(x, bakeoff_df["mean_pinball"], color=COLORS[:len(model_labels)], edgecolor="white")
axes[1].set_xticks(x); axes[1].set_xticklabels(model_labels)
axes[1].set_ylabel("Mean Pinball Loss")
axes[1].set_title("Pinball Loss at τ (lower = better)")
for i, v in enumerate(bakeoff_df["mean_pinball"]):
    axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)

fig.suptitle("Model Bake-off: Empirical vs GLM vs LGBM vs DFL", fontweight="bold")
plt.tight_layout()
plt.savefig(CHARTS / "10_bakeoff.png", dpi=150)
plt.close()

# Revenue comparison chart
fig, ax = plt.subplots(figsize=(9, 5))
models  = ["Optimized\n(Empirical)", "Status Quo\n(Proportional)", "Uniform"]
revenues = [rev_opt, rev_prop, rev_unif]
colors   = ["#4e79a7", "#f28e2b", "#999999"]
bars = ax.bar(models, revenues, color=colors, edgecolor="white")
ax.set_ylabel("Revenue per Tuesday midday slot ($)")
ax.set_title("Revenue Comparison: Optimized vs Baselines\n(Same 13,000 cabs, different zone allocation)")
for bar, v in zip(bars, revenues):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
            f"${v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
gain_text = f"Gain vs status quo:\n${rev_opt-rev_prop:,.0f}/slot\n${(rev_opt-rev_prop)*52:,.0f}/year"
ax.text(0.98, 0.97, gain_text, transform=ax.transAxes, ha="right", va="top",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
plt.tight_layout()
plt.savefig(CHARTS / "11_revenue_comparison.png", dpi=150)
plt.close()

# Under vs Overage pie
under_total = float((Cu * np.maximum(d_true_mean - q_star, 0)).sum())
over_total  = float((Co * np.maximum(q_star - d_true_mean, 0)).sum())
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([under_total, over_total],
       labels=[f"Underage\n${under_total:,.0f}", f"Overage\n${over_total:,.0f}"],
       colors=["#e15759", "#4e79a7"], autopct="%1.1f%%", startangle=90,
       textprops={"fontsize": 12})
ax.set_title("NV Cost Breakdown: Underage vs Overage\n(Optimized allocation, peak Tuesday)")
plt.tight_layout()
plt.savefig(CHARTS / "12_under_vs_overage.png", dpi=150)
plt.close()

# Top understocked zones
zone_deficit = d_true_mean - q_star
top10_under = pd.Series(zone_deficit, index=zones).nlargest(10)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(range(10), top10_under.values[::-1], color="#e15759", edgecolor="white")
ax.set_yticks(range(10))
ax.set_yticklabels([f"Zone {z}" for z in top10_under.index[::-1]])
ax.set_xlabel("Unserved demand (trips)")
ax.set_title("Top-10 Most Understocked Zones — Peak Tuesday\n(Demand far exceeds allocation)")
plt.tight_layout()
plt.savefig(CHARTS / "13_understocked_zones.png", dpi=150)
plt.close()

print("  Saved: 10–13 bake-off & business charts")

# ─────────────────────────────────────────────────────────────────────────────
# 8. PROXY SENSITIVITY & WHAT-IF
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8/8] Proxy sensitivity & what-if scenarios …")

proxy_q = {}
for agg in ("global", "borough", "cluster"):
    df_agg  = proxy_mod.add_demand(df_full, aggregation=agg)
    tr_agg  = data_mod.temporal_split(df_agg)[0]
    emp_agg = EmpiricalAllocator().fit(tr_agg)
    samp_agg = emp_agg.get_samples(zones, PEAK_BUCKET, PEAK_DOW)
    r_agg   = opt_mod.solve(samp_agg, Cu, Co, floors, caps, FLEET, zones, method="water_filling")
    proxy_q[agg] = r_agg.q_star

proxy_sens = proxy_sensitivity_summary(proxy_q, zones)
print("\n  Proxy sensitivity:")
print(proxy_sens.to_string(index=False))

# Proxy scatter: global vs borough
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(proxy_q["global"], proxy_q["borough"], alpha=0.5, s=20, color="#4e79a7")
lim = max(proxy_q["global"].max(), proxy_q["borough"].max()) + 10
ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect agreement")
ax.set_xlabel("Global proxy → q* per zone")
ax.set_ylabel("Borough proxy → q* per zone")
ax.set_title("Proxy Sensitivity: Global vs Borough Allocation\n"
             "Points off diagonal = zones where proxy choice matters")
ax.legend()
plt.tight_layout()
plt.savefig(CHARTS / "14_proxy_sensitivity.png", dpi=150)
plt.close()

# What-if: fleet reduction, rainy day, equity floor
what_if_results = []
base_rev = revenue_generated(q_star, d_true_mean, Cu)

for label, cfg_kwargs in [
    ("Baseline\n(13k fleet)", {}),
    ("Fleet cut\n(11k cabs)", {"fleet_size": 11000}),
    ("Rainy day\n(precip=0.3)", {"weather_override": {"precipitation": 0.3}}),
    ("Equity floor\n(α=0.25)", {"floor_alpha": 0.25}),
]:
    fleet_wi = cfg_kwargs.get("fleet_size", FLEET)
    alpha_wi = cfg_kwargs.get("floor_alpha", 0.15)
    bounds_wi = bounds_mod.compute_bounds(
        train, PEAK_BUCKET, PEAK_DOW, floor_alpha=alpha_wi, fleet_size=fleet_wi
    ).set_index("PULocationID").reindex(zones).fillna(0)
    fl_wi = bounds_wi["floor"].values.astype(float)
    cp_wi = np.maximum(bounds_wi["cap"].values.astype(float), fl_wi + 1)
    r_wi  = opt_mod.solve(emp_samples, Cu, Co, fl_wi, cp_wi, fleet_wi, zones, method="slsqp")
    rev_wi = revenue_generated(r_wi.q_star_int.astype(float), d_true_mean, Cu)
    what_if_results.append({"scenario": label, "fleet": fleet_wi,
                             "revenue": rev_wi, "delta_vs_base": rev_wi - base_rev,
                             "fleet_used": r_wi.fleet_used})

wi_df = pd.DataFrame(what_if_results)
print("\n  What-if scenarios:")
print(wi_df[["scenario","fleet","revenue","delta_vs_base"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors_wi = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
bars = ax.bar(wi_df["scenario"].str.replace("\n", " "), wi_df["revenue"],
              color=colors_wi, edgecolor="white")
ax.set_ylabel("Revenue per slot ($)")
ax.set_title("What-If Scenarios: Revenue Impact of Parameter Changes")
for bar, row in zip(bars, wi_df.itertuples()):
    delta_str = f"{row.delta_vs_base:+,.0f}" if row.delta_vs_base != 0 else "baseline"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
            f"${row.revenue:,.0f}\n({delta_str})", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(CHARTS / "15_what_if.png", dpi=150)
plt.close()

print("  Saved: 14–15 sensitivity & what-if charts")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("OUTPUTS SAVED")
print(f"{'='*65}")
print(f"  outputs/bakeoff_results.csv")
print(f"  outputs/q_star_peak_tuesday.csv")
print(f"  outputs/charts/  ({len(list(CHARTS.glob('*.png')))} PNG files)")
print(f"\nAll done.")
