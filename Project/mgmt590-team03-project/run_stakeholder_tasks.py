"""Stakeholder deliverable runner — Tasks 1 & 2.

Usage:
    python run_stakeholder_tasks.py --task both
    python run_stakeholder_tasks.py --task 1 --demand-model lgbm
    python run_stakeholder_tasks.py --slots "1:06:00-08:59,5:19:00-23:59,6:09:00-15:59" --task 2

Slots format: comma-separated  <dow>:<time_bucket>
  e.g.  1:06:00-08:59  = Monday early AM
        5:19:00-23:59  = Friday evening
        6:09:00-15:59  = Saturday midday
"""

from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from engine.config import OptimizationConfig
from engine import data as data_mod
from engine import proxy as proxy_mod
from engine import costs as costs_mod
from engine import bounds as bounds_mod
from engine import optimize as opt_mod
from engine.evaluate import (
    compute_all_metrics, bakeoff_summary,
    fleet_adjustment_cost, borough_summary,
)
from engine.models.empirical import EmpiricalAllocator
from engine.models.linear_demand import LogLinearDemandModel
from engine.models.qr_lgbm import LGBMQuantileModel


BOROUGH_NAMES = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}


def _df_to_md(df: pd.DataFrame) -> str:
    """Render DataFrame as a simple Markdown table without requiring tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(round(v, 4) if isinstance(v, float) else v) for v in row) + " |")
    return "\n".join([header, sep] + rows)

DEFAULT_SLOTS = [
    (1, "06:00-08:59"),   # Monday early AM
    (5, "19:00-23:59"),   # Friday evening
    (6, "09:00-15:59"),   # Saturday midday
]

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NYC Taxi Stakeholder Tasks 1 & 2")
    p.add_argument(
        "--slots",
        default=None,
        help='Slots as "dow:bucket,..." e.g. "1:06:00-08:59,5:19:00-23:59"',
    )
    p.add_argument("--task", default="both", choices=["1", "2", "both"])
    p.add_argument(
        "--demand-model",
        default="auto",
        choices=["auto", "saa", "glm", "lgbm"],
        help="Force demand model or auto-select winner of bake-off",
    )
    p.add_argument("--fleet", type=int, default=13000)
    p.add_argument("--rental-rate", type=float, default=45.0)
    p.add_argument("--standdown-saving", type=float, default=20.0)
    p.add_argument(
        "--week", type=int, default=None,
        help="ISO week number for GLM/LGBM prediction (e.g. 16). SAA ignores this.",
    )
    p.add_argument(
        "--year", type=int, default=2026,
        help="Year for GLM/LGBM prediction (default 2026).",
    )
    return p.parse_args()


def parse_slots(slot_str: Optional[str]) -> List[Tuple[int, str]]:
    if slot_str is None:
        return DEFAULT_SLOTS
    slots = []
    for s in slot_str.split(","):
        s = s.strip()
        dow_str, bucket = s.split(":", 1)
        slots.append((int(dow_str), bucket))
    return slots


# ── Data loading (shared across all slots) ────────────────────────────────────

def load_data(fleet: int, seed: int = 42):
    print("\n[DATA] Loading 3-year dataset …")
    df_full = data_mod.prepare(n_clusters=5, seed=seed)
    df = proxy_mod.add_demand(df_full, aggregation="borough")
    train, val, test = data_mod.temporal_split(df)
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return df, df_full, train, val, test


# ── Step 0: model selection bake-off ─────────────────────────────────────────

def run_bakeoff(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    fleet: int,
    seed: int = 42,
    force_model: Optional[str] = None,
) -> Tuple[str, dict]:
    """Train SAA/GLM/LGBM once; bake-off on test; return winner name + model dict."""

    # Use a representative slot for cost/bounds — we need some scenario to compute tau
    # For bake-off we use the flat-Co (Model A) config with bucket="09:00-15:59", dow=2
    _bakeoff_bucket = "09:00-15:59"
    _bakeoff_dow = 2

    costs_flat = costs_mod.compute_costs(
        train, cu_multiplier=1.0, co_multiplier=1.0, co_mode="flat"
    )

    all_zones = np.array(sorted(train["PULocationID"].unique()))
    sc = costs_flat[costs_flat["time_bucket"] == _bakeoff_bucket].set_index("PULocationID")
    sc = sc.reindex(all_zones)
    Cu = sc["Cu"].values
    Co = sc["Co"].values
    tau_flat = sc["tau"].values

    bounds_df = bounds_mod.compute_bounds(
        train, time_bucket=_bakeoff_bucket, day_of_week=_bakeoff_dow,
        floor_alpha=0.15, cap_multiplier=1.5, floor_overrides={}, cap_overrides={},
        fleet_size=fleet,
    ).set_index("PULocationID").reindex(all_zones).fillna(0)
    floors = bounds_df["floor"].values.astype(float)
    caps = np.maximum(bounds_df["cap"].values.astype(float), floors + 1)

    print("\n[STEP 0] Training demand models …")

    # SAA (empirical) — always needed: it's the dispersion/blend base for GLM & LGBM too
    emp = EmpiricalAllocator().fit(train)
    emp_samples = emp.get_samples(all_zones, _bakeoff_bucket, _bakeoff_dow)

    forced = None
    if force_model and force_model != "auto":
        forced = {"saa": "empirical", "glm": "glm", "lgbm": "lgbm"}.get(force_model, force_model)

    # Only train the model(s) actually needed
    glm = None
    if forced is None or forced == "glm":
        glm = LogLinearDemandModel(seed=seed)
        glm.fit(train)

    lgbm = None
    if forced is None or forced == "lgbm":
        tau_df = costs_flat[["PULocationID", "time_bucket", "tau"]].copy()
        lgbm = LGBMQuantileModel(seed=seed)
        lgbm.fit(train, val, tau_df)

    if forced is not None:
        print(f"\n  => Operational model forced via --demand-model: {forced} "
              f"(skipping bake-off training/evaluation of the other models)")
        return forced, {"empirical": emp, "glm": glm, "lgbm": lgbm}

    print("\n[STEP 0] Baking off on test set …")
    bakeoff_results = {"empirical": [], "glm": [], "lgbm": []}

    for week_test in sorted(test["week"].unique()):
        sub_w = test[(test["time_bucket"] == _bakeoff_bucket) & (test["day_of_week"] == _bakeoff_dow)
                     & (test["week"] == week_test)]
        if sub_w.empty:
            continue
        demand_true = (
            sub_w.groupby("PULocationID")["demand"].mean()
            .reindex(all_zones).fillna(0).values.astype(float)
        )

        # SAA
        res_emp = opt_mod.solve_water_filling(emp_samples, Cu, Co, floors, caps, fleet, all_zones)

        # GLM
        glm_zones, glm_pred = glm.predict_for_scenario(
            train, _bakeoff_bucket, _bakeoff_dow, week_test, tau=tau_flat, zones=all_zones
        )
        glm_series = pd.Series(glm_pred, index=glm_zones)
        glm_mean = np.array([glm_series.get(z, emp_samples.mean(axis=0)[i]) for i, z in enumerate(all_zones)])
        glm_blend = np.maximum(glm_mean + (emp_samples - emp_samples.mean(axis=0)), 0)
        res_glm = opt_mod.solve_water_filling(glm_blend, Cu, Co, floors, caps, fleet, all_zones)

        # LGBM
        lgbm_zones, lgbm_pred = lgbm.predict_for_scenario(
            train, _bakeoff_bucket, _bakeoff_dow, week_test
        )
        lgbm_series = pd.Series(lgbm_pred, index=lgbm_zones)
        lgbm_mean = np.array([lgbm_series.get(z, emp_samples.mean(axis=0)[i]) for i, z in enumerate(all_zones)])
        lgbm_blend = np.maximum(lgbm_mean + (emp_samples - emp_samples.mean(axis=0)), 0)
        res_lgbm = opt_mod.solve_water_filling(lgbm_blend, Cu, Co, floors, caps, fleet, all_zones)

        for name, res in [("empirical", res_emp), ("glm", res_glm), ("lgbm", res_lgbm)]:
            m = compute_all_metrics(res.q_star_int.astype(float), demand_true, Cu, Co, tau_flat, fleet)
            m["week"] = week_test
            bakeoff_results[name].append(m)

    bakeoff_df = bakeoff_summary(bakeoff_results)
    print("\nBake-off results (test set):")
    print(bakeoff_df[["model", "mean_nv_cost", "mean_fill_rate", "mean_pinball", "rank"]].to_string(index=False))

    bakeoff_df.to_csv(OUT_DIR / "bakeoff_model_selection.csv", index=False)

    winner = bakeoff_df.iloc[0]["model"]
    print(f"\n  => Operational model: {winner}")
    return winner, {"empirical": emp, "glm": glm, "lgbm": lgbm}


# ── Per-slot cost/bounds/samples helper ──────────────────────────────────────

def build_slot_inputs(
    train: pd.DataFrame,
    dow: int,
    time_bucket: str,
    fleet: int,
    co_mode: str = "flat",
    co_borough_multipliers: Optional[dict] = None,
    co_zone_overrides: Optional[dict] = None,
    seed: int = 42,
):
    costs_df = costs_mod.compute_costs(
        train,
        cu_multiplier=1.0,
        co_multiplier=1.0,
        co_mode=co_mode,
        co_borough_multipliers=co_borough_multipliers,
        co_zone_overrides=co_zone_overrides,
    )
    sc = costs_df[costs_df["time_bucket"] == time_bucket].set_index("PULocationID")
    zones = np.array(sorted(sc.index.tolist()))
    sc = sc.reindex(zones)
    Cu = sc["Cu"].values
    Co = sc["Co"].values
    tau = sc["tau"].values

    bounds_df = bounds_mod.compute_bounds(
        train, time_bucket=time_bucket, day_of_week=dow,
        floor_alpha=0.15, cap_multiplier=1.5,
        floor_overrides={}, cap_overrides={}, fleet_size=fleet,
    ).set_index("PULocationID").reindex(zones).fillna(0)
    floors = bounds_df["floor"].values.astype(float)
    caps = np.maximum(bounds_df["cap"].values.astype(float), floors + 1)

    emp = EmpiricalAllocator().fit(train)
    emp_samples = emp.get_samples(zones, time_bucket, dow)

    return zones, Cu, Co, tau, floors, caps, emp_samples


def realized_nv_test(
    q_int: np.ndarray,
    test: pd.DataFrame,
    zones: np.ndarray,
    time_bucket: str,
    dow: int,
    Cu: np.ndarray,
    Co: np.ndarray,
    tau: np.ndarray,
    fleet: int,
    week: Optional[int] = None,
    year: Optional[int] = None,
) -> dict:
    """Realized newsvendor cost on the held-out test set.

    If `week` is given, evaluate against the true demand for that *single*
    (dow, time_bucket, week, year) slice only — no averaging.  If `week` is
    None, average the metrics across every test week containing this
    (dow, time_bucket).
    """
    sub = test[(test["time_bucket"] == time_bucket) & (test["day_of_week"] == dow)]
    if week is not None:
        sub = sub[sub["week"] == week]
        if year is not None and "year" in sub.columns:
            sub = sub[sub["year"] == year]
        if sub.empty:
            return {}
        d_true = sub.groupby("PULocationID")["demand"].mean().reindex(zones).fillna(0).values.astype(float)
        return compute_all_metrics(q_int.astype(float), d_true, Cu, Co, tau, fleet)

    if sub.empty:
        return {}
    week_metrics = []
    for w in sorted(sub["week"].unique()):
        sw = sub[sub["week"] == w]
        d_true = sw.groupby("PULocationID")["demand"].mean().reindex(zones).fillna(0).values.astype(float)
        week_metrics.append(compute_all_metrics(q_int.astype(float), d_true, Cu, Co, tau, fleet))
    if not week_metrics:
        return {}
    df_m = pd.DataFrame(week_metrics)
    return {k: df_m[k].mean() for k in df_m.columns}


# ── Task 1: Model A vs B ──────────────────────────────────────────────────────

def _build_demand_samples(
    winner: str,
    models: dict,
    train: pd.DataFrame,
    zones: np.ndarray,
    time_bucket: str,
    dow: int,
    tau: np.ndarray,
    week: Optional[int] = None,
    year: int = 2026,
) -> np.ndarray:
    """Return a blended demand sample matrix for the given slot using the winning model.

    SAA/empirical:
        Uses the full distribution of historical (dow, bucket) rows as-is.
        week/year are ignored — SAA needs many samples to be meaningful.

    GLM/LGBM:
        The model predicts a *point estimate* of demand for the specific
        (dow, bucket, week, year). We then add empirical deviations (zero-mean
        noise from training history) around that point to produce a sample matrix
        of the same shape. This shifts the distribution centre to the model's
        week-specific forecast while preserving realistic dispersion.

        If week is None, falls back to the median training week (less granular).
    """
    emp = models["empirical"]
    emp_samples = emp.get_samples(zones, time_bucket, dow)   # (n_weeks, n_zones)
    emp_mean = emp_samples.mean(axis=0)                       # (n_zones,)
    emp_deviation = emp_samples - emp_mean                    # zero-mean noise

    if winner == "empirical":
        # SAA: ignore week/year, use full distribution
        return emp_samples

    # For GLM/LGBM: choose which data slice to pass for feature extraction
    # Filter to the requested year if possible; fall back to all training data
    df_for_pred = train[train["year"] == year] if year is not None and "year" in train.columns else train
    if df_for_pred.empty:
        df_for_pred = train

    # Use the specified week; if None, use median training week
    pred_week = week if week is not None else int(train["week"].median())

    if winner == "glm":
        glm = models["glm"]
        _, glm_pred = glm.predict_for_scenario(
            df_for_pred, time_bucket, dow, pred_week, tau=tau, zones=zones
        )
        glm_series = pd.Series(glm_pred, index=zones)
        model_mean = np.array([glm_series.get(z, emp_mean[i]) for i, z in enumerate(zones)])
        print(f"      [GLM] Predicting for dow={dow} bucket={time_bucket} week={pred_week} year={year}")

    elif winner == "lgbm":
        lgbm = models["lgbm"]
        lgbm_zones, lgbm_pred = lgbm.predict_for_scenario(
            df_for_pred, time_bucket, dow, pred_week
        )
        lgbm_series = pd.Series(lgbm_pred, index=lgbm_zones)
        model_mean = np.array([lgbm_series.get(z, emp_mean[i]) for i, z in enumerate(zones)])
        print(f"      [LGBM] Predicting for dow={dow} bucket={time_bucket} week={pred_week} year={year}")

    else:
        return emp_samples   # fallback

    return np.maximum(model_mean + emp_deviation, 0)


def run_task1(
    train: pd.DataFrame,
    test: pd.DataFrame,
    models: dict,
    winner: str,
    dow: int,
    time_bucket: str,
    fleet: int,
    borough_map: dict,
    rental_rate: float = 45.0,
    standdown_saving: float = 20.0,
    week: Optional[int] = None,
    year: int = 2026,
) -> dict:
    week_label = f"_wk{week}yr{year}" if week is not None else "_allwks"
    slot_label = f"dow{dow}_{time_bucket.replace(':','').replace('-','_')}{week_label}"
    print(f"\n  [Task 1] {slot_label} — Model A (flat) vs Model B (borough Co) …")
    print(f"           Demand model: {winner}  week={week}  year={year}")

    from engine.config import _DEFAULT_BOROUGH_MULTIPLIERS, _DEFAULT_ZONE_OVERRIDES

    results = {}
    for label, co_mode, multipliers, overrides in [
        ("A_flat", "flat", None, None),
        ("B_borough", "borough", _DEFAULT_BOROUGH_MULTIPLIERS, _DEFAULT_ZONE_OVERRIDES),
    ]:
        zones, Cu, Co, tau, floors, caps, emp_samples = build_slot_inputs(
            train, dow, time_bucket, fleet,
            co_mode=co_mode,
            co_borough_multipliers=multipliers,
            co_zone_overrides=overrides,
        )

        # Use the winning demand model's sample matrix (not raw empirical for GLM/LGBM)
        demand_samples = _build_demand_samples(
            winner, models, train, zones, time_bucket, dow, tau, week=week, year=year
        )

        res_wf = opt_mod.solve_water_filling(demand_samples, Cu, Co, floors, caps, fleet, zones)
        res_sl = opt_mod.solve_slsqp(demand_samples, Cu, Co, floors, caps, fleet, zones)

        test_metrics_wf = realized_nv_test(
            res_wf.q_star_int, test, zones, time_bucket, dow, Cu, Co, tau, fleet,
            week=week, year=year,
        )
        bor_summary_df = borough_summary(res_wf.q_star, zones, borough_map)

        results[label] = {
            "zones": zones, "Cu": Cu, "Co": Co, "tau": tau,
            "floors": floors, "caps": caps,
            "emp_samples": emp_samples,
            "demand_samples": demand_samples,
            "res_wf": res_wf, "res_sl": res_sl,
            "shadow_price": res_wf.shadow_price,   # λ(F₀) for this cost model
            "test_metrics": test_metrics_wf,
            "borough_summary": bor_summary_df,
        }

    # 3-way cross-check under Model B
    zB = results["B_borough"]
    try:
        cross_check = opt_mod.compare_solvers(
            zB["demand_samples"], zB["Cu"], zB["Co"], zB["floors"], zB["caps"], fleet, zB["zones"]
        )
    except Exception as e:
        warnings.warn(f"3-way cross-check failed: {e}")
        cross_check = pd.DataFrame()

    # Top-5 tau-shift zones
    tau_A = results["A_flat"]["tau"]
    tau_B = results["B_borough"]["tau"]
    zones_A = results["A_flat"]["zones"]
    tau_shift = np.abs(tau_B - tau_A)
    top5_idx = np.argsort(-tau_shift)[:5]
    tau_table = pd.DataFrame({
        "PULocationID": zones_A[top5_idx],
        "tau_A": tau_A[top5_idx].round(4),
        "tau_B": tau_B[top5_idx].round(4),
        "delta_tau": tau_shift[top5_idx].round(4),
    })

    print(f"\n    Borough q* totals — Model A vs B:")
    bor_A = results["A_flat"]["borough_summary"].rename(columns={"total_q": "q_A"})
    bor_B = results["B_borough"]["borough_summary"].rename(columns={"total_q": "q_B"})
    bor_merged = bor_A.merge(bor_B[["borough_id", "q_B"]], on="borough_id")
    bor_merged["borough"] = bor_merged["borough_id"].map(BOROUGH_NAMES)
    bor_merged["delta_q"] = (bor_merged["q_B"] - bor_merged["q_A"]).round(1)
    print(bor_merged[["borough", "q_A", "q_B", "delta_q"]].to_string(index=False))

    lam_A = results["A_flat"]["shadow_price"] or 0.0
    lam_B = results["B_borough"]["shadow_price"] or 0.0
    print(f"\n    Shadow price λ(F₀=13000) — A (flat): {lam_A:.4f}   B (borough): {lam_B:.4f}")
    print(f"    (λ = marginal NV cost reduction per extra cab at fixed fleet F₀)")

    print(f"\n    Realized NV cost on test — A: {results['A_flat']['test_metrics'].get('nv_cost', float('nan')):.0f}  "
          f"B: {results['B_borough']['test_metrics'].get('nv_cost', float('nan')):.0f}")

    print(f"\n    Top-5 tau-shift zones (A→B):")
    print(tau_table.to_string(index=False))

    if not cross_check.empty:
        print(f"\n    3-way solver cross-check (Model B):")
        print(cross_check.to_string(index=False))

    return {
        "slot": slot_label,
        "model_A": results["A_flat"],
        "model_B": results["B_borough"],
        "cross_check": cross_check,
        "tau_shift_top5": tau_table,
        "borough_comparison": bor_merged,
        "week": week,
        "year": year,
    }


# ── Task 2: Elastic fleet ─────────────────────────────────────────────────────

def run_task2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    task1_result: dict,
    fleet: int,
    rental_rate: float,
    standdown_saving: float,
    dow: int,
    time_bucket: str,
) -> dict:
    slot_label = task1_result["slot"]
    print(f"\n  [Task 2] {slot_label} — Elastic fleet …")

    from engine.config import _DEFAULT_BOROUGH_MULTIPLIERS, _DEFAULT_ZONE_OVERRIDES

    # Model C uses Model B costs + elastic fleet
    zB = task1_result["model_B"]
    zones = zB["zones"]
    Cu, Co, tau = zB["Cu"], zB["Co"], zB["tau"]
    floors, caps = zB["floors"], zB["caps"]
    demand_samples = zB["demand_samples"]  # winning model's blended samples
    fleet_min = float(floors.sum())

    # Water-filling elastic (primary)
    elastic_wf = opt_mod.solve_elastic_fleet(
        demand_samples, Cu, Co, floors, caps,
        F0=float(fleet), r=rental_rate, s=standdown_saving,
        zones=zones, fleet_min=fleet_min,
    )

    # SLSQP elastic (cross-check)
    elastic_sl = opt_mod.solve_elastic_slsqp(
        demand_samples, Cu, Co, floors, caps,
        F0=float(fleet), r=rental_rate, s=standdown_saving,
        zones=zones, fleet_min=fleet_min,
    )

    # Model A / B fixed-fleet costs: reuse the realized-test-set NV cost from Task 1
    # (same basis — actual held-out demand for this dow/bucket/week/year — so A, B, C
    # are directly comparable, matching the PDF's "Total cost ... vs Model B" framing).
    zA = task1_result["model_A"]
    week = task1_result.get("week")
    year = task1_result.get("year")
    nv_A = float(zA["test_metrics"].get("nv_cost", float("nan")))
    nv_B = float(zB["test_metrics"].get("nv_cost", float("nan")))
    adj_cost_B = 0.0  # fixed fleet

    # Model C cost: realized test-set NV cost under Model B's costs at the elastic
    # allocation q*_C, plus the fleet adjustment cost for moving F0 -> F*
    if elastic_wf.status == "optimal:stay":
        # F* == F0: Model C's allocation problem is identical to Model B's (same
        # Cu/Co/floors/caps/fleet/demand_samples), so reuse Model B's q*/realized
        # NV cost directly. Otherwise solve_elastic_fleet's independent
        # lambda-search + integer-rounding can land on a slightly different q*
        # than solve_water_filling's, and — since nv is evaluated against ONE
        # specific realized day, not the smooth SAA expectation — even a few
        # units shifted between zones can swing nv_C away from nv_B despite
        # both being at the same fleet size.
        nv_C = nv_B
    else:
        metrics_C = realized_nv_test(
            elastic_wf.q_star_int, test, zones, time_bucket, dow,
            Cu, Co, tau, fleet, week=week, year=year,
        )
        nv_C = float(metrics_C.get("nv_cost", elastic_wf.objective))
    adj_C = elastic_wf.fleet_adjustment_cost or 0.0
    total_C = nv_C + adj_C

    # SAA-expected total cost at each fleet size — this is the objective the
    # elastic solver actually optimizes (decision-time view, based on historical
    # demand_samples). "Stay at F0" is always feasible for C, so saa_total_C_at_Fstar
    # <= saa_total_B_at_F0 by construction. Comparing this to the realized total_B
    # vs total_C (evaluated against actual test-set demand) shows the gap between
    # the SAA-optimal decision and its out-of-sample outcome.
    saa_total_B_at_F0 = float(zB["res_wf"].objective)
    saa_total_C_at_Fstar = float(elastic_wf.objective + adj_C)

    print(f"\n    Elastic fleet result (water-filling):")
    print(f"      F*            : {elastic_wf.fleet_optimal:.0f}  ({elastic_wf.status})")
    print(f"      Shadow price  : {elastic_wf.shadow_price:.4f}")
    print(f"      Break-even r  : {elastic_wf.breakeven_rental:.4f}")
    print(f"      NV cost (C)   : {nv_C:.0f}")
    print(f"      Adj cost      : {adj_C:.0f}")
    print(f"      Total cost (C): {total_C:.0f}")

    print(f"\n    A/B/C total cost comparison (all on realized test-set NV cost):")
    print(f"      Model A (flat,  fixed F0={fleet}): NV={nv_A:.0f}  adj=0  total={nv_A:.0f}")
    print(f"      Model B (borough,fixed F0={fleet}): NV={nv_B:.0f}  adj=0  total={nv_B:.0f}")
    print(f"      Model C (borough,elastic F*):    NV={nv_C:.0f}  adj={adj_C:.0f}  total={total_C:.0f}")

    dead_zone_note = (
        f"Dead zone [{standdown_saving:.0f}, {rental_rate:.0f}]: "
        f"if {standdown_saving:.0f} ≤ λ(F0) ≤ {rental_rate:.0f} the fleet stays at F0={fleet}"
    )
    print(f"\n    {dead_zone_note}")

    # SLSQP cross-check
    print(f"\n    Elastic cross-check (wf vs slsqp):")
    comp = opt_mod.compare_elastic_solvers(
        demand_samples, Cu, Co, floors, caps,
        F0=float(fleet), r=rental_rate, s=standdown_saving,
        zones=zones, fleet_min=fleet_min,
    )
    print(comp.to_string(index=False))

    return {
        "slot": slot_label,
        "elastic_wf": elastic_wf,
        "elastic_sl": elastic_sl,
        "nv_A": nv_A, "nv_B": nv_B, "nv_C": nv_C,
        "adj_C": adj_C,
        "total_A": nv_A, "total_B": nv_B, "total_C": total_C,
        "F_star": elastic_wf.fleet_optimal,
        "lam_A": task1_result["model_A"].get("shadow_price") or 0.0,
        "shadow_price": elastic_wf.shadow_price,
        "breakeven_rental": elastic_wf.breakeven_rental,   # = λ_B
        "direction": elastic_wf.status,
        "dead_zone_note": dead_zone_note,
        "saa_total_B_at_F0": saa_total_B_at_F0,
        "saa_total_C_at_Fstar": saa_total_C_at_Fstar,
        "cross_check": comp,
    }


# ── Save CSVs ─────────────────────────────────────────────────────────────────

def _merge_and_save_csv(new_df: pd.DataFrame, out_path: Path) -> None:
    """Replace rows for the slots in `new_df`, keep all other slots' rows from
    the existing CSV (so a custom single-slot rerun doesn't wipe other slots)."""
    if out_path.exists():
        old_df = pd.read_csv(out_path)
        new_slots = set(new_df["slot"].unique())
        old_df = old_df[~old_df["slot"].isin(new_slots)]
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(out_path, index=False)


def save_task1_csv(task1_results: List[dict]):
    rows = []
    for r in task1_results:
        slot = r["slot"]
        for model, key in [("A_flat", "model_A"), ("B_borough", "model_B")]:
            bor = r.get("borough_comparison", pd.DataFrame())
            if not bor.empty:
                for _, row in bor.iterrows():
                    rows.append({
                        "slot": slot,
                        "model": model,
                        "borough": row.get("borough", ""),
                        "total_q": row.get(f"q_{model[0]}", 0),
                    })
    if rows:
        new_df = pd.DataFrame(rows)
        out_path = OUT_DIR / "task1_AvsB.csv"
        _merge_and_save_csv(new_df, out_path)
        print(f"\nSaved: {out_path}")


def save_task2_csv(task2_results: List[dict]):
    rows = []
    for r in task2_results:
        rows.append({
            "slot": r["slot"],
            "F_star": r["F_star"],
            "direction": r["direction"],
            "shadow_price": r["shadow_price"],
            "breakeven_rental": r["breakeven_rental"],
            "nv_cost_A": r["nv_A"],
            "nv_cost_B": r["nv_B"],
            "nv_cost_C": r["nv_C"],
            "fleet_adj_cost_C": r["adj_C"],
            "total_cost_A": r["total_A"],
            "total_cost_B": r["total_B"],
            "total_cost_C": r["total_C"],
        })
    if rows:
        new_df = pd.DataFrame(rows)
        out_path = OUT_DIR / "task2_ABC.csv"
        _merge_and_save_csv(new_df, out_path)
        print(f"Saved: {out_path}")


# ── Markdown report ────────────────────────────────────────────────────────────

def _parse_existing_slot_sections(md_text: str, section_header: str) -> "OrderedDict[str, str]":
    """Return {slot_id: section_text} (including the '### Slot: x' line) for a
    '## <section_header>' block, preserving order."""
    sections: "OrderedDict[str, str]" = OrderedDict()
    pattern = rf"## {re.escape(section_header)}.*?(?=\n## |\Z)"
    m = re.search(pattern, md_text, re.DOTALL)
    if not m:
        return sections
    body = m.group(0)
    parts = re.split(r"(?=\n### Slot: )", body)
    for part in parts[1:]:
        slot_m = re.match(r"\n### Slot: (\S+)", part)
        if slot_m:
            sections[slot_m.group(1)] = part.lstrip("\n")
    return sections


def write_stakeholder_response(
    task1_results: List[dict],
    task2_results: List[dict],
    winner: str,
    rental_rate: float,
    standdown_saving: float,
    fleet: int,
    forced_model: bool = False,
):
    if forced_model:
        model_line = (f"Operational demand model **forced via --demand-model: {winner}** "
                       f"(bake-off training/evaluation skipped).\n")
        ranking_line = ""
    else:
        model_line = f"Operational demand model selected via bake-off (SAA / GLM / LGBM): **{winner}**.\n"
        ranking_line = "Full ranking saved to `outputs/bakeoff_model_selection.csv`.\n"

    lines = [
        "# Stakeholder Response — NYC Yellow Taxi Demand Optimization\n",
        "## Model Selection\n",
        model_line,
        ranking_line,
        "",
        "---\n",
        "## Task 1 — Zone-Specific Overage Costs (Model A vs Model B)\n",
        "Co_z = α_{b(z)} · Co_t.  Borough multipliers (higher outer-borough repositioning cost):",
        "| Borough | α |",
        "|---|---|",
        "| Manhattan | 1.00 |",
        "| Brooklyn | 1.25 |",
        "| Queens (non-airport) | 1.40 |",
        "| Bronx | 1.50 |",
        "| Staten Island | 1.80 |",
        "| JFK (zone 132) | 0.70 (airport override) |",
        "| LGA (zone 138) | 0.70 (airport override) |",
        "",
        "**Why outer boroughs have higher idle cost:** Repositioning an idle cab from Staten Island",
        "or the Bronx back to a high-demand corridor takes longer → higher opportunity cost.",
        "",
        "**Airport zones:** Lower idle cost because airports self-serve a queue; an extra cab there",
        "has low repositioning burden → lower overage penalty → Model B allocates more to JFK/LGA.\n",
    ]

    new_task1_sections: "OrderedDict[str, str]" = OrderedDict()
    for r in task1_results:
        slot = r["slot"]
        slot_lines = [f"### Slot: {slot}\n"]
        bor = r.get("borough_comparison", pd.DataFrame())
        if not bor.empty:
            slot_lines.append("**Borough q\\* totals (A flat vs B borough-specific):**\n")
            slot_lines.append(_df_to_md(bor[["borough", "q_A", "q_B", "delta_q"]]))
            slot_lines.append("")
        tau5 = r.get("tau_shift_top5", pd.DataFrame())
        if not tau5.empty:
            slot_lines.append("\n**Top-5 τ-shift zones (A→B):**\n")
            slot_lines.append(_df_to_md(tau5))
            slot_lines.append("")
        lam_A = r["model_A"].get("shadow_price") or 0.0
        lam_B = r["model_B"].get("shadow_price") or 0.0
        slot_lines.append(f"\n**Shadow price λ(F₀) at fixed fleet:**  "
                          f"Model A (flat Co) = {lam_A:.4f}   Model B (borough Co) = {lam_B:.4f}\n")
        slot_lines.append(f"*(λ = marginal NV cost reduction per extra cab; "
                          f"break-even rental in Task 2 equals Model B's λ)*\n")

        mA = r["model_A"]["test_metrics"]
        mB = r["model_B"]["test_metrics"]
        if mA and mB:
            delta = mB.get("nv_cost", 0) - mA.get("nv_cost", 0)
            slot_lines.append(f"\n**Realized NV cost on test:**  "
                              f"A={mA.get('nv_cost', 0):.0f}  B={mB.get('nv_cost', 0):.0f}  "
                              f"Δ={delta:+.0f}\n")
        cc = r.get("cross_check", pd.DataFrame())
        if not cc.empty:
            slot_lines.append("\n**3-way solver cross-check under Model B:**\n")
            slot_lines.append(_df_to_md(cc))
            slot_lines.append("")
        new_task1_sections[slot] = "\n".join(slot_lines)

    new_task2_sections: "OrderedDict[str, str]" = OrderedDict()
    for r in task2_results:
        slot = r["slot"]
        slot_lines = [f"### Slot: {slot}\n"]
        slot_lines.append(f"| Metric | Value |")
        slot_lines.append(f"|---|---|")
        slot_lines.append(f"| F* | {r['F_star']:.0f} ({r['direction']}) |")
        slot_lines.append(f"| λ_A (flat Co, fixed fleet) | {r.get('lam_A', float('nan')):.4f} |")
        slot_lines.append(f"| λ_B = break-even rental (borough Co, fixed fleet) | {r['breakeven_rental']:.4f} |")
        slot_lines.append(f"| Shadow price λ(F*) at optimal fleet | {r['shadow_price']:.4f} |")
        slot_lines.append(f"| Total cost A (flat,fixed) | {r['total_A']:.0f} |")
        slot_lines.append(f"| Total cost B (borough,fixed) | {r['total_B']:.0f} |")
        slot_lines.append(f"| Total cost C (borough,elastic) | {r['total_C']:.0f} |")
        slot_lines.append(f"| Fleet adj cost (C) | {r['adj_C']:.0f} |")
        slot_lines.append(f"| Budget binds at F*? | {'yes' if r['shadow_price'] > 1e-4 else 'no'} |")
        slot_lines.append("")
        slot_lines.append(f"\n**A→C cost decomposition:**")
        ab = r['total_B'] - r['total_A']
        bc = (r['total_C'] or 0) - r['total_B']
        slot_lines.append(f"- A→B (spatial reallocation): {ab:+.0f}")
        slot_lines.append(f"- B→C (fleet right-sizing): {bc:+.0f}\n")

        if r["direction"] != "optimal:stay":
            saa_b = r["saa_total_B_at_F0"]
            saa_c = r["saa_total_C_at_Fstar"]
            slot_lines.append(f"\n**Decision-time (SAA) vs. realized (test-set) cost:**\n")
            slot_lines.append(f"| Basis | Total cost @ F0 (Model B's choice) | Total cost @ F* (Model C's choice) |")
            slot_lines.append(f"|---|---|---|")
            slot_lines.append(f"| SAA-expected (decision time, historical samples) | {saa_b:.0f} | {saa_c:.0f} |")
            slot_lines.append(f"| Realized (this test set) | {r['total_B']:.0f} | {r['total_C']:.0f} |\n")
            saa_gain = saa_b - saa_c
            realized_gap = r['total_C'] - r['total_B']
            verdict = "underperformed" if realized_gap > 0 else "outperformed"
            slot_lines.append(
                f"Based on the historical (SAA) demand distribution, contracting/expanding to F*={r['F_star']:.0f} "
                f"looked **{saa_gain:+.0f} better** than staying at F0 — \"stay at F0\" is always a feasible "
                f"option for Model C, so its SAA-optimal total cost can never be worse than Model B's. "
                f"However, on this specific test set, Model C **{verdict}** Model B by {abs(realized_gap):.0f} "
                f"because realized demand differed from the historical samples the decision was based on. "
                f"This is the expected decision-time-vs-realized-outcome gap of any SAA-based policy, not an error "
                f"in the optimization.\n"
            )
        new_task2_sections[slot] = "\n".join(slot_lines)

    # Merge with existing per-slot sections so a rerun for a subset of slots
    # (e.g. one custom slot/week) doesn't wipe out other slots' results.
    md_path = OUT_DIR / "STAKEHOLDER_RESPONSE.md"
    old_md = md_path.read_text() if md_path.exists() else ""
    merged_task1 = _parse_existing_slot_sections(old_md, "Task 1 — Zone-Specific Overage Costs (Model A vs Model B)")
    merged_task1.update(new_task1_sections)
    merged_task2 = _parse_existing_slot_sections(old_md, "Task 2 — Elastic Fleet (Model C)")
    merged_task2.update(new_task2_sections)

    full_lines = list(lines)
    full_lines.extend(merged_task1.values())
    full_lines.append("")
    full_lines.append("---\n")
    full_lines.append("## Task 2 — Elastic Fleet (Model C)\n")
    full_lines.append(
        f"Fleet F is a continuous decision variable jointly optimized with q.  "
        f"Adjustment cost: r·(F−F₀)⁺ − s·(F₀−F)⁺ with r={rental_rate:.0f}, s={standdown_saving:.0f}, F₀={fleet}.\n"
    )
    full_lines.append(
        f"**Dead zone [{standdown_saving:.0f}, {rental_rate:.0f}]:** "
        f"when s ≤ λ(F₀) ≤ r the fleet stays at F₀ (marginal NV saving doesn't justify rental).\n"
    )
    full_lines.append("**Shadow price interpretation:** λ(F) = marginal NV cost reduction per extra cab.")
    full_lines.append("TLC should expand if λ(F₀) > r, contract if λ(F₀) < s.\n")
    full_lines.append("**Break-even rental rate** = λ(F₀): the rental rate at which TLC is indifferent about expansion.\n")
    full_lines.append(
        "**Total cost basis:** all of A/B/C below are the *realized* newsvendor cost on the held-out "
        "test set (same basis as Task 1's \"Realized NV cost on test\"), plus the fleet adjustment "
        "cost for C — so A/B/C are directly comparable.\n"
    )
    full_lines.extend(merged_task2.values())

    md_path.write_text("\n".join(full_lines))
    print(f"\nSaved: {md_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    slots = parse_slots(args.slots)
    t_total = time.perf_counter()

    df, df_full, train, val, test = load_data(args.fleet)

    # Build global borough map from prepared data
    from engine.data import _build_borough_map
    borough_map = _build_borough_map(train["PULocationID"].unique())

    winner, models = run_bakeoff(
        train, val, test,
        fleet=args.fleet,
        force_model=args.demand_model,
    )
    forced_model = bool(args.demand_model and args.demand_model != "auto")

    task1_results = []
    task2_results = []

    for dow, time_bucket in slots:
        print(f"\n{'='*65}")
        print(f"SLOT: dow={dow}  bucket={time_bucket}")
        print('='*65)

        if args.task in ("1", "both"):
            t1 = run_task1(
                train, test, models, winner,
                dow=dow, time_bucket=time_bucket, fleet=args.fleet,
                borough_map=borough_map,
                rental_rate=args.rental_rate,
                standdown_saving=args.standdown_saving,
                week=args.week, year=args.year,
            )
            task1_results.append(t1)

            if args.task in ("2", "both"):
                t2 = run_task2(
                    train, test, t1,
                    fleet=args.fleet,
                    rental_rate=args.rental_rate,
                    standdown_saving=args.standdown_saving,
                    dow=dow, time_bucket=time_bucket,
                )
                task2_results.append(t2)

    save_task1_csv(task1_results)
    save_task2_csv(task2_results)
    write_stakeholder_response(
        task1_results, task2_results,
        winner=winner,
        rental_rate=args.rental_rate,
        standdown_saving=args.standdown_saving,
        fleet=args.fleet,
        forced_model=forced_model,
    )

    print(f"\nTotal runtime: {time.perf_counter()-t_total:.1f}s")


if __name__ == "__main__":
    main()
