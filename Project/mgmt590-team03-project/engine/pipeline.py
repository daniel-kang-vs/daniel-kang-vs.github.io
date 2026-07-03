"""End-to-end Phase-1 optimization pipeline.

Usage:
    from engine.pipeline import run_pipeline
    from engine.config import OptimizationConfig

    config = OptimizationConfig(time_bucket="09:00-15:59", day_of_week=2, week=50)
    result = run_pipeline(config)
    print(result["q_star"])          # 263-vector
    print(result["bakeoff"])         # model comparison table
"""

from __future__ import annotations

import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from engine.config import OptimizationConfig
from engine import data as data_mod
from engine import proxy as proxy_mod
from engine import costs as costs_mod
from engine import bounds as bounds_mod
from engine import optimize as opt_mod
from engine.evaluate import compute_all_metrics, bakeoff_summary, proxy_sensitivity_summary
from engine.numeric import sanitize_allocation
from engine.models.empirical import EmpiricalAllocator
from engine.models.linear_demand import LogLinearDemandModel
from engine.models.qr_lgbm import LGBMQuantileModel
from engine.models.dfl_spo import LinearDFLModel


def _actual_demand_for_scenario(
    df: pd.DataFrame,
    config: OptimizationConfig,
    zones: np.ndarray,
    saa_fallback: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Observed `demand` for the exact (year?, week, dow, time_bucket), per zone.

    Zones with no matching row fall back to their SAA sample mean
    (`saa_fallback`). When `config.year` is None, averages across whichever
    years have data for that (week, dow, time_bucket). If no rows match at
    all, returns `saa_fallback` for every zone.
    """
    mask = (
        (df["day_of_week"] == config.day_of_week)
        & (df["time_bucket"] == config.time_bucket)
        & (df["week"] == config.week)
    )
    if config.year is not None:
        mask &= (df["year"] == config.year)
    sub = df[mask]
    if sub.empty:
        return saa_fallback, "saa_mean"

    actual = sub.groupby("PULocationID")["demand"].mean().reindex(zones)
    n_missing = int(actual.isna().sum())
    if n_missing == 0:
        return actual.values.astype(float), "actual"
    filled = actual.fillna(pd.Series(saa_fallback, index=zones))
    return filled.values.astype(float), f"actual+saa_mean({n_missing} zones)"


def run_pipeline(
    config: OptimizationConfig,
    df_full: Optional[pd.DataFrame] = None,
    linear_model: Optional[LogLinearDemandModel] = None,
    lgbm_model: Optional[LGBMQuantileModel] = None,
    dfl_model: Optional[LinearDFLModel] = None,
    empirical: Optional[EmpiricalAllocator] = None,
    verbose: bool = True,
    skip_solver_check: bool = False,
    skip_proxy_sensitivity: bool = False,
    include_dfl: bool = True,
    empirical_only: bool = False,
) -> dict:
    """Full Phase-1 pipeline for a given OptimizationConfig.

    If pre-trained models are supplied (lgbm_model, dfl_model) they are used directly
    (fast re-solve path).  Otherwise models are trained from scratch (refit path).

    Returns a dict with:
        q_star          : 263-vector (integer) from the configured solver
        zones           : PULocationID array (aligned to q_star)
        bakeoff         : DataFrame comparing empirical / QR-PtO / DFL
        solver_check    : DataFrame comparing SLSQP / water_filling / cvxpy on one scenario
        proxy_sensitivity: DataFrame comparing proxy aggregation levels
        metrics         : dict of realized metrics for the primary solver
        runtime_s       : total pipeline runtime in seconds
        config          : the config used
    """
    t_start = time.perf_counter()

    # ── 1. Load & prepare data ────────────────────────────────────────────────
    if df_full is None:
        if verbose: print("[1/7] Loading and preparing data …")
        df_full = data_mod.prepare(n_clusters=config.n_clusters, seed=config.random_seed)

    # ── 2. Add demand proxy ───────────────────────────────────────────────────
    if verbose: print(f"[2/7] Building demand proxy (aggregation={config.proxy_aggregation}) …")
    df = proxy_mod.add_demand(
        df_full,
        aggregation=config.proxy_aggregation,
        demand_multiplier=config.demand_multiplier,
    )

    # ── 3. Temporal split ─────────────────────────────────────────────────────
    train, val, test = data_mod.temporal_split(df)

    # ── 4. Cost parameters ────────────────────────────────────────────────────
    if verbose: print("[3/7] Computing Cu/Co/tau …")
    costs_df = costs_mod.compute_costs(
        train,
        cu_multiplier=config.cu_multiplier,
        co_multiplier=config.co_multiplier,
        co_mode=config.co_mode,
        co_borough_multipliers=config.co_borough_multipliers,
        co_zone_overrides=config.co_zone_overrides,
    )

    scenario_costs = costs_mod.get_costs_for_scenario(costs_df, config.time_bucket)
    zones = np.array(sorted(scenario_costs.index.tolist()))
    Cu = scenario_costs.reindex(zones)["Cu"].values
    Co = scenario_costs.reindex(zones)["Co"].values
    tau = scenario_costs.reindex(zones)["tau"].values

    # ── 5. Caps & floors ──────────────────────────────────────────────────────
    if verbose: print("[4/7] Computing bounds (caps & floors) …")
    bounds_df = bounds_mod.compute_bounds(
        train,
        time_bucket=config.time_bucket,
        day_of_week=config.day_of_week,
        floor_alpha=config.floor_alpha,
        cap_multiplier=config.cap_multiplier,
        floor_overrides=config.floor_overrides,
        cap_overrides=config.cap_overrides,
        fleet_size=config.fleet_size,
    ).set_index("PULocationID").reindex(zones).fillna(0)

    floors = bounds_df["floor"].values.astype(float)
    caps = bounds_df["cap"].values.astype(float)
    caps = np.maximum(caps, floors + 1)   # ensure cap > floor always

    feasibility = bounds_mod.check_feasibility(
        bounds_df.reset_index(), config.fleet_size
    )
    if not feasibility["feasible"]:
        warnings.warn(f"Bounds infeasible: {feasibility['issues']}")

    # ── 6. Empirical allocator ─────────────────────────────────────────────────
    if empirical is None:
        if verbose:
            print("[5/7] Fitting empirical allocator …")
        empirical = EmpiricalAllocator().fit(train)
    elif verbose:
        print("[5/7] Using cached empirical allocator …")
    emp_samples = empirical.get_samples(zones, config.time_bucket, config.day_of_week)

    # ── 6a. Solver cross-check (empirical demand samples) ────────────────────
    if skip_solver_check or empirical_only:
        solver_check = pd.DataFrame()
    elif not skip_solver_check:
        if verbose:
            print("      Cross-checking SLSQP / water_filling / cvxpy …")
        solver_check = opt_mod.compare_solvers(
            emp_samples, Cu, Co, floors, caps, config.fleet_size, zones
        )
        if verbose:
            print(solver_check.to_string(index=False))

    res_emp = opt_mod.solve(
        emp_samples, Cu, Co, floors, caps, config.fleet_size, zones,
        method="water_filling" if empirical_only else config.solver,
    )

    if empirical_only:
        if verbose:
            print("[6/7] Agent path — SAA (empirical) only; skipping GLM/LGBM/DFL …")

        primary_res = res_emp
        q_primary = sanitize_allocation(primary_res.q_star_int.astype(float))
        elastic_result = None

        if config.elastic_fleet:
            if verbose:
                print("[+] Elastic fleet optimization (SAA demand) …")
            fleet_min = config.fleet_min if config.fleet_min is not None else int(floors.sum())
            elastic_result = opt_mod.solve_elastic_fleet(
                emp_samples, Cu, Co, floors, caps,
                F0=float(config.fleet_baseline),
                r=config.rental_rate,
                s=config.standdown_saving,
                zones=zones,
                fleet_min=float(fleet_min),
            )
            primary_res = elastic_result
            q_primary = sanitize_allocation(elastic_result.q_star_int.astype(float))

        t_total = time.perf_counter() - t_start
        if verbose:
            print(f"\n{'='*60}")
            print(f"Scenario: bucket={config.time_bucket}, dow={config.day_of_week}, week={config.week}")
            print(f"Co mode: {config.co_mode} | Elastic fleet: {config.elastic_fleet}")
            print(f"Fleet used: {primary_res.fleet_used} / {config.fleet_size}")
            print(f"Primary model: empirical (SAA)")
            print(f"Total pipeline runtime: {t_total:.1f}s")

        saa_mean = sanitize_allocation(emp_samples.mean(axis=0))
        actual_demand, demand_source = _actual_demand_for_scenario(df, config, zones, saa_mean)
        scenario_metrics = compute_all_metrics(
            q_primary,
            sanitize_allocation(actual_demand),
            Cu, Co, tau, config.fleet_size,
        )
        scenario_metrics["demand_source"] = demand_source
        bakeoff_df = bakeoff_summary({"empirical": [scenario_metrics]})
        if not bakeoff_df.empty:
            bakeoff_df = bakeoff_df[bakeoff_df["model"] == "empirical"].reset_index(drop=True)

        out = {
            "q_star": q_primary,
            "q_star_continuous": sanitize_allocation(primary_res.q_star),
            "zones": zones,
            "Cu": Cu,
            "Co": Co,
            "tau": tau,
            "floors": floors,
            "caps": caps,
            "primary_model": "empirical",
            "bakeoff": bakeoff_df,
            "solver_check": solver_check,
            "proxy_sensitivity": pd.DataFrame(),
            "metrics": scenario_metrics,
            "feasibility": feasibility,
            "runtime_s": round(t_total, 2),
            "config": config,
            "models": {"linear": None, "lgbm": None, "dfl": None, "empirical": empirical},
            "solve_results": {"empirical": res_emp},
            "emp_samples": emp_samples,
            "train": train,
            "val": val,
            "test": test,
        }
        if elastic_result is not None:
            direction = str(elastic_result.status).replace("optimal:", "")
            out["elastic"] = {
                "F0": float(config.fleet_baseline),
                "F_star": elastic_result.fleet_optimal,
                "direction": direction,
                "shadow_price": elastic_result.shadow_price,
                "breakeven_rental": elastic_result.breakeven_rental,
                "nv_cost": float(elastic_result.objective),
                "fleet_adjustment_cost": elastic_result.fleet_adjustment_cost,
                "total_cost": elastic_result.total_cost,
                "status": elastic_result.status,
                "rental_rate": config.rental_rate,
                "standdown_saving": config.standdown_saving,
            }
        return out

    # ── 7. Track A: Log-linear (Poisson GLM) ─────────────────────────────────
    if verbose: print("[6/7] Track A — Log-linear Poisson GLM …")
    if linear_model is None:
        linear_model = LogLinearDemandModel(seed=config.random_seed)
        linear_model.fit(train)
        if verbose:
            prep = linear_model._pipeline.named_steps["prep"]
            model = linear_model._pipeline.named_steps["model"]
            for name, coef in zip(prep.get_feature_names_out(), model.coef_):
                print(f"{name:30s}  {coef:+.4f}")
            print(f"{'intercept':30s}  {model.intercept_:+.4f}")
    elif verbose:
        print("      Using cached GLM …")

    qr_zones, qr_pred_raw = linear_model.predict_for_scenario(
        df,
        config.time_bucket,
        config.day_of_week,
        config.week,
        tau=tau,
        zones=zones,
        weather_override=config.weather_override,
    )
    # Align QR predictions to the canonical zones array (fill missing zones with emp mean)
    qr_pred_series = pd.Series(qr_pred_raw, index=qr_zones)
    emp_mean = emp_samples.mean(axis=0)
    qr_mean = np.array([qr_pred_series.get(z, emp_mean[i]) for i, z in enumerate(zones)])
    qr_mean = np.maximum(qr_mean, 0)
    # Blend: QR mean + empirical dispersion for a richer sample set
    emp_deviation = emp_samples - emp_mean
    qr_samples_blend = np.maximum(qr_mean + emp_deviation, 0)   # (n_emp, n_zones)

    res_qr = opt_mod.solve(
        qr_samples_blend, Cu, Co, floors, caps, config.fleet_size, zones,
        method=config.solver
    )

    # ── 7b. Track A2: LightGBM Quantile (per-sample τ pinball) ───────────────
    if verbose: print("[6/7] Track A2 — LightGBM per-sample-τ quantile regression …")
    if lgbm_model is None:
        lgbm_model = LGBMQuantileModel(seed=config.random_seed)
        lgbm_model.fit(train, None, costs_df)
    elif verbose:
        print("      Using cached LightGBM …")

    lgbm_zones, lgbm_pred_raw = lgbm_model.predict_for_scenario(
        df, config.time_bucket, config.day_of_week, config.week,
        weather_override=config.weather_override,
    )
    lgbm_pred_series = pd.Series(lgbm_pred_raw, index=lgbm_zones)
    lgbm_mean = np.array([lgbm_pred_series.get(z, emp_mean[i]) for i, z in enumerate(zones)])
    lgbm_mean = np.maximum(lgbm_mean, 0)
    lgbm_samples_blend = np.maximum(lgbm_mean + emp_deviation, 0)

    res_lgbm = opt_mod.solve(
        lgbm_samples_blend, Cu, Co, floors, caps, config.fleet_size, zones,
        method=config.solver
    )

    # ── 8. Track B: DFL (SPO+, L-BFGS-B) — optional ─────────────────────────
    if include_dfl:
        if verbose: print("[6/7] Track B — DFL/SPO+ with L-BFGS-B …")
        if dfl_model is None:
            dfl_model = LinearDFLModel(seed=config.random_seed)
            dfl_model.fit(
                train, Cu, Co, floors, caps, config.fleet_size, zones,
                time_bucket=config.time_bucket,
                day_of_week=config.day_of_week,
            )

        elif verbose:
            print("      Using cached DFL …")

        dfl_zones_arr, dfl_pred_raw = dfl_model.predict(
            df,
            config.time_bucket,
            config.day_of_week,
            config.week,
            weather_override=config.weather_override,
        )
        dfl_pred_series = pd.Series(dfl_pred_raw, index=dfl_zones_arr)
        dfl_mean = np.array([dfl_pred_series.get(z, emp_mean[i]) for i, z in enumerate(zones)])
        dfl_mean = np.maximum(dfl_mean, 0)
        dfl_samples_blend = np.maximum(dfl_mean + emp_deviation, 0)
        res_dfl = opt_mod.solve(
            dfl_samples_blend, Cu, Co, floors, caps, config.fleet_size, zones,
            method=config.solver
        )
    else:
        res_dfl = None
        dfl_model = None

    # ── 9. Evaluate on test set ───────────────────────────────────────────────
    if verbose: print("[7/7] Evaluating on test set …")
    test_mask = (
        (test["time_bucket"] == config.time_bucket)
        & (test["day_of_week"] == config.day_of_week)
        & (test["week"] == config.week)
    )
    test_sub = test[test_mask]

    if test_sub.empty:
        # Fall back to all matching test weeks for this (bucket, dow) if the exact
        # week isn't in the test split (e.g. scenario week is in train/val).
        test_mask_any_week = (
            (test["time_bucket"] == config.time_bucket)
            & (test["day_of_week"] == config.day_of_week)
        )
        test_sub = test[test_mask_any_week]

    if test_sub.empty:
        warnings.warn("No test data for this scenario; using validation set.")
        test_mask2 = (
            (val["time_bucket"] == config.time_bucket)
            & (val["day_of_week"] == config.day_of_week)
            & (val["week"] == config.week)
        )
        test_sub = val[test_mask2]
        if test_sub.empty:
            test_mask2_any = (
                (val["time_bucket"] == config.time_bucket)
                & (val["day_of_week"] == config.day_of_week)
            )
            test_sub = val[test_mask2_any]

    candidates = [("empirical", res_emp), ("qr_pto", res_qr), ("lgbm_pto", res_lgbm)]
    if include_dfl and res_dfl is not None:
        candidates.append(("dfl", res_dfl))

    bakeoff_results = {name: [] for name, _ in candidates}

    for week_test in sorted(test_sub["week"].unique()):
        sub_w = test_sub[test_sub["week"] == week_test]
        demand_true = (
            sub_w.groupby("PULocationID")["demand"].mean()
            .reindex(zones).fillna(0).values.astype(float)
        )

        for model_name, res in candidates:
            m = compute_all_metrics(
                res.q_star_int.astype(float),
                demand_true,
                Cu, Co, tau,
                config.fleet_size,
            )
            m["week"] = week_test
            bakeoff_results[model_name].append(m)

    bakeoff_df = bakeoff_summary(bakeoff_results)

    # ── 10. Proxy sensitivity ─────────────────────────────────────────────────
    if skip_proxy_sensitivity:
        proxy_sensitivity = pd.DataFrame()
    else:
        proxy_q = {}
        for agg in ("global", "borough", "cluster"):
            try:
                df_agg = proxy_mod.add_demand(df_full, aggregation=agg)
                emp_agg = EmpiricalAllocator().fit(data_mod.temporal_split(df_agg)[0])
                samp_agg = emp_agg.get_samples(zones, config.time_bucket, config.day_of_week)
                r_agg = opt_mod.solve(
                    samp_agg, Cu, Co, floors, caps, config.fleet_size, zones,
                    method="water_filling"
                )
                proxy_q[agg] = r_agg.q_star
            except Exception as e:
                warnings.warn(f"Proxy sensitivity for {agg} failed: {e}")

        proxy_sensitivity = proxy_sensitivity_summary(proxy_q, zones) if proxy_q else pd.DataFrame()

    # ── Select primary result ─────────────────────────────────────────────────
    primary_model = bakeoff_df.iloc[0]["model"] if not bakeoff_df.empty else "empirical"
    result_map = {"empirical": res_emp, "qr_pto": res_qr, "lgbm_pto": res_lgbm}
    if include_dfl and res_dfl is not None:
        result_map["dfl"] = res_dfl
    primary_res = result_map.get(primary_model, res_emp)
    q_primary = sanitize_allocation(primary_res.q_star_int.astype(float))

    # ── Elastic fleet (Task 2) ────────────────────────────────────────────────
    elastic_result = None
    if config.elastic_fleet:
        if verbose: print("[+] Elastic fleet optimization …")
        fleet_min = config.fleet_min if config.fleet_min is not None else int(floors.sum())
        elastic_result = opt_mod.solve_elastic_fleet(
            emp_samples, Cu, Co, floors, caps,
            F0=float(config.fleet_baseline),
            r=config.rental_rate,
            s=config.standdown_saving,
            zones=zones,
            fleet_min=float(fleet_min),
        )
        primary_res = elastic_result

    t_total = time.perf_counter() - t_start

    if verbose:
        print(f"\n{'='*60}")
        print(f"Scenario: bucket={config.time_bucket}, dow={config.day_of_week}, week={config.week}")
        print(f"Fleet used: {primary_res.fleet_used} / {config.fleet_size}")
        print(f"Feasibility: {feasibility}")
        print(f"\nBake-off results:")
        print(bakeoff_df.to_string(index=False))
        if not proxy_sensitivity.empty:
            print(f"\nProxy sensitivity:")
            print(proxy_sensitivity.to_string(index=False))
        print(f"\nTotal pipeline runtime: {t_total:.1f}s")

    out = {
        "q_star": q_primary,
        "q_star_continuous": sanitize_allocation(primary_res.q_star),
        "zones": zones,
        "Cu": Cu,
        "Co": Co,
        "tau": tau,
        "floors": floors,
        "caps": caps,
        "primary_model": primary_model,
        "bakeoff": bakeoff_df,
        "solver_check": solver_check,
        "proxy_sensitivity": proxy_sensitivity,
        "metrics": compute_all_metrics(
            q_primary,
            sanitize_allocation(emp_samples.mean(axis=0)),
            Cu, Co, tau, config.fleet_size,
        ),
        "feasibility": feasibility,
        "runtime_s": round(t_total, 2),
        "config": config,
        "models": {"linear": linear_model, "lgbm": lgbm_model, "dfl": dfl_model, "empirical": empirical},
        "solve_results": {"empirical": res_emp, "qr_pto": res_qr, "lgbm_pto": res_lgbm},
        "emp_samples": emp_samples,
        "train": train,
        "val": val,
        "test": test,
    }
    if elastic_result is not None:
        direction = str(elastic_result.status).replace("optimal:", "")
        out["elastic"] = {
            "F0": float(config.fleet_baseline),
            "F_star": elastic_result.fleet_optimal,
            "direction": direction,
            "shadow_price": elastic_result.shadow_price,
            "breakeven_rental": elastic_result.breakeven_rental,
            "nv_cost": float(elastic_result.objective),
            "fleet_adjustment_cost": elastic_result.fleet_adjustment_cost,
            "total_cost": elastic_result.total_cost,
            "status": elastic_result.status,
            "rental_rate": config.rental_rate,
            "standdown_saving": config.standdown_saving,
        }
    return out
