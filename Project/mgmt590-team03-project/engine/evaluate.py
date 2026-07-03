"""Metrics, bake-off comparison, and backtest utilities.

Primary metric: out-of-sample realized newsvendor cost
    Σ_z [ Cu_z * max(D_z - q_z, 0) + Co_z * max(q_z - D_z, 0) ]

Secondary: pinball loss at τ, fill-rate, fleet utilization, SPO regret vs oracle.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from engine.numeric import safe_int


# ── Per-scenario metrics ─────────────────────────────────────────────────────

def realized_nv_cost(
    q: np.ndarray,
    demand_realized: np.ndarray,   # true demand per zone
    Cu: np.ndarray,
    Co: np.ndarray,
) -> float:
    """Total realized newsvendor cost (the primary eval metric)."""
    understock = np.maximum(demand_realized - q, 0)
    overstock = np.maximum(q - demand_realized, 0)
    return float((Cu * understock + Co * overstock).sum())


def pinball_loss(q: np.ndarray, demand_realized: np.ndarray, tau: np.ndarray) -> float:
    """Mean pinball (quantile) loss at τ."""
    residual = demand_realized - q
    loss = np.where(residual >= 0, tau * residual, (tau - 1) * residual)
    return float(loss.mean())


def fill_rate(q: np.ndarray, demand_realized: np.ndarray) -> float:
    """Fraction of demand that is served: Σmin(q,D) / ΣD."""
    served = np.minimum(q, demand_realized).sum()
    total_demand = demand_realized.sum()
    return float(served / total_demand) if total_demand > 0 else 1.0


def spo_regret(
    q: np.ndarray,
    demand_realized: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
) -> float:
    """SPO regret vs oracle: NV(q, D) - NV(q_oracle, D).
    The oracle would have allocated q_oracle = D (clamp to actual demand).
    """
    oracle = demand_realized.copy()
    nv_q = realized_nv_cost(q, demand_realized, Cu, Co)
    nv_oracle = realized_nv_cost(oracle, demand_realized, Cu, Co)
    return float(nv_q - nv_oracle)


def fleet_utilization(q: np.ndarray, fleet_size: int) -> float:
    return float(q.sum() / fleet_size)


def revenue_generated(q: np.ndarray, demand_realized: np.ndarray, Cu: np.ndarray) -> float:
    """Revenue from served trips: Σ Cu_z * min(q_z, D_z)."""
    return float((Cu * np.minimum(q, demand_realized)).sum())


def revenue_foregone(q: np.ndarray, demand_realized: np.ndarray, Cu: np.ndarray) -> float:
    """Revenue lost from unserved trips: Σ Cu_z * (D_z - q_z)+."""
    return float((Cu * np.maximum(demand_realized - q, 0)).sum())


def compute_all_metrics(
    q: np.ndarray,
    demand_realized: np.ndarray,
    Cu: np.ndarray,
    Co: np.ndarray,
    tau: np.ndarray,
    fleet_size: int,
) -> dict:
    return {
        "nv_cost": realized_nv_cost(q, demand_realized, Cu, Co),
        "pinball_loss": pinball_loss(q, demand_realized, tau),
        "fill_rate": fill_rate(q, demand_realized),
        "spo_regret": spo_regret(q, demand_realized, Cu, Co),
        "fleet_utilization": fleet_utilization(q, fleet_size),
        "total_understock": float(np.maximum(demand_realized - q, 0).sum()),
        "total_overstock": float(np.maximum(q - demand_realized, 0).sum()),
        "fleet_used": safe_int(q.sum()),
    }


# ── Bake-off: compare multiple allocations on the same test scenarios ────────

def bakeoff_summary(
    results: Dict[str, List[dict]],
) -> pd.DataFrame:
    """Aggregate a dict of {model_name: [metrics_dict per scenario]} into a summary table."""
    rows = []
    for model_name, metrics_list in results.items():
        if not metrics_list:
            continue
        df_m = pd.DataFrame(metrics_list)
        rows.append(
            {
                "model": model_name,
                "mean_nv_cost": df_m["nv_cost"].mean(),
                "total_nv_cost": df_m["nv_cost"].sum(),
                "mean_fill_rate": df_m["fill_rate"].mean(),
                "mean_pinball": df_m["pinball_loss"].mean(),
                "mean_spo_regret": df_m["spo_regret"].mean(),
                "mean_fleet_util": df_m["fleet_utilization"].mean(),
                "n_scenarios": len(df_m),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "mean_nv_cost",
                "total_nv_cost",
                "mean_fill_rate",
                "mean_pinball",
                "mean_spo_regret",
                "mean_fleet_util",
                "n_scenarios",
                "rank",
            ]
        )
    summary = pd.DataFrame(rows).sort_values("mean_nv_cost")
    summary["rank"] = range(1, len(summary) + 1)
    return summary.reset_index(drop=True)


# ── Proxy sensitivity summary ─────────────────────────────────────────────────

def fleet_adjustment_cost(F: float, F0: float, r: float, s: float) -> float:
    """cost(F) = r*(F-F0)+ - s*(F0-F)+"""
    return r * max(F - F0, 0.0) - s * max(F0 - F, 0.0)


def borough_summary(
    q: np.ndarray,
    zones: np.ndarray,
    borough_map: dict,
) -> pd.DataFrame:
    """Aggregate q* totals per borough.

    borough_map: {zone_id -> borough_id}
    Returns a DataFrame with columns: borough_id, total_q, n_zones.
    """
    rows = []
    for zone_id, q_val in zip(zones, q):
        rows.append({"PULocationID": int(zone_id), "q": float(q_val),
                     "borough_id": borough_map.get(int(zone_id), 5)})
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("borough_id")
        .agg(total_q=("q", "sum"), n_zones=("q", "count"))
        .reset_index()
        .sort_values("borough_id")
    )
    return summary


def proxy_sensitivity_summary(
    q_results: Dict[str, np.ndarray],   # {aggregation_level: q*}
    zones: np.ndarray,
) -> pd.DataFrame:
    """Compare q* vectors across proxy aggregation levels."""
    ref_key = "borough" if "borough" in q_results else next(iter(q_results))
    ref = q_results[ref_key]
    rows = []
    for name, q in q_results.items():
        diff = q - ref
        rows.append(
            {
                "proxy": name,
                "total_allocated": safe_int(q.sum()),
                "max_zone_diff": float(np.abs(diff).max()),
                "mean_abs_zone_diff": float(np.abs(diff).mean()),
                "n_zones_changed": int((np.abs(diff) > 0.5).sum()),
            }
        )
    return pd.DataFrame(rows)
