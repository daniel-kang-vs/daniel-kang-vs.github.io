"""Compute data-derived per-zone caps and floors for the allocation constraints.

cap_z   = ceil(cap_multiplier * D_p95(z, bucket, dow))
floor_z = ceil(floor_alpha   * D_median(z, bucket, dow))

Both are exposed as agent-tunable parameters (via OptimizationConfig).
Feasibility invariants checked before solving:
  sum(floors) <= fleet_size   (headroom for optimization)
  floor_z <= cap_z            (box validity)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from engine.numeric import safe_int


def compute_bounds(
    df: pd.DataFrame,
    time_bucket: str,
    day_of_week: int,
    floor_alpha: float = 0.15,
    cap_multiplier: float = 1.5,
    floor_overrides: dict[int, int] | None = None,
    cap_overrides: dict[int, int] | None = None,
    fleet_size: int = 13000,
) -> pd.DataFrame:
    """Return a DataFrame with columns (PULocationID, floor, cap) for the given scenario.

    Pools all months for the (bucket, dow) combination to get robust quantile estimates.
    """
    # Filter to the relevant bucket and dow
    mask = (df["time_bucket"] == time_bucket) & (df["day_of_week"] == day_of_week)
    sub = df[mask].copy()

    if sub.empty:
        raise ValueError(f"No data for bucket={time_bucket!r}, dow={day_of_week}")

    # Demand column must exist (added by proxy.add_demand)
    if "demand" not in sub.columns:
        raise ValueError("Run proxy.add_demand() before compute_bounds()")

    stats = (
        sub.groupby("PULocationID")["demand"]
        .agg(
            d_median="median",
            d_p95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    stats["d_median"] = stats["d_median"].fillna(0)
    stats["d_p95"] = stats["d_p95"].fillna(0)

    stats["floor"] = np.ceil(floor_alpha * stats["d_median"]).astype(int).clip(lower=0)
    stats["cap"] = np.ceil(cap_multiplier * stats["d_p95"]).astype(int).clip(lower=1)

    # Ensure floor <= cap
    stats["floor"] = stats[["floor", "cap"]].min(axis=1)

    # Apply overrides
    if floor_overrides:
        for zone_id, val in floor_overrides.items():
            stats.loc[stats["PULocationID"] == zone_id, "floor"] = val
    if cap_overrides:
        for zone_id, val in cap_overrides.items():
            stats.loc[stats["PULocationID"] == zone_id, "cap"] = val

    # Re-enforce floor <= cap after overrides
    stats["floor"] = stats[["floor", "cap"]].min(axis=1)

    # If total floors exceed fleet_size, scale proportionally so the problem stays feasible
    total_floor = stats["floor"].sum()
    if total_floor > fleet_size:
        scale = (fleet_size * 0.25) / total_floor   # keep floors to ≤25% of fleet
        stats["floor"] = np.floor(stats["floor"] * scale).astype(int).clip(lower=0)
        stats["floor"] = stats[["floor", "cap"]].min(axis=1)

    return stats[["PULocationID", "floor", "cap"]]


def check_feasibility(bounds: pd.DataFrame, fleet_size: int) -> dict:
    """Check that the bounds are feasible given the fleet.

    Returns a dict with 'feasible' bool and diagnostic info.
    """
    total_floor = bounds["floor"].sum()
    total_cap = bounds["cap"].sum()
    headroom = fleet_size - total_floor
    floor_pct = total_floor / fleet_size * 100

    issues = []
    if total_floor > fleet_size:
        issues.append(
            f"sum(floors)={total_floor} > fleet_size={fleet_size}: INFEASIBLE. "
            "Reduce floor_alpha or floor_overrides."
        )
    if (bounds["floor"] > bounds["cap"]).any():
        bad = bounds[bounds["floor"] > bounds["cap"]]["PULocationID"].tolist()
        issues.append(f"floor > cap for zones: {bad}")

    if issues:
        for msg in issues:
            warnings.warn(msg)

    return {
        "feasible": len(issues) == 0,
        "total_floor": safe_int(total_floor),
        "total_cap": safe_int(total_cap),
        "headroom": safe_int(headroom),
        "floor_pct_of_fleet": round(float(floor_pct), 1) if np.isfinite(floor_pct) else 0.0,
        "issues": issues,
    }
