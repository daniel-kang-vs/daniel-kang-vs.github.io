"""Demand realization: convert observed trips to latent demand using market-share proxy.

Formula (per borough/cluster/global):
    mul_factor_b = avg_yellow_b / (avg_yellow_b + avg_fhv_b)
    demand_{z,t,d,m} = mul_factor_{b(z)} * (yellow_pickups + fhv_pickups)

Rationale: a cell where yellow is locally over/under-represented relative to its group
creates genuine over/under-supply signal for the newsvendor.  Computing at borough (not
global) granularity removes the structural Manhattan vs outer-borough bias.
"""

from __future__ import annotations

import pandas as pd


def _mul_factors_global(df: pd.DataFrame) -> pd.Series:
    """Single global scalar: avg_yellow / (avg_yellow + avg_fhv)."""
    avg_y = df["yellow_pickups"].mean()
    avg_f = df["fhv_pickups"].mean()
    denom = avg_y + avg_f
    if denom == 0:
        return pd.Series(0.5, index=df.index)
    factor = avg_y / denom
    return pd.Series(factor, index=df.index)


def _mul_factors_by_group(df: pd.DataFrame, group_col: str) -> pd.Series:
    """Per-group mul_factor using year-average pickups per group."""
    group_means = df.groupby(group_col)[["yellow_pickups", "fhv_pickups"]].mean()
    group_means["denom"] = group_means["yellow_pickups"] + group_means["fhv_pickups"]
    group_means["factor"] = group_means["yellow_pickups"] / group_means["denom"].replace(0, 1)
    return df[group_col].map(group_means["factor"])


def add_demand(
    df: pd.DataFrame,
    aggregation: str = "borough",
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Add 'demand' column (realized latent demand proxy) to df.

    aggregation: 'borough' | 'cluster' | 'global'
    demand_multiplier: uniform scaling for scenario analysis (agent-tunable).
    """
    df = df.copy()
    total_pickups = df["yellow_pickups"] + df["fhv_pickups"]

    if aggregation == "global":
        factors = _mul_factors_global(df)
    elif aggregation == "borough":
        if "borough_id" not in df.columns:
            raise ValueError("Run data.add_borough() before add_demand(aggregation='borough')")
        factors = _mul_factors_by_group(df, "borough_id")
    elif aggregation == "cluster":
        if "cluster_id" not in df.columns:
            raise ValueError("Run data.add_kmeans_cluster() before add_demand(aggregation='cluster')")
        factors = _mul_factors_by_group(df, "cluster_id")
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    df["mul_factor"] = factors.values
    df["demand"] = (df["mul_factor"] * total_pickups * demand_multiplier).round().astype(int)
    return df


def demand_sensitivity_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary comparing global vs borough vs cluster demand proxies.

    Useful for auditing how much the proxy choice shifts the realized demand.
    """
    rows = []
    for agg in ("global", "borough", "cluster"):
        tmp = add_demand(df, aggregation=agg)
        rows.append(
            {
                "aggregation": agg,
                "total_demand": tmp["demand"].sum(),
                "mean_demand_per_cell": tmp["demand"].mean(),
                "std_demand_per_cell": tmp["demand"].std(),
                "pct_undersupply_cells": (tmp["demand"] > tmp["yellow_pickups"]).mean() * 100,
            }
        )
    return pd.DataFrame(rows)
