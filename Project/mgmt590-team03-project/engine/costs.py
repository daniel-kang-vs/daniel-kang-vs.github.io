"""Compute newsvendor cost parameters Cu, Co, and critical fractile tau per zone×bucket.

Cu_{z,t} = underage cost = avg yellow total fare for zone z in bucket t (year mean).
           Represents the lost revenue per unserved trip.
Co_t     = overage cost  = all-zones average fare for bucket t (opportunity cost of idle cab).
           Bucket-level only (same across zones within a bucket).
tau_{z,t} = Cu_{z,t} / (Cu_{z,t} + Co_t)

High-fare zones (e.g. airports) get higher tau → stocked more aggressively.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


_FARE_CLIP_UPPER = 200.0   # clip yellow_avg_total_fare; the 2092.5 outlier in base_fare is separate


def compute_costs(
    df: pd.DataFrame,
    cu_multiplier: float = 1.0,
    co_multiplier: float = 1.0,
    co_mode: str = "flat",
    co_borough_multipliers: Optional[dict] = None,
    co_zone_overrides: Optional[dict] = None,
) -> pd.DataFrame:
    """Return a DataFrame indexed by (PULocationID, time_bucket) with columns:
    Cu, Co, tau, n_obs.

    When co_mode='borough', Co_z = Co_raw * alpha_z * co_multiplier where
    alpha_z = co_zone_overrides.get(zone, co_borough_multipliers.get(borough_id, 1.0)).
    When co_mode='flat' (default) every zone gets the same bucket-level Co.
    """
    fare_clipped = df["yellow_avg_total_fare"].clip(upper=_FARE_CLIP_UPPER)
    fare_weighted = fare_clipped * df["yellow_pickups"]

    # ── Cu: zone × bucket year-average fare ──────────────────────────────────
    cu_df = (
        df.assign(fare_w=fare_weighted)
        .groupby(["PULocationID", "time_bucket"], as_index=False)
        .agg(
            fare_sum=("fare_w", "sum"),
            n_yellow_obs=("yellow_pickups", "sum"),
        )
    )
    cu_df["Cu_raw"] = cu_df["fare_sum"] / cu_df["n_yellow_obs"].replace(0, np.nan)
    cu_df = cu_df.drop(columns=["fare_sum"])

    # Impute missing Cu (yellow_pickups always 0) from borough×bucket mean.
    if "borough_id" in df.columns:
        borough_mean = (
            df.assign(fare_w=fare_weighted)
            .groupby(["borough_id", "time_bucket"], as_index=False)
            .agg(
                fare_sum=("fare_w", "sum"),
                pickups=("yellow_pickups", "sum"),
            )
        )
        borough_mean["Cu_borough_mean"] = (
            borough_mean["fare_sum"] / borough_mean["pickups"].replace(0, np.nan)
        )
        borough_mean = borough_mean[["borough_id", "time_bucket", "Cu_borough_mean"]]

        zone_borough = df[["PULocationID", "borough_id"]].drop_duplicates()
        cu_df = cu_df.merge(zone_borough, on="PULocationID", how="left")
        cu_df = cu_df.merge(borough_mean, on=["borough_id", "time_bucket"], how="left")
        cu_df["Cu_raw"] = cu_df["Cu_raw"].fillna(cu_df["Cu_borough_mean"])

    global_bucket_mean = cu_df.groupby("time_bucket")["Cu_raw"].transform("mean")
    cu_df["Cu_raw"] = cu_df["Cu_raw"].fillna(global_bucket_mean)
    cu_df["Cu"] = (cu_df["Cu_raw"] * cu_multiplier).clip(lower=0.01)

    # ── Co: all-zones average fare per bucket ─────────────────────────────────
    co_series = (
        df.assign(fare_w=fare_clipped * df["yellow_pickups"])
        .groupby("time_bucket", as_index=False)
        .agg(
            fare_sum=("fare_w", "sum"),
            pickups=("yellow_pickups", "sum"),
        )
    )
    co_series["Co_raw"] = co_series["fare_sum"] / co_series["pickups"].replace(0, np.nan)
    co_series["Co"] = (co_series["Co_raw"] * co_multiplier).clip(lower=0.01)

    # ── Merge Cu and bucket-level Co ─────────────────────────────────────────
    costs = cu_df[["PULocationID", "time_bucket", "Cu", "n_yellow_obs"]].merge(
        co_series[["time_bucket", "Co_raw"]], on="time_bucket", how="left"
    )

    # ── Apply per-zone alpha (borough mode) ──────────────────────────────────
    if co_mode == "borough" and "borough_id" in df.columns:
        if co_borough_multipliers is None:
            co_borough_multipliers = {}
        if co_zone_overrides is None:
            co_zone_overrides = {}
        zone_borough = df[["PULocationID", "borough_id"]].drop_duplicates()
        costs = costs.merge(zone_borough, on="PULocationID", how="left")
        costs["borough_id"] = costs["borough_id"].fillna(5).astype(int)

        def _alpha(row):
            z = int(row["PULocationID"])
            if z in co_zone_overrides:
                return co_zone_overrides[z]
            return co_borough_multipliers.get(int(row["borough_id"]), 1.0)

        costs["alpha_z"] = costs.apply(_alpha, axis=1)
    else:
        costs["alpha_z"] = 1.0

    costs["Co"] = (costs["Co_raw"] * costs["alpha_z"] * co_multiplier).clip(lower=0.01)

    # ── Recompute tau per-zone ────────────────────────────────────────────────
    costs["tau"] = costs["Cu"] / (costs["Cu"] + costs["Co"])
    costs["tau"] = costs["tau"].clip(lower=1e-6, upper=1 - 1e-6)

    return costs


def get_costs_for_scenario(
    costs: pd.DataFrame,
    time_bucket: str,
) -> pd.DataFrame:
    """Return costs for a specific time bucket, indexed by PULocationID."""
    return costs[costs["time_bucket"] == time_bucket].set_index("PULocationID")
