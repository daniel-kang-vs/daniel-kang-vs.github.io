"""Data loading, borough/cluster mapping, and temporal splits."""

from __future__ import annotations

import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from engine.numeric import safe_int

_ROOT = pathlib.Path(__file__).parent.parent
DATA_PATHS = [
    _ROOT / "2024_agg.parquet",
    _ROOT / "2025_agg.parquet",
    _ROOT / "2026_jan_apr_agg.parquet",
]

# TLC borough mapping for the 263 zones in the dataset.
# Source: NYC TLC taxi-zone lookup (borough field).  We inline a simplified version here so
# the engine has no external file dependency.  Unknown zones fall back to borough_id=5 (Other).
_BOROUGH_MAP: dict[int, int] = {}  # populated lazily by _build_borough_map()

# Borough id → name (for readability in reports)
BOROUGH_NAMES = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island / Other"}

# TLC zone-to-borough lookup (LocationID -> borough string -> int)
# Full list derived from https://data.cityofnewyork.us/api/views/755u-8jsi/rows.csv
_TLC_BOROUGH_STR: dict[int, str] = {
    # Manhattan (1)
    **{z: "Manhattan" for z in [
        4,12,13,24,41,42,43,45,48,50,68,74,75,79,87,88,90,100,
        107,113,114,116,120,125,127,128,137,140,141,142,143,144,
        148,151,152,153,158,161,162,163,164,166,170,186,194,202,
        209,211,224,229,230,231,232,233,234,236,237,238,239,243,
        244,246,249,261,262,263
    ]},
    # Bronx (2)
    **{z: "Bronx" for z in [
        3,18,20,31,32,46,47,51,58,59,60,69,78,81,94,119,126,
        136,147,159,167,168,169,174,182,183,184,185,199,200,208,
        212,213,220,235,240,241,242,247,248
    ]},
    # Brooklyn (3)
    **{z: "Brooklyn" for z in [
        11,14,17,21,22,25,26,29,33,34,35,36,37,39,40,49,52,54,
        55,61,62,63,65,66,67,71,72,76,77,80,85,89,91,97,106,108,
        111,112,123,133,149,150,154,155,165,177,178,181,188,189,
        190,195,210,217,222,225,227,228,255,256,257,258
    ]},
    # Queens (4)
    **{z: "Queens" for z in [
        2,7,8,9,10,15,16,19,27,28,30,38,53,56,57,64,70,73,82,
        83,86,92,93,95,96,98,101,102,117,121,122,124,129,130,131,
        132,134,135,138,139,145,146,157,160,171,173,175,179,180,
        191,192,193,196,197,198,201,203,205,207,215,216,218,219,
        221,226,252,253,260
    ]},
    # Staten Island (5)
    **{z: "Staten Island / Other" for z in [
        5,6,23,44,84,99,109,110,115,118,156,172,176,187,204,206,
        214,245,251,254,259
    ]},
}


def _build_borough_map(zones: np.ndarray) -> dict[int, int]:
    str_to_int = {
        "Manhattan": 1, "Bronx": 2, "Brooklyn": 3,
        "Queens": 4, "Staten Island / Other": 5
    }
    m = {}
    for z in zones:
        if pd.isna(z) or not np.isfinite(float(z)):
            continue
        z_int = safe_int(z)
        borough_str = _TLC_BOROUGH_STR.get(z_int, "Staten Island / Other")
        m[z_int] = str_to_int[borough_str]
    return m


def _sanitize_agg_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop/coerce rows with invalid keys before int casts in downstream code."""
    df = df.copy()
    if "PULocationID" in df.columns:
        df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce")
        df = df.dropna(subset=["PULocationID"])
        df["PULocationID"] = df["PULocationID"].astype(int)
    for col in ("week", "month", "day_of_week", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=[col])
            df[col] = df[col].astype(int)
    if "time_bucket" in df.columns:
        df = df[df["time_bucket"].notna()]
        df["time_bucket"] = df["time_bucket"].astype(str)
    return df


def _normalize_agg_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _sanitize_agg_df(df)
    if "year" not in df.columns:
        df["year"] = 2025
    return (
        df.sort_values(["year", "month"])
        .drop_duplicates(
            subset=["year", "week", "day_of_week", "time_bucket", "PULocationID"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def _load_fallback_agg() -> pd.DataFrame | None:
    """Use agent cache or cleaned_trips when multi-year agg parquets are absent."""
    import warnings

    cleaned = _ROOT / "cleaned_trips_2025.parquet"
    cache = _ROOT / "agent_runs" / "cache" / "agg_from_cleaned_trips_2025.parquet"
    stamp = _ROOT / "agent_runs" / "cache" / "agg_cache_stamp.txt"

    if cache.exists():
        try:
            return _normalize_agg_df(pd.read_parquet(cache))
        except OSError as exc:
            warnings.warn(f"Corrupt agg cache ({exc}); will rebuild from cleaned_trips.")
            cache.unlink(missing_ok=True)
            if stamp.exists():
                stamp.unlink(missing_ok=True)

    zone_agg = _ROOT / "zone_time_bucket_dow_month_agg_2025.parquet"
    if zone_agg.exists():
        try:
            return _normalize_agg_df(pd.read_parquet(zone_agg))
        except OSError as exc:
            warnings.warn(f"Could not read {zone_agg.name}: {exc}")

    if cleaned.exists():
        from agent.data_source import build_agg_from_trips

        cache.parent.mkdir(parents=True, exist_ok=True)
        build_agg_from_trips(cleaned, cache)
        return _normalize_agg_df(pd.read_parquet(cache))
    return None


def load_raw() -> pd.DataFrame:
    """Load multi-year agg parquets, or fall back to cleaned_trips-derived cache."""
    frames = []
    for p in DATA_PATHS:
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            import warnings
            warnings.warn(f"Data file not found, skipping: {p}")

    if frames:
        return _normalize_agg_df(pd.concat(frames, ignore_index=True))

    fallback = _load_fallback_agg()
    if fallback is not None:
        import warnings
        warnings.warn(
            "Using fallback aggregate from cleaned_trips cache "
            "(place 2024/2025/2026 agg parquets in project root for full 3-year data)."
        )
        return fallback

    raise FileNotFoundError(
        f"None of the data files found: {DATA_PATHS}. "
        f"Place cleaned_trips_2025.parquet in {_ROOT} or add the yearly agg parquets."
    )


def add_borough(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'borough_id' column via the TLC static lookup."""
    global _BOROUGH_MAP
    zones = df["PULocationID"].dropna().unique()
    if not _BOROUGH_MAP:
        _BOROUGH_MAP = _build_borough_map(zones)
    df = df.copy()
    df["borough_id"] = df["PULocationID"].map(_BOROUGH_MAP).fillna(5).astype(int)
    return df


def add_kmeans_cluster(df: pd.DataFrame, n_clusters: int = 5, seed: int = 42) -> pd.DataFrame:
    """Add a 'cluster_id' column based on KMeans over zone-level activity profiles.

    Profile features: mean yellow_pickups, fhv_pickups, yellow_avg_total_fare per zone.
    """
    profile = (
        df.groupby("PULocationID")
        .agg(
            mean_yellow=("yellow_pickups", "mean"),
            mean_fhv=("fhv_pickups", "mean"),
            mean_fare=("yellow_avg_total_fare", "mean"),
        )
        .fillna(0)
    )
    X = StandardScaler().fit_transform(profile.values)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
    cluster_map = dict(zip(profile.index, labels.astype(int)))
    df = df.copy()
    df["cluster_id"] = df["PULocationID"].map(cluster_map).fillna(0).astype(int)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings for week, month, and day_of_week to avoid ordinal artifacts."""
    df = df.copy()
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * (df["day_of_week"] - 1) / 7)
    df["dow_cos"] = np.cos(2 * np.pi * (df["day_of_week"] - 1) / 7)
    return df


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test split — cross-year when 2026 data exists, else ISO-week split."""
    if (df["year"] == 2026).any():
        yr2026 = df["year"] == 2026
        month_counts = df.loc[yr2026, "month"].value_counts()
        full_months = month_counts[month_counts >= 100].index
        if len(full_months) == 0:
            full_months = month_counts.index
        max_month = safe_int(full_months.max(), default=1)
        val_month = max(safe_int(max_month) - 1, 1)

        test_mask = yr2026 & (df["month"] == max_month)
        val_mask = yr2026 & (df["month"] == val_month)
        train_mask = ~test_mask & ~val_mask
        return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()

    train = df[df["week"] <= 44].copy()
    val = df[(df["week"] >= 45) & (df["week"] <= 48)].copy()
    test = df[df["week"] >= 49].copy()
    return train, val, test


def prepare(n_clusters: int = 5, seed: int = 42) -> pd.DataFrame:
    """Full pipeline: load → borough → cluster → cyclical features."""
    df = load_raw()
    df = add_borough(df)
    df = add_kmeans_cluster(df, n_clusters=n_clusters, seed=seed)
    df = add_cyclical_features(df)
    return df
