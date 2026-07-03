"""Build engine-ready aggregates from cleaned_trips_2025 (cached, read-only source)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from agent.catalog import file_fingerprint
from agent.overlays import CACHE_ROOT, CANONICAL_SOURCE

CACHE_AGG = CACHE_ROOT / "agg_from_cleaned_trips_2025.parquet"
WEATHER_COLS = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "apparent_temperature",
]
NUMERIC_COLS = [
    "yellow_pickups",
    "fhv_pickups",
    "pickup_count",
    "yellow_avg_total_fare",
    "yellow_avg_base_fare",
    "yellow_avg_tip",
    "fhv_avg_total_fare",
    "fhv_avg_base_fare",
    "fhv_avg_tip",
    "zone_hours",
    *WEATHER_COLS,
]
BUCKET_HOURS = {
    "00:00-05:59": 6,
    "06:00-08:59": 3,
    "09:00-15:59": 7,
    "16:00-18:59": 3,
    "19:00-23:59": 5,
}


def _source_path() -> Path:
    if CANONICAL_SOURCE.exists():
        return CANONICAL_SOURCE
    raise FileNotFoundError(
        f"Missing {CANONICAL_SOURCE.name}. Place cleaned_trips_2025.parquet in project root."
    )


def _cache_is_fresh() -> bool:
    if not CACHE_AGG.exists():
        return False
    stamp = CACHE_ROOT / "agg_cache_stamp.txt"
    if not stamp.exists():
        return False
    return stamp.read_text().strip() == file_fingerprint(_source_path())


def build_agg_from_trips(source: Path, dest: Path) -> Path:
    """Aggregate trip-level cleaned_trips into engine weekly grid."""
    import duckdb

    dest.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT
                2025 AS year,
                CAST(month AS INTEGER) AS month,
                CAST(week AS INTEGER) AS week,
                CAST(day_of_week AS INTEGER) AS day_of_week,
                CAST(time_bucket AS VARCHAR) AS time_bucket,
                CAST(PULocationID AS INTEGER) AS PULocationID,
                SUM(CASE WHEN trip_type = 'yellow' THEN 1 ELSE 0 END) AS yellow_pickups,
                SUM(CASE WHEN trip_type = 'fhv' THEN 1 ELSE 0 END) AS fhv_pickups,
                SUM(CASE WHEN trip_type = 'yellow' THEN 1 ELSE 0 END)
                  + SUM(CASE WHEN trip_type = 'fhv' THEN 1 ELSE 0 END) AS pickup_count,
                AVG(CASE WHEN trip_type = 'yellow' THEN total_fare END) AS yellow_avg_total_fare,
                AVG(CASE WHEN trip_type = 'yellow' THEN base_fare END) AS yellow_avg_base_fare,
                AVG(CASE WHEN trip_type = 'yellow' THEN tip END) AS yellow_avg_tip,
                AVG(CASE WHEN trip_type = 'fhv' THEN total_fare END) AS fhv_avg_total_fare,
                AVG(CASE WHEN trip_type = 'fhv' THEN base_fare END) AS fhv_avg_base_fare,
                AVG(CASE WHEN trip_type = 'fhv' THEN tip END) AS fhv_avg_tip,
                0.0 AS temperature_2m,
                0.0 AS precipitation,
                0.0 AS relative_humidity_2m,
                0.0 AS apparent_temperature
            FROM read_parquet('{source.as_posix()}')
            GROUP BY 1,2,3,4,5,6
        ) TO '{dest.as_posix()}' (FORMAT PARQUET)
        """
    )
    con.close()

    df = pd.read_parquet(dest)
    df["zone_hours"] = df["time_bucket"].map(BUCKET_HOURS).fillna(1).astype(float)
    df = _coerce_engine_dtypes(df)
    df.to_parquet(dest, index=False)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    (CACHE_ROOT / "agg_cache_stamp.txt").write_text(file_fingerprint(source), encoding="utf-8")
    return dest


def build_agg_from_upload(upload_path: Path, dest: Path) -> Path:
    source = upload_path
    if upload_path.suffix.lower() == ".csv":
        import duckdb

        tmp = upload_path.with_suffix(".parquet")
        con = duckdb.connect()
        con.execute(
            f"""
            COPY (SELECT * FROM read_csv_auto('{upload_path.as_posix()}'))
            TO '{tmp.as_posix()}' (FORMAT PARQUET)
            """
        )
        con.close()
        source = tmp
    return build_agg_from_trips(source, dest)


def ensure_cached_agg_from_cleaned_trips() -> Path:
    if CANONICAL_SOURCE.exists():
        if _cache_is_fresh():
            return CACHE_AGG
        return build_agg_from_trips(CANONICAL_SOURCE, CACHE_AGG)

    # No cleaned_trips_2025.parquet — fall back to the multi-year agg parquets
    # (2024_agg/2025_agg/2026_jan_apr_agg) that run_stakeholder_tasks.py uses.
    if not CACHE_AGG.exists():
        from engine.data import load_raw

        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        df = _coerce_engine_dtypes(load_raw())
        df.to_parquet(CACHE_AGG, index=False)
    return CACHE_AGG


def _coerce_engine_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns (especially weather) are float — object dtypes break models."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def load_raw_agg(session_id: str) -> pd.DataFrame:
    from agent.overlays import resolve_agg_parquet_path

    path = resolve_agg_parquet_path(session_id)
    df = pd.read_parquet(path)
    df["time_bucket"] = df["time_bucket"].astype(str)
    if "PULocationID" in df.columns:
        df["PULocationID"] = pd.to_numeric(df["PULocationID"], errors="coerce")
        df = df.dropna(subset=["PULocationID"])
        df["PULocationID"] = df["PULocationID"].astype(int)
    df = _coerce_engine_dtypes(df)
    df = (
        df.sort_values("month")
        .drop_duplicates(
            subset=["year", "week", "day_of_week", "time_bucket", "PULocationID"],
            keep="last",
        )
        .reset_index(drop=True)
    )
    return df


def load_prepared_dataframe(session_id: str, *, n_clusters: int = 5, seed: int = 42) -> pd.DataFrame:
    """Engine prepare() equivalent using cleaned_trips-derived agg (overlay-aware)."""
    from engine import data as data_mod

    df = load_raw_agg(session_id)
    df = data_mod.add_borough(df)
    df = data_mod.add_kmeans_cluster(df, n_clusters=n_clusters, seed=seed)
    df = data_mod.add_cyclical_features(df)
    return _coerce_engine_dtypes(df)
