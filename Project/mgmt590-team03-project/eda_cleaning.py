#!/usr/bin/env python3
"""
eda_cleaning.py
===============

Exploratory data analysis (EDA) and cleaning for the NYC TLC yellow-cab and
high-volume FHV (fhvhv) trip data produced by ``combine_tlc_parquet.py``.

Built up step by step. Implemented so far:

    Step 1 -- Read the yellow-cab and fhv combined parquet files, tag each row
              with a ``trip_type`` ("yellow"/"fhv"), keep only the pickup
              datetime, pickup/drop-off location IDs, and fare columns, and
              stack them into one DataFrame ``trips_df``.
    Step 2 -- Add ``time_bucket``, ``month``, ``day_of_week`` (1-7, Mon=1), and
              ``week`` (ISO week) derived from the pickup datetime.
    Step 3 -- Drop rows whose base_fare or total_fare is negative.

Laid out for interactive debugging: edit the CONFIG block, run top-to-bottom,
then inspect the leftover variables (``yellow_df``, ``fhv_df``, ``trips_df``).
Each helper is standalone -- nothing is hidden in a main().

MEMORY: the combined fhvhv file has hundreds of millions of rows. The reader
defaults to a representative random sample drawn across the whole file and only
pulls the columns we keep (column projection), so it stays light. Set the
relevant READ_MODE_* to "full" only if the machine can hold the whole file.

Requires: pyarrow, pandas, numpy  (pip install pyarrow pandas numpy)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"{exc}\nThis script needs pandas + numpy. Install them with:\n"
        "    pip install pandas numpy pyarrow"
    ) from exc


# =========================================================================== #
# CONFIG -- edit these, then run the file.
# =========================================================================== #

INPUT_DIR = Path("C:/Users/harem/OneDrive/Desktop/Industry Practicum")

YELLOW_PARQUET = INPUT_DIR / "yellow_tripdata_2025_combined.parquet"
FHV_PARQUET = INPUT_DIR / "fhvhv_tripdata_2025_combined.parquet"

# Read mode per file: "sample" (representative random sample, memory-safe),
# "full" (entire file -- only if it fits in RAM), or "head" (first N rows, fast
# but time-biased).
READ_MODE_YELLOW = "sample"
READ_MODE_FHV = "sample"
SAMPLE_SIZE = 1_000_000      # target rows for "sample"/"head"
RANDOM_STATE = 42            # reproducible sampling
BATCH_SIZE = 250_000         # rows per streamed batch

# --- Column selection / harmonization -------------------------------------- #
# Map each service's source columns -> unified column names. Only these columns
# are read and kept; everything else (IDs, distances, flags, zone text, etc.)
# is dropped per the "keep only ..." requirement. Location IDs are retained, so
# zone labels can always be re-derived from taxi_zone_lookup.csv later.
YELLOW_KEEP: Dict[str, str] = {
    "tpep_pickup_datetime": "pickup_datetime",
    "PULocationID": "PULocationID",
    "DOLocationID": "DOLocationID",
    "fare_amount": "base_fare",            # yellow base (metered) fare
    "total_amount": "total_fare",          # yellow grand total
    "tip_amount": "tip",
    "tolls_amount": "tolls",
    "extra": "extra",
    "mta_tax": "mta_tax",
    "improvement_surcharge": "improvement_surcharge",
    "congestion_surcharge": "congestion_surcharge",
    "Airport_fee": "airport_fee",          # note: capital A in the yellow schema
    "cbd_congestion_fee": "cbd_congestion_fee",
}

FHV_KEEP: Dict[str, str] = {
    "pickup_datetime": "pickup_datetime",
    "PULocationID": "PULocationID",
    "DOLocationID": "DOLocationID",
    "base_passenger_fare": "base_fare",    # fhv base passenger fare
    "tips": "tip",
    "tolls": "tolls",
    "bcf": "bcf",                          # Black Car Fund
    "sales_tax": "sales_tax",
    "congestion_surcharge": "congestion_surcharge",
    "airport_fee": "airport_fee",
    "cbd_congestion_fee": "cbd_congestion_fee",
}

# fhv has no native total column; derive total_fare as the sum of the
# passenger-paid components below (source column names). driver_pay is excluded
# (it is paid TO the driver, not BY the passenger).
FHV_TOTAL_COMPONENTS: List[str] = [
    "base_passenger_fare", "tolls", "bcf", "sales_tax",
    "congestion_surcharge", "airport_fee", "tips", "cbd_congestion_fee",
]

# --- Time buckets ----------------------------------------------------------- #
# pd.cut bins are right-inclusive, so e.g. (5, 8] captures hours 6,7,8.
# NOTE: the requested 5th bucket read "17 to 11:59", which overlaps bucket 4;
# interpreted as 19:00-23:59 to make the five buckets a gap-free partition of
# the day. Adjust here if a different split was intended.
TIME_BUCKET_BINS: List[int] = [-1, 5, 8, 15, 18, 23]
TIME_BUCKET_LABELS: List[str] = [
    "00:00-05:59", "06:00-08:59", "09:00-15:59", "16:00-18:59", "19:00-23:59",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eda_cleaning")


# =========================================================================== #
# Helpers -- standalone and independently callable.
# =========================================================================== #

def read_trip_parquet(
    path: Path,
    mode: str = "sample",
    columns: Optional[Sequence[str]] = None,
    sample_size: int = SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Read a (possibly very large) trip-data parquet file into a DataFrame.

    Only ``columns`` are read (column projection); any requested column absent
    from the file is skipped with a warning. ``mode`` controls how many rows:

      * "sample" -- stream the whole file, keeping each row with probability
        ``sample_size / total_rows`` (Bernoulli). Representative of the entire
        file while only holding the kept rows in memory. Reads in full if the
        file already has <= ``sample_size`` rows.
      * "full"   -- read every row (may exhaust memory on large files).
      * "head"   -- first ``sample_size`` rows only (fast, but time-biased).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {path}\n"
            "Run combine_tlc_parquet.py first, or fix the path in CONFIG."
        )
    if mode not in {"sample", "full", "head"}:
        raise ValueError(f"mode must be 'sample', 'full', or 'head' (got {mode!r}).")

    parquet_file = pq.ParquetFile(path)
    available = set(parquet_file.schema_arrow.names)

    cols: Optional[List[str]] = None
    if columns is not None:
        cols = [c for c in columns if c in available]
        missing = [c for c in columns if c not in available]
        if missing:
            logger.warning("%s missing requested column(s): %s", path.name, missing)

    total_rows = parquet_file.metadata.num_rows
    logger.info("%s: %s rows on disk; reading mode=%r.", path.name, f"{total_rows:,}", mode)

    if mode == "full":
        if total_rows > 20_000_000:
            logger.warning("Loading %s rows in full -- may use a lot of RAM.", f"{total_rows:,}")
        return parquet_file.read(columns=cols).to_pandas()

    if mode == "head":
        first = next(parquet_file.iter_batches(batch_size=sample_size, columns=cols), None)
        df = pa.Table.from_batches([first]).to_pandas() if first is not None else pd.DataFrame()
        logger.info("Loaded first %s rows of %s.", f"{len(df):,}", path.name)
        return df

    # mode == "sample"
    if total_rows <= sample_size:
        logger.info("File has <= sample_size rows; reading in full instead of sampling.")
        return parquet_file.read(columns=cols).to_pandas()

    frac = sample_size / total_rows
    rng = np.random.default_rng(random_state)
    kept_tables = []
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=cols):
        mask = rng.random(batch.num_rows) < frac
        if mask.any():
            kept_tables.append(pa.Table.from_batches([batch]).filter(pa.array(mask)))

    if not kept_tables:
        logger.warning("Sampling kept no rows; returning empty frame.")
        return pd.DataFrame()

    df = pa.concat_tables(kept_tables).to_pandas()
    logger.info("Sampled %s of %s rows (~%.3f%%) from %s.",
                f"{len(df):,}", f"{total_rows:,}", 100 * frac, path.name)
    return df


def prepare_service(
    raw: pd.DataFrame,
    mapping: Dict[str, str],
    trip_type: str,
    total_components: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Select + rename a service's columns to the unified schema and tag it.

    Keeps only the mapped source columns that are present, renames them, inserts
    a leading ``trip_type`` column, and -- if ``total_components`` is given and
    the result has no ``total_fare`` yet -- derives ``total_fare`` as the row sum
    of those (source) components.
    """
    present = {src: dst for src, dst in mapping.items() if src in raw.columns}
    missing = [src for src in mapping if src not in raw.columns]
    if missing:
        logger.warning("[%s] source columns absent (skipped): %s", trip_type, missing)

    out = raw[list(present)].rename(columns=present).copy()
    out.insert(0, "trip_type", trip_type)

    if "total_fare" not in out.columns and total_components:
        comps = [c for c in total_components if c in raw.columns]
        if comps:
            # skipna=True so a missing component is treated as 0 rather than
            # nulling the whole total.
            out["total_fare"] = raw[comps].sum(axis=1, skipna=True).to_numpy()
            logger.info("[%s] derived total_fare from %d components.", trip_type, len(comps))
        else:
            out["total_fare"] = np.nan
            logger.warning("[%s] no components available to derive total_fare.", trip_type)

    return out


def add_time_features(df: pd.DataFrame, datetime_col: str = "pickup_datetime") -> pd.DataFrame:
    """Add time_bucket, month, day_of_week (1-7, Mon=1), and week (ISO) columns.

    Operates in place and also returns ``df``. Rows with an unparseable/missing
    pickup datetime get null time features rather than raising.
    """
    if datetime_col not in df.columns:
        raise KeyError(f"{datetime_col!r} not found; cannot derive time features.")

    dt = pd.to_datetime(df[datetime_col], errors="coerce")
    if dt.isna().any():
        logger.warning("%d rows have unparseable %s; their time features are null.",
                        int(dt.isna().sum()), datetime_col)

    df["time_bucket"] = pd.cut(
        dt.dt.hour, bins=TIME_BUCKET_BINS, labels=TIME_BUCKET_LABELS, ordered=True
    )
    df["month"] = dt.dt.month.astype("Int64")              # 1-12
    df["day_of_week"] = (dt.dt.dayofweek + 1).astype("Int64")  # Mon=1 ... Sun=7
    df["week"] = dt.dt.isocalendar().week.astype("Int64")  # ISO week (1-53)
    return df


def drop_negative_fares(
    df: pd.DataFrame,
    fare_cols: Sequence[str] = ("base_fare", "total_fare"),
) -> pd.DataFrame:
    """Return ``df`` without rows where any of ``fare_cols`` is < 0.

    NaN fares are kept (NaN < 0 is False); only genuinely negative fares are
    removed. Logs how many rows were dropped.
    """
    present = [c for c in fare_cols if c in df.columns]
    if not present:
        logger.warning("None of %s present; no fare filtering applied.", tuple(fare_cols))
        return df

    negative = pd.Series(False, index=df.index)
    for col in present:
        negative |= df[col] < 0

    removed = int(negative.sum())
    logger.info("Dropping %s rows with negative %s (of %s total).",
                f"{removed:,}", " or ".join(present), f"{len(df):,}")
    return df.loc[~negative].copy()


def first_look(df: pd.DataFrame, name: str) -> None:
    """Log a concise look at a DataFrame (shape, memory, dtypes)."""
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    logger.info("[%s] shape = %d rows x %d cols, ~%.1f MB.",
                name, df.shape[0], df.shape[1], mem_mb)
    logger.info("[%s] dtypes:\n%s", name, df.dtypes.to_string())


# =========================================================================== #
# Driver -- top-level so the DataFrames stay inspectable after a run.
# Comment this out to import the helpers only.
# =========================================================================== #

# --- Step 1: read, tag with trip_type, keep wanted columns, stack together. ---
yellow_raw = read_trip_parquet(YELLOW_PARQUET, mode=READ_MODE_YELLOW, columns=list(YELLOW_KEEP))
fhv_raw = read_trip_parquet(FHV_PARQUET, mode=READ_MODE_FHV, columns=list(FHV_KEEP))

yellow_df = prepare_service(yellow_raw, YELLOW_KEEP, "yellow")
fhv_df = prepare_service(fhv_raw, FHV_KEEP, "fhv", total_components=FHV_TOTAL_COMPONENTS)

trips_df = pd.concat([yellow_df, fhv_df], ignore_index=True)
trips_df["trip_type"] = trips_df["trip_type"].astype("category")
logger.info("Combined trips_df: %s rows (%s).",
            f"{len(trips_df):,}",
            ", ".join(f"{k}={v:,}" for k, v in trips_df["trip_type"].value_counts().items()))

# --- Step 2: add time-based feature columns. ---
trips_df = add_time_features(trips_df, datetime_col="pickup_datetime")

# --- Step 3: remove rows with a negative base or total fare. ---
trips_df = drop_negative_fares(trips_df, fare_cols=("base_fare", "total_fare"))

first_look(trips_df, "trips_df")

# Next EDA/cleaning steps will go here as you specify them.
trips_df.to_csv('C:/Users/harem/OneDrive/Desktop/Industry Practicum/cleaned_trips_2025.csv')