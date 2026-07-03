"""Roll up zone-hour aggregates to zone × time_bucket × day_of_week × month × week."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl

ROOT = Path(__file__).resolve().parent
AGG_DIR = ROOT / "data" / "aggregate"

BUCKET_ORDER = [
    "00:00-05:59",
    "06:00-08:59",
    "09:00-15:59",
    "16:00-18:59",
    "19:00-23:59",
]


def _paths(stem: str) -> dict:
    base = AGG_DIR / stem
    return {"output": base.with_suffix(".parquet"), "output_csv": base.with_suffix(".csv")}


CONFIGS = {
    "2024": {
        "input": AGG_DIR / "zone_hour_agg_2024.parquet",
        **_paths("2024_agg"),
        "year": 2024,
    },
    "2025": {
        "input": AGG_DIR / "zone_hour_agg_2025.parquet",
        **_paths("updated_2025_agg"),
        "year": 2025,
    },
    "2026": {
        "input": AGG_DIR / "zone_hour_agg_2026_jan_apr.parquet",
        **_paths("updated_2026_jan_apr_agg"),
        "year": 2026,
    },
}

GROUP_COLS = ["year", "month", "week", "day_of_week", "time_bucket", "PULocationID"]
METRIC_COLS = [
    "pickup_count",
    "yellow_pickups",
    "fhv_pickups",
    "yellow_avg_total_fare",
    "yellow_avg_base_fare",
    "yellow_avg_tip",
    "fhv_avg_total_fare",
    "fhv_avg_base_fare",
    "fhv_avg_tip",
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "apparent_temperature",
]


def _time_bucket_expr() -> pl.Expr:
    h = pl.col("pickup_hour")
    return (
        pl.when(h.is_between(0, 5))
        .then(pl.lit("00:00-05:59"))
        .when(h.is_between(6, 8))
        .then(pl.lit("06:00-08:59"))
        .when(h.is_between(9, 15))
        .then(pl.lit("09:00-15:59"))
        .when(h.is_between(16, 18))
        .then(pl.lit("16:00-18:59"))
        .otherwise(pl.lit("19:00-23:59"))
        .alias("time_bucket")
    )


def _rollup(lf: pl.LazyFrame, group_cols: list[str]) -> pl.LazyFrame:
    return (
        lf.with_columns(_time_bucket_expr())
        .group_by(group_cols)
        .agg(
            pl.col("pickup_count").sum().alias("pickup_count"),
            pl.col("yellow_pickups").sum().alias("yellow_pickups"),
            pl.col("fhv_pickups").sum().alias("fhv_pickups"),
            (
                (pl.col("yellow_avg_total_fare") * pl.col("yellow_pickups")).sum()
                / pl.col("yellow_pickups").sum()
            ).alias("yellow_avg_total_fare"),
            (
                (pl.col("yellow_avg_base_fare") * pl.col("yellow_pickups")).sum()
                / pl.col("yellow_pickups").sum()
            ).alias("yellow_avg_base_fare"),
            (
                (pl.col("yellow_avg_tip") * pl.col("yellow_pickups")).sum()
                / pl.col("yellow_pickups").sum()
            ).alias("yellow_avg_tip"),
            (
                (pl.col("fhv_avg_total_fare") * pl.col("fhv_pickups")).sum()
                / pl.col("fhv_pickups").sum()
            ).alias("fhv_avg_total_fare"),
            (
                (pl.col("fhv_avg_base_fare") * pl.col("fhv_pickups")).sum()
                / pl.col("fhv_pickups").sum()
            ).alias("fhv_avg_base_fare"),
            (
                (pl.col("fhv_avg_tip") * pl.col("fhv_pickups")).sum()
                / pl.col("fhv_pickups").sum()
            ).alias("fhv_avg_tip"),
            (
                (pl.col("temperature_2m") * pl.col("pickup_count")).sum()
                / pl.col("pickup_count").sum()
            ).alias("temperature_2m"),
            (
                (pl.col("precipitation") * pl.col("pickup_count")).sum()
                / pl.col("pickup_count").sum()
            ).alias("precipitation"),
            (
                (pl.col("relative_humidity_2m") * pl.col("pickup_count")).sum()
                / pl.col("pickup_count").sum()
            ).alias("relative_humidity_2m"),
            (
                (pl.col("apparent_temperature") * pl.col("pickup_count")).sum()
                / pl.col("pickup_count").sum()
            ).alias("apparent_temperature"),
        )
    )


def build(year: str) -> None:
    cfg = CONFIGS[year]
    inp = cfg["input"]
    if not inp.exists():
        raise FileNotFoundError(inp)

    print(f"\n{'=' * 60}")
    print(f"Year {year}: {inp.name}")
    print("=" * 60)

    lf = pl.scan_parquet(inp)
    (
        _rollup(lf, ["time_bucket", "PULocationID", "day_of_week", "month", "week"])
        .with_columns(pl.lit(cfg["year"]).alias("year"))
        .with_columns(pl.col("time_bucket").cast(pl.Enum(BUCKET_ORDER)))
        .select(GROUP_COLS + METRIC_COLS)
        .sort(["month", "week", "day_of_week", "time_bucket", "PULocationID"])
        .sink_parquet(cfg["output"], compression="zstd")
    )

    con = duckdb.connect()
    path = str(cfg["output"].resolve())
    print(
        con.execute(f"""
        SELECT COUNT(*) AS rows,
               SUM(pickup_count) AS trips,
               COUNT(DISTINCT PULocationID) AS zones,
               COUNT(DISTINCT month) AS months,
               COUNT(DISTINCT day_of_week) AS days_of_week,
               COUNT(DISTINCT week) AS weeks,
               COUNT(DISTINCT time_bucket) AS time_buckets
        FROM read_parquet('{path}')
        """).df().to_string(index=False)
    )
    print(f"\nSaved: {cfg['output'].name}")
    export_csv(cfg["output"], cfg["output_csv"])


def export_csv(parquet_path: Path, csv_path: Path) -> None:
    con = duckdb.connect()
    group_csv = ", ".join(
        f"{c}::VARCHAR AS {c}" if c == "time_bucket" else c for c in GROUP_COLS
    )
    metric_csv = ", ".join(METRIC_COLS)
    con.execute(f"""
        COPY (
          SELECT {group_csv}, {metric_csv}
          FROM read_parquet('{parquet_path.resolve()}')
          ORDER BY month, week, day_of_week, time_bucket, PULocationID
        ) TO '{csv_path.resolve()}' (HEADER, DELIMITER ',')
    """)
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {csv_path.name} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2024", "2025", "2026", "both"], default="both")
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Export CSV from existing parquet without rebuilding",
    )
    args = parser.parse_args()
    years = ["2025", "2026"] if args.year == "both" else [args.year]
    for y in years:
        cfg = CONFIGS[y]
        if args.csv_only:
            if not cfg["output"].exists():
                raise FileNotFoundError(cfg["output"])
            print(f"\nExport CSV: {cfg['output'].name}")
            export_csv(cfg["output"], cfg["output_csv"])
        else:
            build(y)
    print("\nDone.")


if __name__ == "__main__":
    main()
