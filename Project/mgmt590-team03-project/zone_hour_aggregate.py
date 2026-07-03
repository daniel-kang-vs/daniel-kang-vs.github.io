"""Zone × hour aggregation — yellow/FHV counts and type-specific avg fares + weather."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
AGG_DIR = DATA / "aggregate"

CONFIGS = {
    "2024": {
        "trip_candidates": [DATA / "cleaned_trips_2024.parquet"],
        "weather": DATA / "2024_weather.csv",
        "output": AGG_DIR / "zone_hour_agg_2024.parquet",
        "year_filter": 2024,
    },
    "2025": {
        "trip_candidates": [
            DATA / "cleaned_trips_2025_with_weather.parquet",
            DATA / "cleaned_trips_2025_with_hour.parquet",
            DATA / "cleaned_trips_2025.parquet",
        ],
        "weather": DATA / "2025_weather.csv",
        "output": AGG_DIR / "zone_hour_agg_2025.parquet",
        "year_filter": 2025,
    },
    "2026": {
        "trip_candidates": [
            DATA / "cleaned_trips_2026_jan_apr_with_hour_weather.parquet",
            DATA / "cleaned_trips_2026_jan_apr_with_hour.parquet",
            DATA / "cleaned_trips_2026_jan_apr.parquet",
        ],
        "weather": DATA / "2026_weather.csv",
        "output": AGG_DIR / "zone_hour_agg_2026_jan_apr.parquet",
        "year_filter": 2026,
    },
}


def _load_weather(path: Path) -> pl.LazyFrame:
    return (
        pl.read_csv(
            path,
            schema_overrides={"precipitation (inch)": pl.Float64},
        )
        .rename({
            "temperature_2m (°F)": "temperature_2m",
            "precipitation (inch)": "precipitation",
            "relative_humidity_2m (%)": "relative_humidity_2m",
            "apparent_temperature (°F)": "apparent_temperature",
        })
        .with_columns(
            pl.col("time")
            .str.to_datetime("%Y-%m-%dT%H:%M")
            .dt.truncate("1h")
            .cast(pl.Datetime("ms"))
            .alias("datetime_hour")
        )
        .select(
            "datetime_hour",
            "temperature_2m",
            "precipitation",
            "relative_humidity_2m",
            "apparent_temperature",
        )
        .unique(subset=["datetime_hour"])
        .lazy()
    )


def aggregate(year: str) -> Path:
    cfg = CONFIGS[year]
    trips_path = next(p for p in cfg["trip_candidates"] if p.exists())
    weather_path = cfg["weather"]
    output_path = cfg["output"]
    year_filter = cfg["year_filter"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input trips: {trips_path.name}")
    print(f"Weather:     {weather_path.name}")
    print(f"Output:      {output_path.name}")
    print(f"Year filter: {year_filter}")

    weather = _load_weather(weather_path)

    trips = pl.scan_parquet(trips_path, parallel="none").filter(
        pl.col("pickup_datetime").dt.year() == year_filter
    )

    (
        trips.with_columns(
            pl.col("pickup_datetime")
            .dt.truncate("1h")
            .cast(pl.Datetime("ms"))
            .alias("datetime_hour"),
        )
        .group_by(["datetime_hour", "PULocationID"])
        .agg(
            pl.len().alias("pickup_count"),
            pl.col("trip_type").eq("yellow").sum().alias("yellow_pickups"),
            pl.col("trip_type").eq("fhv").sum().alias("fhv_pickups"),
            pl.col("total_fare")
            .filter(pl.col("trip_type") == "yellow")
            .mean()
            .alias("yellow_avg_total_fare"),
            pl.col("base_fare")
            .filter(pl.col("trip_type") == "yellow")
            .mean()
            .alias("yellow_avg_base_fare"),
            pl.col("tip").filter(pl.col("trip_type") == "yellow").mean().alias("yellow_avg_tip"),
            pl.col("total_fare")
            .filter(pl.col("trip_type") == "fhv")
            .mean()
            .alias("fhv_avg_total_fare"),
            pl.col("base_fare")
            .filter(pl.col("trip_type") == "fhv")
            .mean()
            .alias("fhv_avg_base_fare"),
            pl.col("tip").filter(pl.col("trip_type") == "fhv").mean().alias("fhv_avg_tip"),
        )
        .with_columns(
            pl.col("datetime_hour").dt.hour().alias("pickup_hour"),
            pl.col("datetime_hour").dt.month().alias("month"),
            pl.col("datetime_hour").dt.weekday().alias("day_of_week"),
            pl.col("datetime_hour").dt.week().alias("week"),
        )
        .join(weather, on="datetime_hour", how="left")
        .sort(["datetime_hour", "PULocationID"])
        .sink_parquet(output_path, compression="zstd")
    )

    out = pl.scan_parquet(output_path)
    n_rows = out.select(pl.len()).collect().item()
    summary = out.select(
        pl.len().alias("zone_hours"),
        pl.col("pickup_count").sum().alias("total_pickups"),
        pl.col("yellow_pickups").sum().alias("total_yellow"),
        pl.col("fhv_pickups").sum().alias("total_fhv"),
        pl.col("datetime_hour").min().alias("min_hour"),
        pl.col("datetime_hour").max().alias("max_hour"),
    ).collect()
    print(f"\nDone. Aggregated rows: {n_rows:,}")
    print(summary)
    print(out.collect_schema())
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zone × hour aggregate parquet")
    parser.add_argument(
        "--year",
        choices=sorted(CONFIGS),
        default="2026",
        help="Dataset year to aggregate (default: 2026)",
    )
    args = parser.parse_args()
    aggregate(args.year)


if __name__ == "__main__":
    main()
