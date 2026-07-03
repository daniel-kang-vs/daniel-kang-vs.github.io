#!/usr/bin/env python3
"""Basic exploratory data analysis for cleaned_trips_2025.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent / "cleaned_trips_2025.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "eda_output"

DAY_NAMES = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
TIME_BUCKET_ORDER = [
    "00:00-05:59",
    "06:00-08:59",
    "09:00-15:59",
    "16:00-18:59",
    "19:00-23:59",
]
DEMAND_DIMS = ["trip_type", "PULocationID", "time_bucket", "month", "day_of_week", "week"]
TRIP_TYPE_COLORS = {"yellow": "#FFC107", "fhv": "#2196F3"}


def load_trips(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["pickup_datetime"],
        dtype={
            "trip_type": "category",
            "PULocationID": "Int16",
            "DOLocationID": "Int16",
            "month": "Int8",
            "day_of_week": "Int8",
            "week": "Int8",
        },
    )
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demand_table(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Trip counts (demand) grouped by one or more dimensions."""
    counts = df.groupby(dims, observed=True).size().rename("demand").reset_index()
    if "time_bucket" in dims:
        counts["time_bucket"] = pd.Categorical(
            counts["time_bucket"], categories=TIME_BUCKET_ORDER, ordered=True
        )
        counts = counts.sort_values([c for c in dims if c != "time_bucket"] + ["time_bucket"])
    elif len(dims) == 1:
        counts = counts.sort_values(dims[0])
    else:
        counts = counts.sort_values(dims)
    return counts


def save_demand_outputs(df: pd.DataFrame) -> None:
    """Build demand summaries and charts by trip_type, PU zone, and time dimensions."""
    demand_dir = OUTPUT_DIR / "demand"
    demand_dir.mkdir(parents=True, exist_ok=True)

    print_section("Demand by dimension (trip counts)")

    marginals: dict[str, pd.DataFrame] = {}
    for dim in DEMAND_DIMS:
        table = demand_table(df, [dim])
        marginals[dim] = table
        out = demand_dir / f"demand_by_{dim.lower()}.csv"
        table.to_csv(out, index=False)
        print(f"\n--- {dim} (saved {out.name}) ---")
        preview = table if dim != "PULocationID" else table.nlargest(15, "demand")
        print(preview.to_string(index=False))

    print_section("Demand cross-tabs with trip_type")
    cross_dims = ["time_bucket", "month", "day_of_week", "week", "PULocationID"]
    cross_tabs: dict[str, pd.DataFrame] = {}
    for dim in cross_dims:
        table = demand_table(df, ["trip_type", dim])
        cross_tabs[dim] = table
        out = demand_dir / f"demand_trip_type_x_{dim.lower()}.csv"
        table.to_csv(out, index=False)
        print(f"\n--- trip_type x {dim} (saved {out.name}) ---")
        if dim == "PULocationID":
            top_pu = (
                df["PULocationID"].value_counts().head(10).index.tolist()
            )
            preview = table[table["PULocationID"].isin(top_pu)].sort_values(
                ["PULocationID", "trip_type"]
            )
        elif dim == "day_of_week":
            preview = table.copy()
            preview["day_name"] = preview["day_of_week"].map(DAY_NAMES)
        else:
            preview = table
        print(preview.to_string(index=False))

    # --- Charts ---
    trip_types = sorted(df["trip_type"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    marginals["trip_type"].plot.bar(x="trip_type", y="demand", ax=ax, legend=False, color=[
        TRIP_TYPE_COLORS.get(t, "#888") for t in marginals["trip_type"]["trip_type"]
    ])
    ax.set_title("Demand by trip type")
    ax.set_ylabel("Trips")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(demand_dir / "demand_by_trip_type.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = cross_tabs["time_bucket"].pivot(
        index="time_bucket", columns="trip_type", values="demand"
    ).reindex(TIME_BUCKET_ORDER)
    pivot.plot(kind="bar", ax=ax, color=[TRIP_TYPE_COLORS[t] for t in pivot.columns])
    ax.set_title("Demand by time bucket and trip type")
    ax.set_ylabel("Trips")
    ax.set_xlabel("Time bucket")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Trip type")
    fig.tight_layout()
    fig.savefig(demand_dir / "demand_trip_type_x_time_bucket.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = cross_tabs["month"].pivot(index="month", columns="trip_type", values="demand")
    pivot.plot(kind="bar", ax=ax, color=[TRIP_TYPE_COLORS[t] for t in pivot.columns])
    ax.set_title("Demand by month and trip type")
    ax.set_ylabel("Trips")
    ax.set_xlabel("Month")
    ax.legend(title="Trip type")
    fig.tight_layout()
    fig.savefig(demand_dir / "demand_trip_type_x_month.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    dow = cross_tabs["day_of_week"].copy()
    dow["day_name"] = dow["day_of_week"].map(DAY_NAMES)
    pivot = dow.pivot(index="day_name", columns="trip_type", values="demand")
    pivot = pivot.reindex([DAY_NAMES[i] for i in range(1, 8)])
    pivot.plot(kind="bar", ax=ax, color=[TRIP_TYPE_COLORS[t] for t in pivot.columns])
    ax.set_title("Demand by day of week and trip type")
    ax.set_ylabel("Trips")
    ax.set_xlabel("Day of week")
    ax.legend(title="Trip type")
    fig.tight_layout()
    fig.savefig(demand_dir / "demand_trip_type_x_day_of_week.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for trip_type in trip_types:
        series = cross_tabs["week"].loc[cross_tabs["week"]["trip_type"] == trip_type]
        ax.plot(
            series["week"],
            series["demand"],
            marker="o",
            markersize=3,
            label=trip_type,
            color=TRIP_TYPE_COLORS.get(trip_type, None),
        )
    ax.set_title("Demand by ISO week and trip type")
    ax.set_ylabel("Trips")
    ax.set_xlabel("Week")
    ax.legend(title="Trip type")
    fig.tight_layout()
    fig.savefig(demand_dir / "demand_trip_type_x_week.png", dpi=120)
    plt.close(fig)

    top_n = 20
    top_pu_ids = df["PULocationID"].value_counts().head(top_n).index
    pu_demand = cross_tabs["PULocationID"]
    pu_demand = pu_demand[pu_demand["PULocationID"].isin(top_pu_ids)]
    pu_totals = pu_demand.groupby("PULocationID", observed=True)["demand"].sum()
    pu_order = pu_totals.sort_values().index

    fig, ax = plt.subplots(figsize=(11, 8))
    pivot = pu_demand.pivot(index="PULocationID", columns="trip_type", values="demand")
    pivot = pivot.reindex(pu_order)
    pivot.plot(kind="barh", stacked=True, ax=ax, color=[TRIP_TYPE_COLORS[t] for t in pivot.columns])
    ax.set_title(f"Demand by pickup zone (top {top_n}) and trip type")
    ax.set_xlabel("Trips")
    ax.set_ylabel("PULocationID")
    ax.legend(title="Trip type", loc="lower right")
    fig.tight_layout()
    fig.savefig(demand_dir / "demand_trip_type_x_pulocationid_top20.png", dpi=120)
    plt.close(fig)

    heat = demand_table(
        df[df["PULocationID"].isin(top_pu_ids)],
        ["PULocationID", "time_bucket"],
    )
    heat_pivot = heat.pivot(index="PULocationID", columns="time_bucket", values="demand")
    heat_pivot = heat_pivot.reindex(index=pu_order, columns=TIME_BUCKET_ORDER)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heat_pivot, cmap="YlOrRd", annot=False, fmt="d", ax=ax)
    ax.set_title(f"Demand heatmap: top {top_n} pickup zones x time bucket")
    ax.set_xlabel("Time bucket")
    ax.set_ylabel("PULocationID")
    fig.tight_layout()
    fig.savefig(demand_dir / "demand_pulocationid_x_time_bucket_heatmap.png", dpi=120)
    plt.close(fig)

    print_section("Saved demand outputs")
    for p in sorted(demand_dir.glob("*")):
        print(p.name)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    print_section("Loading data")
    df = load_trips(DATA_PATH)
    print(f"Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    print_section("Schema & missing values")
    print(df.dtypes.to_string())
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values.")
    else:
        pct = (missing / len(df) * 100).round(2)
        print(pd.DataFrame({"missing": missing, "pct": pct}).to_string())

    print_section("Trip type mix")
    trip_counts = df["trip_type"].value_counts()
    print(trip_counts.to_string())
    print(f"\nYellow share: {trip_counts.get('yellow', 0) / len(df):.1%}")

    print_section("Temporal coverage")
    print(f"Date range: {df['pickup_datetime'].min()} -> {df['pickup_datetime'].max()}")
    print(f"Months present: {sorted(df['month'].dropna().unique().tolist())}")
    print(f"Weeks (ISO): {int(df['week'].min())} - {int(df['week'].max())}")

    print_section("Trips by month")
    by_month = df.groupby("month", observed=True).size()
    print(by_month.to_string())

    print_section("Trips by day of week")
    by_dow = df.groupby("day_of_week", observed=True).size()
    by_dow.index = by_dow.index.map(DAY_NAMES)
    print(by_dow.to_string())

    print_section("Trips by time bucket")
    by_bucket = (
        df.groupby("time_bucket", observed=True)
        .size()
        .reindex(TIME_BUCKET_ORDER)
    )
    print(by_bucket.to_string())

    fare_cols = [
        "base_fare",
        "total_fare",
        "tip",
        "tolls",
        "extra",
        "mta_tax",
        "improvement_surcharge",
        "congestion_surcharge",
        "airport_fee",
        "cbd_congestion_fee",
        "bcf",
        "sales_tax",
    ]
    present_fare_cols = [c for c in fare_cols if c in df.columns]

    print_section("Fare summary (USD)")
    print(df[present_fare_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).T.to_string())

    print_section("Fare summary by trip type")
    print(
        df.groupby("trip_type", observed=True)[["base_fare", "total_fare", "tip"]]
        .agg(["count", "mean", "median", "std"])
        .round(2)
        .to_string()
    )

    print_section("Zero / negative fare checks")
    for col in ["base_fare", "total_fare"]:
        zero = (df[col] == 0).sum()
        neg = (df[col] < 0).sum()
        print(f"{col}: zero={zero:,} ({zero/len(df):.2%}), negative={neg:,}")

    print_section("Tip rate (tip / base_fare, where base_fare > 0)")
    tipped = df[df["base_fare"] > 0].copy()
    tipped["tip_rate"] = tipped["tip"] / tipped["base_fare"]
    print(f"Trips with tip > 0: {(tipped['tip'] > 0).sum():,} ({(tipped['tip'] > 0).mean():.1%})")
    print(tipped["tip_rate"].describe(percentiles=[0.5, 0.95]).round(3).to_string())

    print_section("Top 10 pickup locations (PULocationID)")
    print(df["PULocationID"].value_counts().head(10).to_string())

    print_section("Top 10 drop-off locations (DOLocationID)")
    print(df["DOLocationID"].value_counts().head(10).to_string())

    print_section("Same pickup & drop-off (intra-zone trips)")
    same_zone = (df["PULocationID"] == df["DOLocationID"]).sum()
    print(f"{same_zone:,} trips ({same_zone / len(df):.1%})")

    save_demand_outputs(df)

    # --- Plots (aggregates only; full data used for counts) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    trip_counts.plot(kind="bar", ax=axes[0, 0], color=["#FFC107", "#2196F3"][: len(trip_counts)])
    axes[0, 0].set_title("Trips by type")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=0)

    by_month.plot(kind="bar", ax=axes[0, 1], color="#4CAF50")
    axes[0, 1].set_title("Trips by month")
    axes[0, 1].set_xlabel("Month")

    by_dow.plot(kind="bar", ax=axes[1, 0], color="#9C27B0")
    axes[1, 0].set_title("Trips by day of week")
    axes[1, 0].set_xlabel("Day")

    by_bucket.plot(kind="bar", ax=axes[1, 1], color="#FF5722")
    axes[1, 1].set_title("Trips by time of day")
    axes[1, 1].set_xlabel("Time bucket")
    axes[1, 1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_temporal_distribution.png", dpi=120)
    plt.close(fig)

    sample = df.sample(n=min(50_000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "base_fare", "Base fare"),
        (axes[1], "total_fare", "Total fare"),
    ]:
        clipped = sample[col].clip(upper=sample[col].quantile(0.99))
        sns.histplot(clipped, bins=50, kde=True, ax=ax, color="#1976D2")
        ax.set_title(f"{title} (sample, 99th pct cap)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_fare_distributions.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    fare_by_type = df.groupby("trip_type", observed=True)["total_fare"].median()
    fare_by_type.plot(kind="bar", ax=ax, color=["#FFC107", "#2196F3"][: len(fare_by_type)])
    ax.set_title("Median total fare by trip type")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_median_fare_by_type.png", dpi=120)
    plt.close(fig)

    top_n = 15
    pu_top = df["PULocationID"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    pu_top.sort_values().plot(kind="barh", ax=ax, color="#00897B")
    ax.set_title(f"Top {top_n} pickup zones")
    ax.set_xlabel("Trip count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_top_pickup_zones.png", dpi=120)
    plt.close(fig)

    print_section("Saved plots")
    for p in sorted(OUTPUT_DIR.glob("*.png")):
        print(p.name)


if __name__ == "__main__":
    main()
