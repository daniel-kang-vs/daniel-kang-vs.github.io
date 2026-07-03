"""Load precomputed model results from outputs/ CSV files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def _read_csv(name: str) -> pd.DataFrame | None:
    path = OUTPUTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def find_q_star_csv() -> Path | None:
    matches = sorted(OUTPUTS_DIR.glob("q_star_*.csv"))
    return matches[0] if matches else None


def parse_q_star_filename(filename: str | None) -> dict[str, Any]:
    """Parse scenario from outputs/q_star_{bucket}_{dow}_wk{week}.csv."""
    if not filename:
        return {}
    stem = Path(filename).stem  # q_star_0900-1559_4_wk50
    m = re.match(r"q_star_(.+)_(\d+)_wk(\d+)$", stem)
    if not m:
        return {}
    bucket_token, dow, week = m.group(1), int(m.group(2)), int(m.group(3))
    tb_m = re.match(r"(\d{2})(\d{2})-(\d{2})(\d{2})", bucket_token)
    time_bucket = (
        f"{tb_m.group(1)}:{tb_m.group(2)}-{tb_m.group(3)}:{tb_m.group(4)}"
        if tb_m
        else bucket_token
    )
    return {
        "time_bucket": time_bucket,
        "day_of_week": dow,
        "week": week,
    }


def load_baseline_snapshot() -> dict[str, Any]:
    """Return bake-off rankings, scenario breakdown, allocation, and chart paths."""
    bakeoff = _read_csv("bakeoff_results.csv")
    by_scenario = _read_csv("bakeoff_by_scenario.csv")
    q_star_path = find_q_star_csv()
    q_star = pd.read_csv(q_star_path) if q_star_path else None

    best_model = None
    if bakeoff is not None and not bakeoff.empty:
        top = bakeoff.sort_values("rank").iloc[0]
        best_model = {
            "model": top["model"],
            "mean_nv_cost": float(top["mean_nv_cost"]),
            "mean_fill_rate": float(top["mean_fill_rate"]),
            "rank": int(top["rank"]),
            "n_scenarios": int(top["n_scenarios"]),
        }

    charts = {
        "bakeoff": OUTPUTS_DIR / "charts" / "10_bakeoff.png",
        "revenue": OUTPUTS_DIR / "charts" / "11_revenue_comparison.png",
        "allocation": OUTPUTS_DIR / "charts" / "08_allocation_top30.png",
    }

    scenario = parse_q_star_filename(q_star_path.name if q_star_path else None)
    fleet_used = None
    if q_star is not None and not q_star.empty and "q_star" in q_star.columns:
        fleet_used = int(q_star["q_star"].sum())

    return {
        "ready": bakeoff is not None,
        "best_model": best_model,
        "bakeoff": bakeoff,
        "by_scenario": by_scenario,
        "q_star": q_star,
        "q_star_file": q_star_path.name if q_star_path else None,
        "scenario": scenario,
        "fleet_used": fleet_used,
        "charts": {k: v for k, v in charts.items() if v.exists()},
    }
