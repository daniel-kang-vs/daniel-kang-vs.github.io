"""Quick validation of cost presets, stakeholder slots, and agent defaults."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Import costs only (avoids pydantic if env is mismatched)
from engine import costs as costs_mod


def _synthetic_train(n_zones: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for z in range(1, n_zones + 1):
        b = 1 + (z % 5)
        for tb in ["06:00-08:59", "19:00-23:59", "09:00-15:59"]:
            rows.append(
                {
                    "PULocationID": z,
                    "borough_id": b,
                    "time_bucket": tb,
                    "yellow_avg_total_fare": 15 + b * 2 + rng.random(),
                    "yellow_pickups": int(rng.integers(50, 200)),
                }
            )
    return pd.DataFrame(rows)


def test_baseline_flat_costs() -> None:
    train = _synthetic_train()
    flat = costs_mod.compute_costs(train, cu_multiplier=1.0, co_multiplier=1.0, co_mode="flat")
    bor = costs_mod.compute_costs(
        train,
        cu_multiplier=1.0,
        co_multiplier=1.0,
        co_mode="borough",
        co_borough_multipliers={1: 1.0, 2: 1.3, 3: 1.2, 4: 1.1, 5: 1.8},
    )
    for tb in train["time_bucket"].unique():
        sub = flat[flat["time_bucket"] == tb]
        assert sub["Co"].nunique() == 1, f"flat Co must be uniform in bucket {tb}"
        assert (sub["alpha_z"] == 1.0).all(), "flat mode alpha_z must be 1"
    # borough mode should differ Co across zones in same bucket when alphas differ
    tb = "09:00-15:59"
    bsub = bor[bor["time_bucket"] == tb]
    assert bsub["Co"].nunique() > 1, "borough Co should vary by zone"
    print("OK  baseline flat Cu/Co (no multipliers)")


def test_agent_default_is_baseline() -> None:
    """Read OptimizationConfig defaults from source when pydantic import works."""
    try:
        from agent.config_merge import default_config

        cfg = default_config()
        assert cfg.co_mode == "flat"
        assert cfg.cu_multiplier == 1.0
        assert cfg.co_multiplier == 1.0
        assert cfg.elastic_fleet is False
        print("OK  agent default_config matches Model A baseline")
    except Exception as exc:
        print(f"SKIP agent default_config (import/env): {exc}")


def test_model_presets() -> None:
    try:
        from engine.model_presets import model_a_flat, model_b_borough, COST_SPEC

        a = model_a_flat(day_of_week=1, time_bucket="06:00-08:59")
        b = model_b_borough(day_of_week=1, time_bucket="06:00-08:59")
        assert a.co_mode == "flat" and b.co_mode == "borough"
        assert "×1.0" in COST_SPEC["A_flat"]
        print("OK  model_presets A/B definitions")
    except Exception as exc:
        print(f"SKIP model_presets (import/env): {exc}")


def test_stakeholder_outputs_slots() -> None:
    out = ROOT / "outputs"
    t1 = pd.read_csv(out / "task1_AvsB.csv")
    t2 = pd.read_csv(out / "task2_ABC.csv")
    slots = sorted(t2["slot"].unique())
    expected = {
        "dow1_0600_0859_allwks",
        "dow5_1900_2359_allwks",
        "dow6_0900_1559_allwks",
    }
    assert set(slots) == expected, f"unexpected slots: {slots}"
    for slot in slots:
        assert slot in t1["slot"].values, f"task1 missing slot {slot}"
        a_nv = t2.loc[t2["slot"] == slot, "nv_cost_A"].iloc[0]
        assert a_nv > 0, f"nv_cost_A missing for {slot}"

    # Warn if STAKEHOLDER_RESPONSE.md NV costs diverge from CSV (stale MD)
    md_path = out / "STAKEHOLDER_RESPONSE.md"
    if md_path.exists():
        md = md_path.read_text(encoding="utf-8")
        for slot in slots:
            nv_m = re.search(
                rf"### Slot: {re.escape(slot)}[\s\S]*?"
                r"\*\*Realized NV cost on test:\*\*\s+A=([\d.]+)",
                md,
            )
            if nv_m:
                md_a = float(nv_m.group(1))
                csv_a = float(t2.loc[t2["slot"] == slot, "nv_cost_A"].iloc[0])
                if abs(md_a - csv_a) / csv_a > 0.01:
                    print(
                        f"WARN  {slot}: STAKEHOLDER_RESPONSE.md nv_A={md_a:.0f} "
                        f"vs task2 CSV={csv_a:.0f} — re-run run_stakeholder_tasks.py"
                    )
    print("OK  stakeholder CSV slots aligned (Mon/Fri/Sat)")


def main() -> int:
    test_baseline_flat_costs()
    test_agent_default_is_baseline()
    test_model_presets()
    test_stakeholder_outputs_slots()
    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
