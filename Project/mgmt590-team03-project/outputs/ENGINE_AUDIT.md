# Engine & Agent Audit Report (2026-06-09)

Final review before stakeholder delivery. Covers baseline Cu/Co, slot consistency, bake-off CSV drift, and UI data sources.

---

## 1. Baseline Model A — Cu / Co (correct)

**Source:** `engine/costs.py`

| Parameter | Baseline (Model A) | Task 1 Model B | Task 2 Model C |
|-----------|------------------|----------------|----------------|
| `cu_multiplier` | **1.0** | 1.0 | 1.0 |
| `co_multiplier` | **1.0** | 1.0 | 1.0 |
| `co_mode` | **`flat`** | `borough` | `borough` (same as B) |
| `alpha_z` | **1.0 everywhere** | borough table + zone overrides | same as B |
| Fleet | fixed 13,000 | fixed 13,000 | elastic F* (r=$45, s=$20) |

In **flat** mode, `compute_costs` sets `alpha_z = 1.0` for all zones and `Co_z = Co_t` (bucket average). No borough multipliers apply. This matches the original baseline specification.

**Code paths verified:**
- `run_stakeholder_tasks.py` → Model A uses `co_mode="flat"` with explicit `cu_multiplier=1.0`, `co_multiplier=1.0` in `build_slot_inputs`.
- `agent/config_merge.default_config()` → `OptimizationConfig` defaults: `co_mode="flat"`, multipliers 1.0 (agent runs use baseline costs unless user changes config via chat).
- New reference: `engine/model_presets.py` documents A/B/C presets.

---

## 2. Deployment slots — consistent across tasks

All stakeholder outputs use the **same three slots**:

| Slot ID | Label | DOW | Bucket |
|---------|-------|-----|--------|
| `dow1_0600_0859_allwks` | Monday morning | 1 | 06:00–08:59 |
| `dow5_1900_2359_allwks` | Friday evening | 5 | 19:00–23:59 |
| `dow6_0900_1559_allwks` | Saturday midday | 6 | 09:00–15:59 |

**Presentation UI:** one slot selector drives Baseline, Task 1, and Task 2 tabs. Metrics come from `slot_metrics()` in `app/stakeholder_data.py`, which prefers **`task2_ABC.csv`** for overall NV cost (Model A) so all tabs compare the same slot.

**Validation:** `scripts/validate_engine_agent.py` confirms slot alignment in CSVs.

---

## 3. Bake-off CSV drift — reverted

### What changed
Uncommitted edits to `outputs/bakeoff_results.csv` and `outputs/bakeoff_by_scenario.csv` showed small shifts (~0.02–2% per scenario), e.g.:

- `bakeoff_results.csv` empirical `mean_nv_cost`: **369,468** (committed) → **369,556** (local re-run)
- Per-scenario NV costs moved similarly across all 35 bucket×DOW cells

### Cause
These files are produced by `run_results.py` / stakeholder Step 0 bake-off. The drift came from an **incidental re-run** after pipeline/data-loading changes (e.g. `cleaned_trips_2025.parquet` cache path, numeric sanitization). The logic was unchanged; inputs differed slightly.

### Action taken
**Reverted both files to git HEAD** (committed baseline). They remain the authoritative 35-scenario global bake-off for demand-model selection (LGBM winner).

### Note
Slot-specific **Model A overall NV cost** for presentation comes from `task2_ABC.csv`, **not** from the global bake-off mean. The baseline tab now labels the bake-off expander as “35 scenarios — not this slot.”

---

## 4. Stale markdown vs CSV (action recommended)

`STAKEHOLDER_RESPONSE.md` NV costs **do not match** current `task2_ABC.csv`:

| Slot | MD `nv_cost_A` | CSV `nv_cost_A` |
|------|----------------|-----------------|
| Mon AM | 43,728 | 53,460 |
| Fri eve | 668,937 | 652,769 |
| Sat midday | 765,208 | 700,692 |

The UI uses **CSV values** (correct for current model run). The markdown is from an earlier run.

**Recommendation:** Re-run to sync:
```bash
python run_stakeholder_tasks.py --task both
```

---

## 5. Agent Results tab vs Presentation (different baselines)

| View | Data source | Scenario |
|------|-------------|----------|
| **Presentation → Baseline** | `task2_ABC.csv` / `task1_AvsB.csv` | Selected Mon/Fri/Sat slot |
| **Agent → Results** | `bakeoff_results.csv`, `q_star_*.csv` | Global 35-scenario bake-off; q* file is Thu 09:00–15:59 wk50 |

This is intentional: Results tab shows Phase 1 engine bake-off artifacts; Presentation shows stakeholder Task 1/2 slot analysis. Do not mix NV numbers between them.

---

## 6. Validation run

```bash
python scripts/validate_engine_agent.py
```

**Passed:**
- Flat vs borough cost behavior in `engine/costs.py`
- Stakeholder CSV slot alignment

**Skipped (environment):**
- `OptimizationConfig` / agent imports fail with:
  `pydantic-core 2.41.5 incompatible with pydantic (requires 2.46.4)`
- Fix: `pip install 'pydantic-core==2.46.4'` or reinstall `pydantic` in the Anaconda env used by `./run_streamlit.sh`.

Streamlit was still running in the user session; cost logic tests do not depend on pydantic.

---

## 7. UI updates (this pass)

- **Baseline tab:** shows **Overall NV cost (test)** from selected slot, cost-parameter table (Cu×1.0, flat Co×1.0), fleet, shadow price λ.
- **Task 1 / 2 tabs:** slot label + cost spec caption; NV metrics via `slot_metrics()`.
- **Global bake-off** moved to collapsed expander with clear “not this slot” caption.

---

## 8. No issues found in

- Task 2 reusing Model B costs for Model C (correct per spec)
- Model A using its own demand samples for NV in Task 2 comparison
- Borough α only applied when `co_mode="borough"`

---

## 9. Optional follow-ups

1. Re-run stakeholder tasks to refresh `STAKEHOLDER_RESPONSE.md`.
2. Fix pydantic/pydantic-core version in conda env for full agent import tests.
3. If global bake-off should reflect latest data pipeline, re-run `run_results.py` **deliberately** and commit with a note — do not rely on accidental local runs.
