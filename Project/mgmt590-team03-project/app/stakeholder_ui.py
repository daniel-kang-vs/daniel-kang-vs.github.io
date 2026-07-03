"""Streamlit presentation UI for stakeholder Tasks 1 & 2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from app.stakeholder_data import (
    BOROUGH_MULTIPLIERS,
    TASK1_QUESTIONS,
    TASK2_QUESTIONS,
    TIME_BUCKETS,
    borough_q_for_slot,
    load_stakeholder_bundle,
    slot_label,
    slot_metrics,
)
from engine.model_presets import COST_SPEC

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _fmt_num(v, decimals=0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if decimals == 0:
        return f"{float(v):,.0f}"
    return f"{float(v):,.{decimals}f}"


def _slot_select_label(slot_id: str, bundle: dict) -> str:
    label = slot_label(slot_id)
    if slot_id not in bundle.get("slots_with_data", set()):
        return f"{label} — no data yet"
    return label


def _render_common_header(bundle: dict) -> None:
    st.markdown("### Shared context (all models)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline fleet F₀", "13,000")
    c2.metric("Demand model (bake-off winner)", bundle["winner"])
    n_with_data = len(set(bundle["slots"]) & bundle.get("slots_with_data", set()))
    c3.metric("Deployment slots", f"{len(bundle['slots'])} ({n_with_data} populated)")

    with st.expander("Model definitions", expanded=True):
        st.markdown(
            "| Model | Overage cost | Fleet |\n"
            "|-------|--------------|-------|\n"
            "| **A — Baseline** | Flat Co_z = Co_t | Fixed 13,000 |\n"
            "| **B — Task 1** | Co_z = α_b(z) · Co_t | Fixed 13,000 |\n"
            "| **C — Task 2** | Zone-specific (same as B) | F optimized with r=$45, s=$20 |"
        )

    with st.expander("Borough overage multipliers α (Task 1 & 2)", expanded=False):
        st.dataframe(
            pd.DataFrame(BOROUGH_MULTIPLIERS, columns=["Borough / zone", "α"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Higher α in outer boroughs → higher idle repositioning cost → fewer cabs under Model B. "
            "Airport override (0.70) → lower idle cost → more staging at JFK/LGA."
        )

    if bundle["bakeoff_selection"] is not None:
        with st.expander("Demand model bake-off (SAA / GLM / LGBM)", expanded=False):
            df = bundle["bakeoff_selection"]
            show = df[["model", "mean_nv_cost", "mean_fill_rate", "rank"]].copy()
            show["mean_nv_cost"] = show["mean_nv_cost"].map(lambda x: f"{x:,.0f}")
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption("Source: `outputs/bakeoff_model_selection.csv`")


def _render_baseline_tab(bundle: dict, slot_id: str) -> None:
    st.markdown("## Baseline — Model A (original)")
    sm = slot_metrics(bundle, slot_id)
    st.info(
        f"**Slot (shared across Tasks 1 & 2):** {sm['slot_label']}  \n"
        f"**Cost spec:** {COST_SPEC['A_flat']}"
    )

    parsed = bundle["task1_parsed"].get(slot_id, {})
    t2 = None
    if bundle["task2_csv"] is not None:
        sub = bundle["task2_csv"][bundle["task2_csv"]["slot"] == slot_id]
        t2 = sub.iloc[0] if not sub.empty else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall NV cost (test)", _fmt_num(sm.get("nv_cost_A")))
    m2.metric("Total cost (A)", _fmt_num(sm.get("total_cost_A")))
    m3.metric("Shadow price λ(F₀)", _fmt_num(parsed.get("shadow_price_A"), 4))
    m4.metric("Fleet", "13,000 (fixed)")

    st.markdown(
        "#### Cost parameters (Model A)\n"
        "| Parameter | Baseline setting |\n"
        "|-----------|------------------|\n"
        "| Cu_z (underage) | Zone×bucket avg yellow fare, **×1.0** |\n"
        "| Co_z (overage) | Bucket avg fare, **flat** (same all zones, **×1.0**) |\n"
        "| τ_z | Cu_z / (Cu_z + Co_z) — varies by zone fare only |\n"
        "| Fleet | Fixed **13,000** |"
    )

    bor = borough_q_for_slot(bundle["task1_csv"], slot_id)
    if bor is not None and "q_A" in bor.columns:
        st.markdown("#### Borough allocation q* (Model A)")
        show = bor[["borough", "q_A"]].copy()
        show["q_A"] = show["q_A"].round(1)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.bar_chart(show.set_index("borough"))

    if bundle["bakeoff_original"] is not None:
        with st.expander("Global bake-off (35 bucket×DOW scenarios — not this slot)", expanded=False):
            st.caption(
                "Mean NV across all 5 buckets × 7 days on test set. "
                "Use **Overall NV cost** above for this deployment slot."
            )
            show = bundle["bakeoff_original"][["model", "mean_nv_cost", "mean_fill_rate", "rank"]].copy()
            show["mean_nv_cost"] = show["mean_nv_cost"].map(lambda x: f"{x:,.0f}")
            st.dataframe(show, use_container_width=True, hide_index=True)


def _render_task1_tab(bundle: dict, slot_id: str) -> None:
    st.markdown("## Task 1 — Zone-specific overage (A vs B)")
    sm = slot_metrics(bundle, slot_id)
    st.caption(
        f"Slot: **{sm['slot_label']}** · Model B: {COST_SPEC['B_borough']}"
    )

    parsed = bundle["task1_parsed"].get(slot_id, {})

    m1, m2, m3 = st.columns(3)
    m1.metric("NV cost A (flat)", _fmt_num(sm.get("nv_cost_A") or parsed.get("nv_cost_A")))
    m2.metric("NV cost B (borough)", _fmt_num(sm.get("nv_cost_B") or parsed.get("nv_cost_B")))
    delta = parsed.get("nv_delta")
    if delta is None and sm.get("nv_cost_A") and sm.get("nv_cost_B"):
        delta = sm["nv_cost_B"] - sm["nv_cost_A"]
    m3.metric("Δ B − A", _fmt_num(delta), delta_color="inverse" if delta and delta < 0 else "normal")

    bor = parsed.get("borough_table")
    if bor is None:
        bor = borough_q_for_slot(bundle["task1_csv"], slot_id)
        if bor is not None and "q_A" in bor.columns:
            bor = bor.rename(columns={"q_A": "q_A", "q_B": "q_B", "delta_q": "delta_q"})
    if bor is not None and not bor.empty:
        st.markdown("#### Borough q* totals (A vs B)")
        st.dataframe(bor.round(1), use_container_width=True, hide_index=True)
        if "q_A" in bor.columns and "q_B" in bor.columns:
            chart = bor.set_index("borough")[["q_A", "q_B"]]
            st.bar_chart(chart)

    tau = parsed.get("tau_table")
    if tau is not None and not tau.empty:
        st.markdown("#### Top-5 τ-shift zones (A → B)")
        st.caption("τ_z = Cu_z / (Cu_z + Co_z) — largest shifts drive allocation changes.")
        st.dataframe(tau, use_container_width=True, hide_index=True)

    sp_a = parsed.get("shadow_price_A")
    sp_b = parsed.get("shadow_price_B")
    if sp_a is not None:
        st.markdown(
            f"#### Shadow price λ(F₀=13,000)\n"
            f"- Model A (flat): **{sp_a:.4f}**\n"
            f"- Model B (borough): **{sp_b:.4f}**"
        )

    solver = parsed.get("solver_check")
    if solver is not None and not solver.empty:
        st.markdown("#### 3-way solver cross-check (Model B)")
        st.dataframe(solver, use_container_width=True, hide_index=True)

    with st.expander("Presentation Q&A — Task 1", expanded=False):
        for i, q in enumerate(TASK1_QUESTIONS, 1):
            st.markdown(f"**Q{i}.** {q}")


def _render_task2_tab(bundle: dict, slot_id: str) -> None:
    st.markdown("## Task 2 — Elastic fleet (A / B / C)")
    sm = slot_metrics(bundle, slot_id)
    st.caption(
        f"Slot: **{sm['slot_label']}** · Model C: {COST_SPEC['C_elastic']}"
    )

    parsed = bundle["task2_parsed"].get(slot_id, {})
    t2 = None
    if bundle["task2_csv"] is not None:
        sub = bundle["task2_csv"][bundle["task2_csv"]["slot"] == slot_id]
        t2 = sub.iloc[0] if not sub.empty else None

    if t2 is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F*", _fmt_num(t2["F_star"]))
        c2.metric("Direction", str(t2["direction"]).replace("optimal:", ""))
        c3.metric("Shadow price λ(F*)", _fmt_num(t2["shadow_price"], 4))
        c4.metric("Break-even rental", _fmt_num(t2["breakeven_rental"], 4))

        st.markdown("#### Total cost comparison")
        cost_df = pd.DataFrame(
            {
                "Model": ["A (flat, fixed)", "B (borough, fixed)", "C (borough, elastic)"],
                "NV cost": [t2["nv_cost_A"], t2["nv_cost_B"], t2["nv_cost_C"]],
                "Fleet adj": [0, 0, t2["fleet_adj_cost_C"]],
                "Total cost": [t2["total_cost_A"], t2["total_cost_B"], t2["total_cost_C"]],
            }
        )
        cost_df["NV cost"] = cost_df["NV cost"].map(lambda x: f"{x:,.0f}")
        cost_df["Fleet adj"] = cost_df["Fleet adj"].map(lambda x: f"{x:,.0f}")
        cost_df["Total cost"] = cost_df["Total cost"].map(lambda x: f"{x:,.0f}")
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

        ab = parsed.get("decomp_AB", t2["total_cost_B"] - t2["total_cost_A"])
        bc = parsed.get("decomp_BC", t2["total_cost_C"] - t2["total_cost_B"])
        st.markdown("#### A → C cost decomposition")
        d1, d2, d3 = st.columns(3)
        d1.metric("A → B (spatial)", f"{ab:+,.0f}")
        d2.metric("B → C (fleet sizing)", f"{bc:+,.0f}")
        d3.metric("A → C (total)", f"{t2['total_cost_C'] - t2['total_cost_A']:+,.0f}")

    metrics_tbl = parsed.get("metrics_table")
    if metrics_tbl is not None and not metrics_tbl.empty:
        with st.expander("Full metrics table (from report)", expanded=False):
            st.dataframe(metrics_tbl, use_container_width=True, hide_index=True)

    st.info(
        "**Dead zone [20, 45]:** when s ≤ λ(F₀) ≤ r, fleet stays at F₀. "
        "Expand if λ > r; contract if λ < s."
    )

    with st.expander("Presentation Q&A — Task 2", expanded=False):
        for i, q in enumerate(TASK2_QUESTIONS, 1):
            st.markdown(f"**Q{i}.** {q}")


def _run_stakeholder_script(
    slots_str: str,
    task: str,
    *,
    demand_model: str = "auto",
    fleet: int = 13000,
    year: int = 2026,
    week: int | None = None,
    rental_rate: float = 45.0,
    standdown_saving: float = 20.0,
) -> str:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_stakeholder_tasks.py"),
        "--task", task,
        "--slots", slots_str,
        "--demand-model", demand_model,
        "--fleet", str(fleet),
        "--year", str(year),
        "--rental-rate", str(rental_rate),
        "--standdown-saving", str(standdown_saving),
    ]
    if week is not None:
        cmd += ["--week", str(week)]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=3600)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return f"Exit {proc.returncode}\n\n{out[-8000:]}"
    return out[-8000:]


def _scenario_params(key_prefix: str) -> dict:
    """Render demand-model / fleet / year / week / elastic-fleet inputs and return as kwargs."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        demand_model = st.selectbox(
            "Demand model", ["auto", "saa", "glm", "lgbm"], key=f"{key_prefix}_dm",
            help="auto = bake-off winner",
        )
    with c2:
        fleet = st.number_input(
            "Fleet size", min_value=1000, max_value=50000, value=13000, step=500,
            key=f"{key_prefix}_fleet",
        )
    with c3:
        year = st.number_input(
            "Year", min_value=2024, max_value=2026, value=2026, key=f"{key_prefix}_year",
        )
    with c4:
        week = st.number_input(
            "ISO week (0 = none)", min_value=0, max_value=52, value=0,
            key=f"{key_prefix}_week", help="Used by GLM/LGBM; SAA ignores it",
        )
    with st.expander("Elastic fleet params (Model C)", expanded=False):
        c5, c6 = st.columns(2)
        with c5:
            rental_rate = st.number_input(
                "Rental rate ($)", min_value=0.0, value=45.0, step=1.0, key=f"{key_prefix}_rental",
            )
        with c6:
            standdown_saving = st.number_input(
                "Standdown saving ($)", min_value=0.0, value=20.0, step=1.0, key=f"{key_prefix}_standdown",
            )
    return dict(
        demand_model=demand_model,
        fleet=int(fleet),
        year=int(year),
        week=None if week == 0 else int(week),
        rental_rate=rental_rate,
        standdown_saving=standdown_saving,
    )


def render_stakeholder_presentation() -> None:
    st.markdown("## Stakeholder Presentation")
    st.caption(
        "Results from `run_stakeholder_tasks.py` → `outputs/STAKEHOLDER_RESPONSE.md`, "
        "`task1_AvsB.csv`, `task2_ABC.csv`"
    )

    bundle = load_stakeholder_bundle()

    if not bundle["ready"]:
        st.warning(
            "Stakeholder outputs not found. Run:\n\n"
            "```\npython run_stakeholder_tasks.py --task both\n```"
        )
        with st.expander("Run stakeholder tasks from UI", expanded=True):
            st.markdown("**Configure slots** (format: `dow:time_bucket`, comma-separated)")
            default_slots = "1:06:00-08:59,5:19:00-23:59,6:09:00-15:59"
            slots_input = st.text_input("Slots", value=default_slots)
            task_choice = st.selectbox("Task", ["both", "1", "2"])
            params = _scenario_params("init")
            if st.button("Run `run_stakeholder_tasks.py`", type="primary"):
                with st.spinner("Running stakeholder tasks (may take several minutes)…"):
                    log = _run_stakeholder_script(slots_input, task_choice, **params)
                st.code(log)
                st.rerun()
        return

    # Slot selector — always show three canonical slots; extras from CSV appended
    slot_options = {_slot_select_label(sid, bundle): sid for sid in bundle["slots"]}
    selected_label = st.selectbox("Deployment slot", list(slot_options.keys()))
    slot_id = slot_options[selected_label]

    if slot_id not in bundle.get("slots_with_data", set()):
        st.warning(
            f"No saved results for **{slot_label(slot_id)}**. "
            "Expand **Re-run stakeholder tasks** at the bottom and click "
            "**Run all default slots** (or run "
            "`python run_stakeholder_tasks.py --task both` from the project root)."
        )

    _render_common_header(bundle)

    tab_baseline, tab_task1, tab_task2, tab_full = st.tabs(
        ["① Baseline (Model A)", "② Task 1 (A vs B)", "③ Task 2 (A/B/C)", "Full report"]
    )

    with tab_baseline:
        st.caption(f"Slot: **{selected_label}**")
        _render_baseline_tab(bundle, slot_id)

    with tab_task1:
        st.caption(f"Slot: **{selected_label}**")
        _render_task1_tab(bundle, slot_id)

    with tab_task2:
        st.caption(f"Slot: **{selected_label}**")
        _render_task2_tab(bundle, slot_id)

    with tab_full:
        st.markdown(bundle["md_text"])
        st.download_button(
            "Download STAKEHOLDER_RESPONSE.md",
            bundle["md_text"],
            file_name="STAKEHOLDER_RESPONSE.md",
            mime="text/markdown",
        )

    st.divider()
    with st.expander("Re-run stakeholder tasks (updates outputs)", expanded=False):
        st.caption(
            "Updates `task1_AvsB.csv` / `task2_ABC.csv` / `STAKEHOLDER_RESPONSE.md` for the slot(s) "
            "in this run only — other slots' results are preserved. The fleet/year/week/demand model "
            "below apply to whichever slot(s) you run. Note: a custom `--week`/`--year` produces a "
            "slot id like `dow1_0600_0859_wk16yr2026` (separate from the `_allwks` default slots)."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            dow = st.selectbox("Day of week", range(1, 8), format_func=lambda d: f"DOW {d}")
        with c2:
            bucket = st.selectbox("Time bucket", TIME_BUCKETS)
        with c3:
            task_run = st.selectbox("Task to run", ["both", "1", "2"], key="rerun_task")
        params = _scenario_params("rerun")
        slots_custom = st.text_input(
            "Or custom slots string",
            value="1:06:00-08:59,5:19:00-23:59,6:09:00-15:59",
            key="custom_slots",
        )
        single_slot = f"{dow}:{bucket}"
        # run_stakeholder_tasks.py now merges results per-slot into the output
        # files instead of overwriting, so a single-slot run only updates that
        # slot and leaves the others untouched.
        if st.button(f"Run single slot ({single_slot})"):
            with st.spinner("Running…"):
                log = _run_stakeholder_script(single_slot, task_run, **params)
            st.code(log)
            st.rerun()
        if st.button("Run all default slots"):
            with st.spinner("Running all slots…"):
                log = _run_stakeholder_script(slots_custom, task_run, **params)
            st.code(log)
            st.rerun()
