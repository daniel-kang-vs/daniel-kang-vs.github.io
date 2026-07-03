"""Streamlit rendering for stakeholder presentation deliverables."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import streamlit as st

from agent.welcome import DOW_NAMES
from app.baseline_results import load_baseline_snapshot
from app.presentation_brief import (
    BOROUGH_CO_MULTIPLIERS,
    DELIVERABLES,
    DEPLOYMENT_SLOTS,
    FLEET_ELASTICITY,
    TASK1_METRICS,
    TASK1_MODELS,
    TASK1_QUESTIONS,
    TASK2_METRICS,
    TASK2_MODELS,
    TASK2_QUESTIONS,
    agent_prompt_task1,
    agent_prompt_task2,
    empty_task1_table,
    empty_task2_table,
    slot_config,
)
def _init_presentation_state() -> None:
    defaults = {
        "presentation_notes": {q: "" for q in TASK1_QUESTIONS + TASK2_QUESTIONS},
        "presentation_task1_results": empty_task1_table(),
        "presentation_task2_results": empty_task2_table(),
        "presentation_explanation": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _slot_label(slot: dict[str, Any]) -> str:
    dow = DOW_NAMES.get(slot["day_of_week"], slot["day_of_week"])
    return f"{slot['name']} ({dow}, `{slot['time_bucket']}`, wk {slot['week']})"


def _render_baseline_callout() -> None:
    snapshot = load_baseline_snapshot()
    st.markdown("### Base model reference (Model A)")
    st.caption(
        "Pre-stakeholder baseline: flat overage Co_z = Co_t, fixed fleet 13,000. "
        "Compare all Task 1 / Task 2 runs against this."
    )
    if not snapshot["ready"]:
        st.warning("No precomputed baseline in `outputs/`. Run `run_results.py` for Model A reference numbers.")
        return

    sc = snapshot.get("scenario") or {}
    best = snapshot.get("best_model") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baseline fleet", f"{snapshot.get('fleet_used', 13000):,}")
    c2.metric("Scenario", _slot_label(sc) if sc else "see outputs/")
    c3.metric("Mean NV cost", f"{best.get('mean_nv_cost', 0):,.0f}" if best else "—")
    c4.metric("Best model", best.get("model", "—"))
    if snapshot.get("q_star_file"):
        st.caption(f"Source: `outputs/{snapshot['q_star_file']}`, `outputs/bakeoff_results.csv`")


def _render_slot_cards(
    *,
    apply_config_patch: Callable[[dict], None],
    invoke_user_message: Callable[[str], None],
    engine_error: bool,
    busy: bool,
    task: int,
) -> None:
    st.markdown("### Deployment slots (run each model on all three)")
    for slot in DEPLOYMENT_SLOTS:
        with st.expander(_slot_label(slot), expanded=False):
            st.caption(slot["notes"])
            cfg = slot_config(slot)
            st.json(cfg)

            if task == 1:
                models = TASK1_MODELS
                prompt_fn = agent_prompt_task1
            else:
                models = TASK2_MODELS
                prompt_fn = agent_prompt_task2

            cols = st.columns(len(models))
            for col, model in zip(cols, models):
                with col:
                    btn_label = f"Run {model['id']}"
                    if st.button(
                        btn_label,
                        key=f"t{task}_{slot['id']}_{model['id']}",
                        disabled=bool(engine_error) or busy,
                        help=model["description"],
                    ):
                        apply_config_patch(cfg)
                        invoke_user_message(prompt_fn(slot, model["id"]))
                        st.rerun()


def render_presentation_tab(
    *,
    apply_config_patch: Callable[[dict], None],
    invoke_user_message: Callable[[str], None],
    engine_error: bool,
    busy: bool,
) -> None:
    """Main presentation deliverables tab."""
    _init_presentation_state()

    st.markdown("## Stakeholder Modification — Presentation Deliverables")
    st.caption(
        "MGMT 590-037 · Two tasks, three deployment slots, compare against Model A baseline. "
        "Engine updates are in progress — use this tab to track metrics, questions, and run scenarios via the agent."
    )

    # Deliverables checklist
    with st.expander("Deliverables checklist", expanded=True):
        for i, item in enumerate(DELIVERABLES, 1):
            st.markdown(f"{i}. {item}")

    _render_baseline_callout()

    tab_t1, tab_t2, tab_qa, tab_notes = st.tabs(
        ["Task 1 — Zone overage", "Task 2 — Elastic fleet", "Presentation Q&A", "Team notes"]
    )

    with tab_t1:
        st.markdown("### Task 1 — Zone-specific overage costs")
        st.markdown(
            "Replace flat Co_z = Co_t with **Co_z = α_b(z) · Co_t**. "
            "Cu_z unchanged. Fleet, floors, caps, and demand models unchanged."
        )

        st.markdown("#### Borough multipliers α")
        st.dataframe(
            pd.DataFrame(BOROUGH_CO_MULTIPLIERS),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Required metrics (per slot × model)")
        for m in TASK1_METRICS:
            st.markdown(f"- {m}")

        st.markdown("#### Comparison table — Model A vs B")
        t1_df = pd.DataFrame(st.session_state.presentation_task1_results)
        edited_t1 = st.data_editor(
            t1_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="task1_editor",
        )
        if st.button("Save Task 1 table edits"):
            st.session_state.presentation_task1_results = edited_t1.to_dict("records")
            st.success("Task 1 comparison table saved to session.")

        if st.session_state.get("new_metrics"):
            st.info(
                "Latest agent run available — copy metrics from **Results** tab into the table above, "
                "or ask the agent: *Record last run as Task 1 Model B for Saturday midday* (once engine tags runs)."
            )

        st.markdown("#### Quick-run via agent")
        _render_slot_cards(
            apply_config_patch=apply_config_patch,
            invoke_user_message=invoke_user_message,
            engine_error=engine_error,
            busy=busy,
            task=1,
        )

    with tab_t2:
        st.markdown("### Task 2 — Fleet size as a decision variable")
        st.markdown(
            "Jointly optimize **F** and **q** with zone-specific overage (Task 1 Model B). "
            "Fleet adjustment cost uses asymmetric rental vs stand-down rates."
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline F₀", f"{FLEET_ELASTICITY['F0']:,}")
        c2.metric("Rental rate r", f"${FLEET_ELASTICITY['rental_rate_r']:.0f}/cab/slot")
        c3.metric("Stand-down saving s", f"${FLEET_ELASTICITY['standdown_saving_s']:.0f}/cab/slot")
        st.caption(f"Formula: `{FLEET_ELASTICITY['formula']}` — note **r ≠ s** asymmetry.")

        st.markdown("#### Models to compare")
        for model in TASK2_MODELS:
            st.markdown(f"- **{model['label']}**: {model['description']}")

        st.markdown("#### Required metrics (per slot × model)")
        for m in TASK2_METRICS:
            st.markdown(f"- {m}")

        st.markdown("#### Comparison table — Models A / B / C")
        t2_df = pd.DataFrame(st.session_state.presentation_task2_results)
        edited_t2 = st.data_editor(
            t2_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="task2_editor",
        )
        if st.button("Save Task 2 table edits"):
            st.session_state.presentation_task2_results = edited_t2.to_dict("records")
            st.success("Task 2 comparison table saved to session.")

        st.markdown("#### Cost decomposition (A → C)")
        st.markdown(
            "| Component | Source | What to report |\n"
            "|-----------|--------|----------------|\n"
            "| Spatial allocation | Task 1 (A→B) | NV cost change from zone-specific Co |\n"
            "| Fleet right-sizing | Task 2 (B→C) | Fleet adjustment + binding shadow price |\n"
            "| **Total** | A→C | Combined newsvendor + fleet cost |"
        )

        st.markdown("#### Quick-run via agent")
        _render_slot_cards(
            apply_config_patch=apply_config_patch,
            invoke_user_message=invoke_user_message,
            engine_error=engine_error,
            busy=busy,
            task=2,
        )

    with tab_qa:
        st.markdown("### Questions stakeholders expect you to answer")
        st.caption("Prepare plain-language answers for the live presentation.")

        st.markdown("#### Task 1 questions (§1.3)")
        for i, q in enumerate(TASK1_QUESTIONS, 1):
            with st.expander(f"Q{i}. {q[:60]}…" if len(q) > 60 else f"Q{i}. {q}", expanded=i == 1):
                st.markdown(q)
                st.text_area(
                    "Your answer",
                    value=st.session_state.presentation_notes.get(q, ""),
                    key=f"ans_t1_{i}",
                    height=120,
                )

        st.markdown("#### Task 2 questions (§2.3)")
        for i, q in enumerate(TASK2_QUESTIONS, 1):
            with st.expander(f"Q{i}. {q[:60]}…" if len(q) > 60 else f"Q{i}. {q}", expanded=False):
                st.markdown(q)
                st.text_area(
                    "Your answer",
                    value=st.session_state.presentation_notes.get(q, ""),
                    key=f"ans_t2_{i}",
                    height=120,
                )

        if st.button("Save all Q&A drafts"):
            notes: dict[str, str] = {}
            for i, q in enumerate(TASK1_QUESTIONS, 1):
                notes[q] = st.session_state.get(f"ans_t1_{i}", "")
            for i, q in enumerate(TASK2_QUESTIONS, 1):
                notes[q] = st.session_state.get(f"ans_t2_{i}", "")
            st.session_state.presentation_notes = notes
            st.success("Q&A drafts saved to session.")

    with tab_notes:
        st.markdown("### Written explanation (Deliverable 3)")
        st.caption("Synthesize Task 1 + Task 2 findings for the final write-up.")
        st.session_state.presentation_explanation = st.text_area(
            "Draft explanation",
            value=st.session_state.get("presentation_explanation", ""),
            height=400,
            placeholder=(
                "Cover: borough fleet shifts under Model B, airport behavior, NV cost delta A vs B, "
                "pipeline configurability, fleet expand/contract by slot, r≠s asymmetry, shadow price "
                "interpretation, and cost decomposition A→C."
            ),
        )
        if st.session_state.presentation_explanation:
            st.download_button(
                "Download explanation draft (.md)",
                st.session_state.presentation_explanation,
                file_name="stakeholder_explanation_draft.md",
            )
