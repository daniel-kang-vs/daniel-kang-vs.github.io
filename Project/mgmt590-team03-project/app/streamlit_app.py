#!/usr/bin/env python3
"""Streamlit UI — chat agent + config + results."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")
if not os.getenv("OPENAI_API_KEY"):
    load_dotenv(ROOT / ".env.example")

from app.reload_engine import reload_engine_modules

_STALE_ENGINE = reload_engine_modules()

from app.baseline_results import load_baseline_snapshot
from app.engine_check import check_engine_import
from app.stakeholder_ui import render_stakeholder_presentation
from agent.catalog import check_data_freshness, get_data_catalog
from agent.documents import (
    DocumentContext,
    build_document_context,
    parse_upload_bytes,
)
from agent.config_merge import default_config, merge_config
from agent.data_source import build_agg_from_upload
from agent.graph import build_graph_for_ui, run_agent_turn
from agent.overlays import (
    get_pending_overlay,
    list_overlay_summary,
    load_session_config,
    save_temp_upload,
    session_dir,
    set_session_agg_path,
)
from agent.registry import append_run
from agent.model_cache import GLOBAL_MODEL_CACHE_ID, get_cached_models
from agent.report import append_executive_summary, build_single_report
from agent.model_modes import elastic_summary, model_label, normalize_task_config
from agent.tools import invalidate_pipeline_cache, run_fast_pipeline
from agent.welcome import build_welcome_message, DOW_NAMES
from engine.config import OptimizationConfig

st.set_page_config(page_title="NYC Taxi Optimization Agent", layout="wide")

ENGINE_VERSION = "2026-06-10-bakeoff-empty-fix"
AGENT_BUSY_TIMEOUT_S = 180
ENGINE_ERROR = _STALE_ENGINE or check_engine_import()
PYTHON_LABEL = Path(sys.executable).name
USING_ANACONDA = "anaconda" in sys.executable.lower() or "miniconda" in sys.executable.lower()


@st.cache_resource
def get_graph(_engine_version: str):
    return build_graph_for_ui()


def _preview_scenario_caption(
    dow: int,
    bucket: str,
    week: int,
    fleet: int,
    co_mode: str,
    elastic: bool,
    rental: float,
    standdown: float,
    year: int | None = None,
) -> str:
    """Live preview from widget values (updates before Run)."""
    dow_name = DOW_NAMES.get(dow, dow)
    week_label = f"week {week}" + (f", {year}" if year else "")
    if elastic:
        return (
            f"**Preview:** {dow_name}, `{bucket}`, {week_label} · "
            f"**Model C (Task 2)** · borough Co · F₀={fleet:,} → optimizes **F*** "
            f"(r=${rental:.0f}, s=${standdown:.0f}) · SAA"
        )
    if co_mode == "borough":
        return (
            f"**Preview:** {dow_name}, `{bucket}`, {week_label} · "
            f"**Model B (Task 1)** · borough Co · fixed fleet {fleet:,} · SAA"
        )
    return (
        f"**Preview:** {dow_name}, `{bucket}`, {week_label} · "
        f"**Model A (baseline)** · flat Co · fixed fleet {fleet:,} · SAA"
    )


def _models_cached() -> bool:
    try:
        cfg = OptimizationConfig.model_validate(st.session_state.current_config)
        _, ok = get_cached_models(st.session_state.thread_id, cfg)
        if ok:
            return True
        _, ok = get_cached_models(GLOBAL_MODEL_CACHE_ID, cfg)
        return ok
    except Exception:
        return False


_YEAR_UNSET = object()


def run_allocation(
    fleet_size: int,
    *,
    day_of_week: int | None = None,
    time_bucket: str | None = None,
    week: int | None = None,
    year: int | None = _YEAR_UNSET,
    co_mode: str | None = None,
    elastic_fleet: bool | None = None,
    fleet_baseline: int | None = None,
    rental_rate: float | None = None,
    standdown_saving: float | None = None,
    user_visible: bool = True,
) -> None:
    """Fast path: SAA (empirical) only — apply scenario + constraints, then re-solve."""
    if ENGINE_ERROR:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Engine unavailable:\n\n{ENGINE_ERROR}"}
        )
        return

    patch: dict = {"fleet_size": int(fleet_size)}
    if day_of_week is not None:
        patch["day_of_week"] = int(day_of_week)
    if time_bucket is not None:
        patch["time_bucket"] = time_bucket
    if week is not None:
        patch["week"] = int(week)
    if co_mode is not None:
        patch["co_mode"] = co_mode
    if elastic_fleet is not None:
        patch["elastic_fleet"] = bool(elastic_fleet)
    if fleet_baseline is not None:
        patch["fleet_baseline"] = int(fleet_baseline)
    if rental_rate is not None:
        patch["rental_rate"] = float(rental_rate)
    if standdown_saving is not None:
        patch["standdown_saving"] = float(standdown_saving)
    patch = normalize_task_config(patch)

    merged = merge_config(
        OptimizationConfig.model_validate(st.session_state.current_config),
        patch,
    )
    if year is not _YEAR_UNSET:
        merged.year = year
    st.session_state.current_config = merged.model_dump()
    week_label = f"week {merged.week}" + (f", {merged.year}" if merged.year else "")
    if user_visible:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": (
                    f"Run allocation — {model_label(merged)} · "
                    f"{DOW_NAMES.get(merged.day_of_week)}, {merged.time_bucket}, {week_label}"
                ),
            }
        )

    st.session_state.agent_busy = True
    st.session_state.agent_busy_since = time.time()
    try:
        out = run_fast_pipeline(merged, session_id=st.session_state.thread_id, verbose=False)
        metrics = out["metrics"]
        mode = metrics.get("path_mode", "refit")
        run_id = append_run(
            {
                "path": "fast",
                "intent": "run_whatif",
                "session_id": st.session_state.thread_id,
                "config": merged.model_dump(),
                "config_patch": patch,
                "metrics_summary": {
                    "runtime_s": metrics["runtime_s"],
                    "fleet_used": metrics["fleet_used"],
                    "path_mode": mode,
                },
                "ui_snapshot": {"metrics": metrics, "config": merged.model_dump()},
            }
        )
        report = append_executive_summary(
            build_single_report(merged, metrics, run_id, patch),
            metrics,
        )
        st.session_state.new_metrics = metrics
        st.session_state.new_config = merged.model_dump()
        st.session_state.run_id = run_id
        st.session_state.report = report
        st.session_state.run_status = "report_ready"
        note = "re-solve (~15 s)" if mode == "solve_only" else "SAA fit cached — later runs are fast"
        elastic = metrics.get("elastic")
        if merged.elastic_fleet and elastic:
            fleet_block = elastic_summary(elastic, F0=int(merged.fleet_baseline))
        else:
            fleet_block = f"Fixed fleet **{metrics['fleet_used']:,}** / {metrics['fleet_size']:,}"
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"**{model_label(merged)}** · SAA (empirical)\n\n"
                    f"Done in **{metrics['runtime_s']}s** ({note})\n"
                    f"{fleet_block}\n\n"
                    "See **Results & Config** tab."
                ),
            }
        )
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Run failed:\n\n```\n{exc}\n```"}
        )
        st.session_state.run_status = "error"
    finally:
        st.session_state.agent_busy = False
        st.session_state.pop("agent_busy_since", None)


def _clear_stale_agent_busy() -> bool:
    """Drop stuck busy flag if a prior run exceeded the timeout (e.g. browser refresh mid-solve)."""
    if not st.session_state.get("agent_busy"):
        return False
    started = st.session_state.get("agent_busy_since")
    if started is None or (time.time() - float(started)) > AGENT_BUSY_TIMEOUT_S:
        st.session_state.agent_busy = False
        st.session_state.pop("agent_busy_since", None)
        return True
    return False


def _models_cached_for_session() -> bool:
    return _models_cached()


def init_session():
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "messages": [],
        "current_config": default_config().model_dump(),
        "report": "",
        "run_id": None,
        "run_status": None,
        "new_metrics": None,
        "new_config": None,
        "data_fingerprints": check_data_freshness().get("fingerprints", {}),
        "agent_busy": False,
        "welcome_shown": False,
        "pending_clarification_patch": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.welcome_shown:
        cfg = OptimizationConfig.model_validate(st.session_state.current_config)
        st.session_state.messages.append(
            {"role": "assistant", "content": build_welcome_message(cfg)}
        )
        st.session_state.welcome_shown = True


def _primary_mean_nv_cost(metrics: dict) -> str:
    pm = metrics.get("pipeline_metrics") or {}
    if pm.get("nv_cost") is not None:
        return f"{float(pm['nv_cost']):,.0f}"
    primary = metrics.get("primary_model")
    bakeoff = metrics.get("bakeoff") or []
    for row in bakeoff:
        if row.get("model") == primary and row.get("mean_nv_cost") is not None:
            return f"{float(row['mean_nv_cost']):,.0f}"
    if bakeoff:
        top = sorted(bakeoff, key=lambda r: r.get("rank", 999))[0]
        if top.get("mean_nv_cost") is not None:
            return f"{float(top['mean_nv_cost']):,.0f}"
    return "—"


def _nv_cost_source_help(metrics: dict) -> str:
    source = (metrics.get("pipeline_metrics") or {}).get("demand_source")
    labels = {
        "actual": "Realized demand for this exact week/year.",
        "saa_mean": "No exact-week data — using SAA sample mean as demand.",
    }
    if source is None:
        return ""
    return labels.get(source, f"Realized demand ({source}).")


def _scenario_label(config: dict | None, metrics: dict | None = None) -> str:
    if metrics and metrics.get("scenario"):
        sc = metrics["scenario"]
        dow = DOW_NAMES.get(sc.get("day_of_week"), sc.get("day_of_week"))
        return f"{dow}, {sc.get('time_bucket')}, week {sc.get('week')}"
    if config:
        dow = DOW_NAMES.get(config.get("day_of_week"), config.get("day_of_week"))
        return f"{dow}, {config.get('time_bucket')}, week {config.get('week')}"
    return "—"


def apply_graph_result(result: dict):
    st.session_state.run_status = result.get("run_status")
    if result.get("current_config"):
        st.session_state.current_config = result["current_config"]
    if result.get("report"):
        st.session_state.report = result["report"]
    if result.get("run_id"):
        st.session_state.run_id = result["run_id"]
    if result.get("data_fingerprints"):
        st.session_state.data_fingerprints = result["data_fingerprints"]

    if result.get("run_status") == "awaiting_clarification":
        st.session_state.pending_clarification_patch = result.get("pending_clarification_patch")
    elif result.get("run_status") in ("report_ready", "fast_path_complete", "config_updated"):
        st.session_state.pending_clarification_patch = None

    if result.get("metrics") and result.get("run_status") in (
        "fast_path_complete",
        "report_ready",
        "loaded_history",
    ):
        st.session_state.new_metrics = result["metrics"]
        st.session_state.new_config = result.get("current_config") or st.session_state.current_config

    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            st.session_state.messages.append({"role": "assistant", "content": str(msg.content)})


def _format_user_message(text: str, attachment_names: list[str]) -> str:
    if not attachment_names:
        return text
    files = ", ".join(f"`{n}`" for n in attachment_names)
    if text.strip():
        return f"{text.strip()}\n\n📎 Attachments: {files}"
    return f"📎 Attachments: {files}"


def _process_attachments(uploaded_files) -> tuple[DocumentContext | None, list[str]]:
    if not uploaded_files:
        return None, []

    notes: list[str] = []
    parsed = []
    for f in uploaded_files:
        doc = parse_upload_bytes(f.getbuffer(), f.name)
        parsed.append(doc)
        if doc.kind == "trip_data":
            suffix = Path(f.name).suffix or ".parquet"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.getbuffer())
                tmp_path = Path(tmp.name)
            try:
                handle_data_upload_from_path(tmp_path, f.name)
                notes.append(f"Applied trip data `{f.name}` as temp overlay.")
            except Exception as exc:
                notes.append(f"Trip data `{f.name}` failed: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)
        elif doc.error:
            notes.append(f"`{f.name}`: {doc.error}")

    ctx = build_document_context(parsed)
    if ctx.combined_text or ctx.images:
        return ctx, notes
    return None, notes


def handle_data_upload_from_path(tmp_path: Path, original_name: str) -> None:
    sid = st.session_state.thread_id
    save_temp_upload(sid, tmp_path, original_name)
    agg_dest = session_dir(sid) / "agg_overlay.parquet"
    build_agg_from_upload(tmp_path, agg_dest)
    set_session_agg_path(sid, agg_dest)
    invalidate_pipeline_cache(sid)
    st.session_state.current_config = load_session_config(sid).model_dump()


def invoke_user_message(text: str, attachments=None):
    attachment_names = [f.name for f in (attachments or [])]
    display = _format_user_message(text, attachment_names)
    st.session_state.messages.append({"role": "user", "content": display})

    doc_ctx, attach_notes = _process_attachments(attachments)
    if attach_notes:
        st.session_state.messages.append(
            {"role": "assistant", "content": "\n".join(f"- {n}" for n in attach_notes)}
        )

    if ENGINE_ERROR:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Engine unavailable:\n\n{ENGINE_ERROR}"}
        )
        return

    st.session_state.agent_busy = True
    st.session_state.agent_busy_since = time.time()
    try:
        app = get_graph(ENGINE_VERSION)
        result = run_agent_turn(
            app,
            thread_id=st.session_state.thread_id,
            user_message=text,
            current_config=st.session_state.current_config,
            data_fingerprints=st.session_state.data_fingerprints,
            document_context=doc_ctx.to_state() if doc_ctx else None,
            pending_clarification_patch=st.session_state.get("pending_clarification_patch"),
        )
        apply_graph_result(result)
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Error during run:\n\n```\n{exc}\n```"}
        )
        st.session_state.run_status = "error"
    finally:
        st.session_state.agent_busy = False
        st.session_state.pop("agent_busy_since", None)


def handle_data_upload(uploaded_file) -> None:
    sid = st.session_state.thread_id
    suffix = Path(uploaded_file.name).suffix or ".parquet"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        handle_data_upload_from_path(tmp_path, uploaded_file.name)
    except Exception as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Upload failed:\n\n```\n{exc}\n```"}
        )
        return
    finally:
        tmp_path.unlink(missing_ok=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                f"Uploaded `{uploaded_file.name}` to temporary overlay. "
                "Next run will refit models.\n\n"
                + list_overlay_summary(sid)
            ),
        }
    )


def render_results_comparison():
    st.subheader("Results")
    left, right = st.columns(2)

    with left:
        snapshot = load_baseline_snapshot()
        if not snapshot["ready"]:
            st.markdown("**Baseline** (precomputed `outputs/`)")
            st.info("No baseline CSVs found.")
        else:
            sc = snapshot.get("scenario") or {}
            fleet_used = snapshot.get("fleet_used")
            dow = DOW_NAMES.get(sc.get("day_of_week"), sc.get("day_of_week", "—"))
            tb = sc.get("time_bucket", "—")
            week = sc.get("week", "—")
            fleet_label = f"{fleet_used:,}" if fleet_used is not None else "—"
            st.markdown(f"**Baseline** — fleet {fleet_label}, {dow}, {tb}, week {week}")
            best = snapshot["best_model"]
            if best:
                c1, c2, c3 = st.columns(3)
                c1.metric("Fleet used", fleet_label)
                c2.metric("Mean NV cost", f"{best['mean_nv_cost']:,.0f}")
                c3.metric("Fill rate", f"{best.get('mean_fill_rate', 0):.1%}")
            q_star = snapshot["q_star"]
            if q_star is not None and not q_star.empty:
                st.dataframe(q_star.head(15), use_container_width=True, hide_index=True)

    with right:
        metrics = st.session_state.new_metrics
        cfg = st.session_state.new_config or {}
        if not metrics:
            st.markdown("**Latest run**")
            st.info("Run allocation from **Agent Chat** to see results here.")
        else:
            label = metrics.get("model_label") or "Latest run"
            st.markdown(f"**{label}** — {_scenario_label(cfg, metrics)}")
            elastic = metrics.get("elastic")
            is_elastic = bool(metrics.get("elastic_fleet") or cfg.get("elastic_fleet"))
            if is_elastic and elastic:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("F₀ (baseline)", f"{int(elastic.get('F0', cfg.get('fleet_baseline', 0))):,}")
                c2.metric("F* (optimal)", f"{float(elastic.get('F_star', 0)):,.0f}")
                direction = str(elastic.get("direction", "—"))
                c3.metric("Direction", direction)
                c4.metric("λ(F₀)", f"{float(elastic.get('breakeven_rental', 0)):.2f}")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("NV cost", f"{float(elastic.get('nv_cost', 0)):,.0f}")
                c6.metric("Fleet adj", f"{float(elastic.get('fleet_adjustment_cost', 0)):,.0f}")
                c7.metric("Total cost", f"{float(elastic.get('total_cost', 0)):,.0f}")
                c8.metric("Runtime", f"{metrics.get('runtime_s')}s")
                if direction == "stay":
                    r, s = elastic.get("rental_rate"), elastic.get("standdown_saving")
                    lam = elastic.get("breakeven_rental")
                    if r is not None and s is not None and lam is not None:
                        st.caption(
                            f"F* = F₀ because λ(F₀)={float(lam):.2f} is in the dead zone "
                            f"[${float(s):.0f}, ${float(r):.0f}] — fleet does not expand or contract."
                        )
            elif is_elastic and not elastic:
                st.warning(
                    "Task 2 (elastic fleet) was requested but F* metrics are missing. "
                    "Restart Streamlit and run again."
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fleet (fallback)", f"{metrics.get('fleet_used', 0):,}")
                c2.metric("Demand model", "SAA")
                c3.metric("NV cost", _primary_mean_nv_cost(metrics), help=_nv_cost_source_help(metrics))
                c4.metric("Runtime", f"{metrics.get('runtime_s')}s")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Fleet (fixed)", f"{metrics.get('fleet_used', 0):,}")
                c2.metric("Demand model", "SAA")
                c3.metric("NV cost", _primary_mean_nv_cost(metrics), help=_nv_cost_source_help(metrics))
                c4.metric("Runtime", f"{metrics.get('runtime_s')}s")
            rows = metrics.get("q_star_table") or metrics.get("top_zones") or []
            if rows:
                df = pd.DataFrame(rows).sort_values("q_star", ascending=False)
                st.dataframe(df.head(15), use_container_width=True, hide_index=True)


def main():
    init_session()
    stale_busy = _clear_stale_agent_busy()

    st.title("NYC Yellow Taxi Fleet Optimization")
    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.caption(
        f"OpenAI: {'configured' if api_ok else 'heuristic mode'} | "
        f"Python: `{PYTHON_LABEL}` | session: `{st.session_state.thread_id[:8]}`"
    )

    if ENGINE_ERROR:
        st.error(ENGINE_ERROR)
    elif not USING_ANACONDA:
        st.warning("Use `./run_streamlit.sh` (Anaconda) for LightGBM support.")

    busy = st.session_state.get("agent_busy", False)
    if stale_busy:
        st.warning(
            "Cleared a stuck optimization flag from a prior run. "
            "Click **Run allocation** again (or use Reset below)."
        )
    models_ready = _models_cached_for_session()
    cfg = OptimizationConfig.model_validate(st.session_state.current_config)

    tab_presentation, tab_agent, tab_results = st.tabs(
        ["Presentation", "Agent Chat", "Results & Config"]
    )

    with tab_presentation:
        render_stakeholder_presentation()

    with tab_results:
        render_results_comparison()
        if st.session_state.report:
            st.markdown("### Full report")
            st.markdown(st.session_state.report)
            st.download_button(
                "Download report (.md)",
                st.session_state.report,
                file_name=f"run_{st.session_state.run_id or 'latest'}.md",
                mime="text/markdown",
            )
        with st.expander("Current config", expanded=False):
            st.json(cfg.model_dump())

    with tab_agent:
        st.subheader("Agent Chat")
        st.caption(
            "Agent runs use **SAA (empirical)** demand only. Set scenario + Task 1/2 options below or in chat "
            "(DOW, bucket, week, co_mode, elastic fleet, floors, weather)."
        )

        cfg = OptimizationConfig.model_validate(st.session_state.current_config)
        sc1, sc2, sc3, sc4, sc5 = st.columns([2, 2, 1, 1, 1])
        with sc1:
            sel_dow = st.selectbox(
                "Day of week",
                options=list(range(1, 8)),
                format_func=lambda d: f"{d} — {DOW_NAMES[d]}",
                index=int(cfg.day_of_week) - 1,
                disabled=busy,
                key="scenario_dow",
            )
        with sc2:
            buckets = [
                "00:00-05:59",
                "06:00-08:59",
                "09:00-15:59",
                "16:00-18:59",
                "19:00-23:59",
            ]
            sel_bucket = st.selectbox(
                "Time bucket",
                options=buckets,
                index=buckets.index(cfg.time_bucket) if cfg.time_bucket in buckets else 2,
                disabled=busy,
                key="scenario_bucket",
            )
        with sc3:
            sel_week = st.number_input(
                "ISO week",
                min_value=1,
                max_value=52,
                value=int(cfg.week),
                disabled=busy,
                key="scenario_week",
            )
        with sc4:
            fleet_size = st.number_input(
                "Fleet / F₀",
                min_value=1000,
                max_value=20000,
                value=int(cfg.fleet_baseline if cfg.elastic_fleet else cfg.fleet_size),
                step=500,
                disabled=busy,
                key="scenario_fleet",
            )
        with sc5:
            year_options = ["Any", 2024, 2025, 2026]
            year_index = year_options.index(cfg.year) if cfg.year in year_options else 0
            sel_year_choice = st.selectbox(
                "Year",
                options=year_options,
                index=year_index,
                disabled=busy,
                key="scenario_year",
                help="Year of actual demand used to evaluate NV cost. "
                "'Any' averages across years with data for this week. "
                "SAA allocation (q*) itself pools all years regardless.",
            )
            sel_year = None if sel_year_choice == "Any" else int(sel_year_choice)

        m1, m2, m3 = st.columns([2, 1, 1])
        with m2:
            sel_elastic = st.checkbox(
                "Elastic fleet (Task 2 / Model C)",
                value=bool(cfg.elastic_fleet),
                disabled=busy,
                key="scenario_elastic",
            )
        with m1:
            if sel_elastic:
                st.selectbox(
                    "Overage cost",
                    options=["borough"],
                    format_func=lambda _: "Model C — borough Co_z (Task 2 requires Model B costs)",
                    disabled=True,
                    key="scenario_co_mode_elastic",
                )
                sel_co_mode = "borough"
            else:
                sel_co_mode = st.selectbox(
                    "Overage cost (Task 1)",
                    options=["flat", "borough"],
                    format_func=lambda x: "Model A — flat Co" if x == "flat" else "Model B — borough Co_z",
                    index=0 if cfg.co_mode == "flat" else 1,
                    disabled=busy,
                    key="scenario_co_mode",
                )
        with m3:
            if sel_elastic:
                sel_rental = st.number_input("r ($)", min_value=0.0, value=float(cfg.rental_rate), step=5.0, key="scenario_r")
                sel_standdown = st.number_input("s ($)", min_value=0.0, value=float(cfg.standdown_saving), step=5.0, key="scenario_s")
            else:
                sel_rental = float(cfg.rental_rate)
                sel_standdown = float(cfg.standdown_saving)

        st.caption(
            _preview_scenario_caption(
                int(sel_dow), sel_bucket, int(sel_week), int(fleet_size),
                sel_co_mode, bool(sel_elastic), float(sel_rental), float(sel_standdown),
                year=sel_year,
            )
        )

        if st.session_state.get("pending_clarification_patch"):
            st.warning(
                "Waiting for clarification — your next message will be merged with the partial scenario above."
            )

        f1, f2 = st.columns([3, 1])
        with f2:
            run_clicked = st.button(
                "Run allocation",
                type="primary",
                disabled=bool(ENGINE_ERROR) or busy,
                use_container_width=True,
            )

        if run_clicked:
            scenario_patch: dict = {
                "day_of_week": int(sel_dow),
                "time_bucket": sel_bucket,
                "week": int(sel_week),
                "fleet_size": int(fleet_size),
                "co_mode": "borough" if sel_elastic else sel_co_mode,
                "elastic_fleet": bool(sel_elastic),
                "rental_rate": float(sel_rental),
                "standdown_saving": float(sel_standdown),
            }
            if sel_elastic:
                scenario_patch["fleet_baseline"] = int(fleet_size)
            scenario_patch = normalize_task_config(scenario_patch)
            merged_preview = merge_config(cfg, scenario_patch)
            merged_preview.year = sel_year
            st.session_state.current_config = merged_preview.model_dump()
            with st.spinner(
                "Re-solving SAA (~15 s)…"
                if _models_cached()
                else "Fitting SAA empirical allocator (~30–60 s)…"
            ):
                run_allocation(
                    int(fleet_size),
                    day_of_week=int(sel_dow),
                    time_bucket=sel_bucket,
                    week=int(sel_week),
                    year=sel_year,
                    co_mode=sel_co_mode,
                    elastic_fleet=bool(sel_elastic),
                    fleet_baseline=int(fleet_size) if sel_elastic else None,
                    rental_rate=float(sel_rental) if sel_elastic else None,
                    standdown_saving=float(sel_standdown) if sel_elastic else None,
                )
            st.rerun()

        chat_box = st.container(height=400)
        with chat_box:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

        chat_attachments = st.file_uploader(
            "Attachments (optional)",
            type=["pdf", "docx", "doc", "txt", "md", "png", "jpg", "jpeg", "webp", "parquet", "csv"],
            accept_multiple_files=True,
            key="chat_attachments",
            label_visibility="collapsed",
        )
        with st.form("chat_form", clear_on_submit=True):
            c1, c2 = st.columns([11, 1])
            with c1:
                user_input = st.text_input(
                    "Message",
                    placeholder="e.g. q* for Friday dow 5, 06:00-08:59, ISO week 14, fleet 11500, run",
                    disabled=bool(ENGINE_ERROR) or busy,
                    label_visibility="collapsed",
                )
            with c2:
                submitted = st.form_submit_button("Send", disabled=bool(ENGINE_ERROR) or busy)

        if submitted and (user_input or chat_attachments):
            message = user_input.strip() if user_input else "Apply the attached document(s)."
            invoke_user_message(message, attachments=chat_attachments)
            st.rerun()

        if busy:
            note = "Re-solving SAA (~15 s)…" if models_ready else "Fitting SAA (~30–60 s)…"
            st.info(f"Running optimization. {note}")
            if st.button("Reset stuck run", key="reset_agent_busy"):
                st.session_state.agent_busy = False
                st.session_state.pop("agent_busy_since", None)
                st.rerun()
        elif models_ready:
            st.caption("SAA cached — change scenario/constraints and **Run allocation** (~15 s).")
        else:
            st.caption("First **Run allocation** fits SAA empirical demand (~30–60 s); later runs are faster.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear chat", disabled=busy):
                st.session_state.thread_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.welcome_shown = False
                st.session_state.new_metrics = None
                st.session_state.report = ""
                st.session_state.run_id = None
                st.session_state.pending_clarification_patch = None
                init_session()
                st.rerun()
        with c2:
            if st.button("Show last report", disabled=busy):
                invoke_user_message("show last report")
                st.rerun()


if __name__ == "__main__":
    main()
