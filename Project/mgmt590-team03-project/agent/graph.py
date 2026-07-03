"""LangGraph workflow for the NYC taxi optimization agent."""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

import numpy as np
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agent.catalog import check_data_freshness, get_data_catalog
from agent.documents import DocumentContext
from agent.config_merge import default_config, merge_config
from agent.extract import classify_intent, extract_scenario
from agent.registry import append_run, load_runs
from agent.report import append_executive_summary, build_full_report, build_single_report
from agent.model_modes import elastic_summary, model_label
from agent.scenario_parse import DOW_NAMES, format_scenario_summary
from agent.schemas import IntentType, ReportTier
from agent.overlays import get_pending_overlay, list_overlay_summary
from agent.tools import (
    apply_and_validate,
    estimate_slow_job,
    persist_config_overlay,
    run_fast_pipeline,
    run_slow_bakeoff,
)
from agent.welcome import build_welcome_message
from engine.config import OptimizationConfig


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    intent: Optional[IntentType]
    config_patch: Optional[dict[str, Any]]
    extraction: Optional[dict[str, Any]]
    pending_clarification_patch: Optional[dict[str, Any]]
    current_config: Optional[dict[str, Any]]
    validation_errors: Optional[list[str]]
    run_id: Optional[str]
    run_status: Optional[str]
    report_tier: Optional[ReportTier]
    job_estimate: Optional[dict[str, Any]]
    awaiting_confirm: Optional[bool]
    awaiting_permanent_confirm: Optional[bool]
    session_id: Optional[str]
    metrics: Optional[dict[str, Any]]
    report: Optional[str]
    data_fingerprints: Optional[dict[str, str]]
    document_context: Optional[dict[str, Any]]


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _config_from_state(state: AgentState) -> OptimizationConfig:
    if state.get("current_config"):
        return OptimizationConfig.model_validate(state["current_config"])
    return default_config()


def _doc_ctx_from_state(state: AgentState) -> DocumentContext | None:
    raw = state.get("document_context")
    if not raw:
        return None
    return DocumentContext(
        combined_text=raw.get("combined_text", ""),
        images=raw.get("images") or [],
    )


def node_classify_intent(state: AgentState) -> dict[str, Any]:
    result = classify_intent(state["messages"], _doc_ctx_from_state(state))
    intent = result.intent
    if intent in ("new_data", "full_bakeoff"):
        intent = "run_whatif"
    return {"intent": intent}


def node_extract_config(state: AgentState) -> dict[str, Any]:
    pending = state.get("pending_clarification_patch")
    extraction = extract_scenario(
        state["messages"],
        _doc_ctx_from_state(state),
        pending_patch=pending,
    )
    patch = extraction.non_empty_patch()
    return {
        "extraction": _sanitize(extraction.model_dump()),
        "config_patch": _sanitize(patch) if patch else None,
    }


def node_clarify(state: AgentState) -> dict[str, Any]:
    ext = state.get("extraction") or {}
    question = ext.get("clarification_question") or "Could you clarify your scenario?"
    summary = ext.get("interpreted_summary") or format_scenario_summary(state.get("config_patch") or {})
    notes = ext.get("notes") or []
    partial = state.get("config_patch") or {}

    lines = [
        "### Need clarification",
        question,
        "",
        f"**Understood so far:** {summary}",
    ]
    if notes:
        lines.append("")
        lines.append("**Notes:**")
        for n in notes:
            lines.append(f"- {n}")
    lines.append("")
    lines.append(
        "_Reply with the missing details (e.g. `Friday, 06:00-08:59, ISO week 14, fleet 11500, run`)._"
    )

    return {
        "run_status": "awaiting_clarification",
        "pending_clarification_patch": _sanitize(partial) if partial else None,
        "messages": [AIMessage(content="\n".join(lines))],
    }


def node_validate_config(state: AgentState) -> dict[str, Any]:
    base = _config_from_state(state)
    patch = state.get("config_patch") or {}
    session_id = state.get("session_id") or "default"
    merged, ok, errors, feasibility = apply_and_validate(base, patch, session_id=session_id)
    out: dict[str, Any] = {
        "current_config": _sanitize(merged.model_dump()),
        "validation_errors": errors,
        "run_status": "validated" if ok else "validation_failed",
        "pending_clarification_patch": None,
    }
    if feasibility:
        out["feasibility"] = _sanitize(feasibility)
    if not ok:
        out["messages"] = [
            AIMessage(content="Configuration validation failed:\n- " + "\n- ".join(errors))
        ]
    return out


def _scenario_confirm_message(config: OptimizationConfig, extraction: dict | None) -> str:
    dow = DOW_NAMES.get(config.day_of_week, config.day_of_week)
    fleet_line = (
        f"F₀={config.fleet_baseline:,}, r=${config.rental_rate:.0f}, s=${config.standdown_saving:.0f} → optimizes F*"
        if config.elastic_fleet
        else f"fixed fleet {config.fleet_size:,}"
    )
    week_label = f"ISO week {config.week}" + (f", {config.year}" if config.year else "")
    lines = [
        f"**{model_label(config)}**",
        f"**Scenario:** {dow}, `{config.time_bucket}`, {week_label}",
        f"**Fleet:** {fleet_line}",
        f"**Demand model:** SAA (empirical) only",
    ]
    if extraction:
        notes = extraction.get("notes") or []
        for n in notes:
            lines.append(f"- {n}")
    return "\n".join(lines)


def node_execute_fast_path(state: AgentState) -> dict[str, Any]:
    config = OptimizationConfig.model_validate(state["current_config"])
    session_id = state.get("session_id") or "default"
    out = run_fast_pipeline(config, session_id=session_id, verbose=False)

    patch = state.get("config_patch") or {}
    if patch:
        persist_config_overlay(session_id, patch)

    pending = get_pending_overlay(session_id)
    metrics = _sanitize(out["metrics"])
    mode = metrics.get("path_mode", "refit")
    mode_note = (
        "Fast re-solve (cached models)."
        if mode == "solve_only"
        else "Full refit (first run or model/data change)."
    )

    run_id = append_run(
        {
            "path": "fast",
            "intent": state.get("intent"),
            "session_id": session_id,
            "config": _sanitize(config.model_dump()),
            "config_patch": _sanitize(patch) if patch else None,
            "has_overlay": pending["has_any"],
            "metrics_summary": {
                "runtime_s": metrics["runtime_s"],
                "fleet_used": metrics["fleet_used"],
                "primary_model": metrics["primary_model"],
                "path_mode": mode,
            },
            "ui_snapshot": {
                "metrics": metrics,
                "config": _sanitize(config.model_dump()),
                "config_patch": _sanitize(patch) if patch else None,
                "scenario": metrics.get("scenario"),
            },
        }
    )
    confirm = _scenario_confirm_message(config, state.get("extraction"))
    elastic = metrics.get("elastic")
    fleet_result = (
        elastic_summary(elastic, F0=int(config.fleet_baseline))
        if config.elastic_fleet and elastic
        else f"Fleet **{metrics['fleet_used']:,}** / {metrics['fleet_size']:,}"
    )
    result: dict[str, Any] = {
        "metrics": metrics,
        "run_id": run_id,
        "report_tier": "single",
        "run_status": "fast_path_complete",
        "awaiting_confirm": False,
        "awaiting_permanent_confirm": pending["has_any"],
        "messages": [
            AIMessage(
                content=(
                    f"{confirm}\n\n"
                    f"Run complete in **{metrics['runtime_s']}s** ({mode_note})\n"
                    f"{fleet_result}\n\n"
                    "Open the **Results & Config** tab for q* and full report."
                )
            )
        ],
    }
    if pending["has_any"]:
        result["messages"].append(
            AIMessage(
                content=(
                    "Changes saved in a **temporary overlay** only.\n\n"
                    + list_overlay_summary(session_id)
                )
            )
        )
    return result


def node_estimate_job(state: AgentState) -> dict[str, Any]:
    intent = state.get("intent")
    freshness = check_data_freshness(state.get("data_fingerprints"))
    job_type = "new_data" if intent == "new_data" else "full_bakeoff"
    estimate = estimate_slow_job(job_type, freshness.get("changed_files"))
    return {
        "job_estimate": estimate.model_dump(),
        "data_fingerprints": freshness["fingerprints"],
        "awaiting_confirm": True,
        "run_status": "awaiting_confirm",
        "report_tier": "full",
        "messages": [
            AIMessage(content=estimate.message + "\n\nClick **Confirm slow job** to proceed.")
        ],
    }


def node_execute_slow_path(state: AgentState) -> dict[str, Any]:
    out = run_slow_bakeoff()
    run_id = append_run(
        {
            "path": "slow",
            "intent": state.get("intent"),
            "metrics_summary": {
                "runtime_s": out["metrics"]["runtime_s"],
                "status": out["metrics"]["status"],
            },
        }
    )
    return {
        "metrics": out["metrics"],
        "run_id": run_id,
        "report_tier": "full",
        "run_status": "slow_path_complete",
        "awaiting_confirm": False,
    }


def node_build_report(state: AgentState) -> dict[str, Any]:
    run_id = state.get("run_id") or "pending"
    metrics = state.get("metrics") or {}
    tier = state.get("report_tier") or metrics.get("report_tier") or "single"

    if tier == "full":
        report = build_full_report(metrics, run_id)
    else:
        config = OptimizationConfig.model_validate(state["current_config"])
        report = build_single_report(config, metrics, run_id, state.get("config_patch"))

    report = append_executive_summary(report, metrics)

    reports_dir = Path(__file__).resolve().parent.parent / "agent_runs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if run_id and run_id != "pending":
        (reports_dir / f"{run_id}.md").write_text(report, encoding="utf-8")
        append_run({"run_id": run_id, "report_saved": True, "report_tier": tier})

    return {
        "report": report,
        "metrics": metrics,
        "config_patch": state.get("config_patch"),
        "current_config": state.get("current_config"),
        "run_status": "report_ready",
    }


def node_show_last_report(state: AgentState) -> dict[str, Any]:
    runs = load_runs(limit=5)
    if not runs:
        return {
            "messages": [AIMessage(content="No saved runs yet. Run a what-if scenario first.")],
            "run_status": "no_history",
        }
    last = runs[0]
    run_id = last.get("run_id", "")
    report_path = Path(__file__).resolve().parent.parent / "agent_runs" / "reports" / f"{run_id}.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else (
        f"Run `{run_id}` — report file missing. Run a fast-path scenario first."
    )
    ui = last.get("ui_snapshot") or {}
    return {
        "run_id": run_id,
        "report": report,
        "metrics": ui.get("metrics") or last.get("metrics"),
        "current_config": ui.get("config") or last.get("config"),
        "config_patch": ui.get("config_patch") or last.get("config_patch"),
        "messages": [
            AIMessage(content=f"Loaded report from run `{run_id}`. See **Results** below.")
        ],
        "run_status": "loaded_history",
    }


def node_chitchat(state: AgentState) -> dict[str, Any]:
    cfg = _config_from_state(state)
    catalog = get_data_catalog()
    ready = "ready" if catalog["primary_ready"] else "missing — place cleaned_trips_2025.parquet in project root"
    msg = build_welcome_message(cfg) + f"\n\n**Data:** `{catalog['primary_data']}` — {ready}"
    return {"messages": [AIMessage(content=msg)], "run_status": "chitchat"}


def route_after_classify(state: AgentState) -> str:
    intent = state.get("intent") or "chitchat"
    if intent == "show_last_report":
        return "show_last_report"
    return "extract_config"


def route_after_extract(state: AgentState) -> str:
    ext = state.get("extraction") or {}
    if ext.get("clarification_question"):
        return "clarify"

    intent = state.get("intent") or "chitchat"
    patch = state.get("config_patch") or {}
    run_requested = bool(ext.get("run_requested"))

    if run_requested or intent in ("run_whatif", "change_constraint") or patch:
        return "validate_config"
    return "chitchat"


def route_after_validate(state: AgentState) -> str:
    if state.get("run_status") == "validation_failed":
        return "failed"
    ext = state.get("extraction") or {}
    intent = state.get("intent") or "chitchat"
    if ext.get("run_requested") or intent in ("run_whatif", "change_constraint"):
        return "execute"
    if state.get("config_patch"):
        return "execute"
    return "apply_only"


def node_apply_config_only(state: AgentState) -> dict[str, Any]:
    cfg = _config_from_state(state)
    summary = format_scenario_summary(cfg.model_dump())
    return {
        "run_status": "config_updated",
        "messages": [
            AIMessage(
                content=(
                    f"Configuration updated: **{summary}**\n\n"
                    "Say **run allocation** or click **Run allocation** to optimize q*."
                )
            )
        ],
    }


def build_graph(*, interrupt_before_slow: bool = True):
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", node_classify_intent)
    workflow.add_node("extract_config", node_extract_config)
    workflow.add_node("clarify", node_clarify)
    workflow.add_node("validate_config", node_validate_config)
    workflow.add_node("apply_config_only", node_apply_config_only)
    workflow.add_node("execute_fast_path", node_execute_fast_path)
    workflow.add_node("estimate_job", node_estimate_job)
    workflow.add_node("execute_slow_path", node_execute_slow_path)
    workflow.add_node("build_report", node_build_report)
    workflow.add_node("show_last_report", node_show_last_report)
    workflow.add_node("chitchat", node_chitchat)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "extract_config": "extract_config",
            "show_last_report": "show_last_report",
        },
    )
    workflow.add_conditional_edges(
        "extract_config",
        route_after_extract,
        {
            "clarify": "clarify",
            "validate_config": "validate_config",
            "chitchat": "chitchat",
        },
    )
    workflow.add_conditional_edges(
        "validate_config",
        route_after_validate,
        {"failed": END, "execute": "execute_fast_path", "apply_only": "apply_config_only"},
    )
    workflow.add_edge("execute_fast_path", "build_report")
    workflow.add_edge("estimate_job", "execute_slow_path")
    workflow.add_edge("execute_slow_path", "build_report")
    workflow.add_edge("build_report", END)
    workflow.add_edge("apply_config_only", END)
    workflow.add_edge("clarify", END)
    workflow.add_edge("show_last_report", END)
    workflow.add_edge("chitchat", END)

    memory = MemorySaver()
    interrupt_nodes = ["execute_slow_path"] if interrupt_before_slow else []
    return workflow.compile(checkpointer=memory, interrupt_before=interrupt_nodes)


def build_graph_for_ui():
    """Stateless graph for Streamlit — no slow-path interrupts or stale checkpoints."""
    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", node_classify_intent)
    workflow.add_node("extract_config", node_extract_config)
    workflow.add_node("clarify", node_clarify)
    workflow.add_node("validate_config", node_validate_config)
    workflow.add_node("apply_config_only", node_apply_config_only)
    workflow.add_node("execute_fast_path", node_execute_fast_path)
    workflow.add_node("build_report", node_build_report)
    workflow.add_node("show_last_report", node_show_last_report)
    workflow.add_node("chitchat", node_chitchat)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "extract_config": "extract_config",
            "show_last_report": "show_last_report",
        },
    )
    workflow.add_conditional_edges(
        "extract_config",
        route_after_extract,
        {
            "clarify": "clarify",
            "validate_config": "validate_config",
            "chitchat": "chitchat",
        },
    )
    workflow.add_conditional_edges(
        "validate_config",
        route_after_validate,
        {"failed": END, "execute": "execute_fast_path", "apply_only": "apply_config_only"},
    )
    workflow.add_edge("execute_fast_path", "build_report")
    workflow.add_edge("build_report", END)
    workflow.add_edge("apply_config_only", END)
    workflow.add_edge("clarify", END)
    workflow.add_edge("show_last_report", END)
    workflow.add_edge("chitchat", END)

    return workflow.compile()


def run_agent_turn(
    app,
    *,
    thread_id: str,
    user_message: str,
    current_config: Optional[dict[str, Any]] = None,
    data_fingerprints: Optional[dict[str, str]] = None,
    document_context: Optional[dict[str, Any]] = None,
    pending_clarification_patch: Optional[dict[str, Any]] = None,
    resume_slow: bool = False,
) -> dict[str, Any]:
    """Invoke or resume the graph for one UI turn."""
    config = {"configurable": {"thread_id": thread_id}}

    if resume_slow:
        result = app.invoke(None, config=config)
    else:
        init: AgentState = {
            "messages": [HumanMessage(content=user_message)],
            "current_config": current_config,
            "session_id": thread_id,
            "data_fingerprints": data_fingerprints,
            "document_context": document_context,
            "pending_clarification_patch": pending_clarification_patch,
        }
        result = app.invoke(init, config=config)

    return result
