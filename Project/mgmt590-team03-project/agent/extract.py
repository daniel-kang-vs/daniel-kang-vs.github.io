"""LLM extraction with heuristic fallback when OpenAI is unavailable."""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.messages import BaseMessage

from agent.documents import DocumentContext, format_prompt_with_documents
from agent.guidance import extraction_system_prompt, load_guidance
from agent.llm import structured_completion
from agent.scenario_parse import (
    ScenarioParseResult,
    _wants_run,
    _wants_scenario,
    format_scenario_summary,
    merge_pending_clarification,
    parse_scenario_text,
)
from agent.schemas import ConfigPatch, ExtractionResult, IntentClassification


def _last_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "human" or msg.__class__.__name__ == "HumanMessage":
            return str(msg.content)
    return ""


def _heuristic_intent(text: str) -> IntentClassification:
    t = text.lower()
    if any(k in t for k in ("last report", "previous report", "show report")):
        return IntentClassification(intent="show_last_report", rationale="heuristic")
    if _wants_run(text):
        return IntentClassification(intent="run_whatif", rationale="heuristic_run")
    if _wants_scenario(text):
        return IntentClassification(intent="run_whatif", rationale="heuristic_scenario")
    if any(k in t for k in ("full bake", "bake-off", "bakeoff")):
        return IntentClassification(intent="full_bakeoff", rationale="heuristic")
    if any(k in t for k in ("new data", "upload data", "update data")):
        return IntentClassification(intent="new_data", rationale="heuristic")
    if any(
        k in t
        for k in (
            "fleet",
            "constraint",
            "floor",
            "cap",
            "weather",
            "tau",
            "change",
            "set ",
            "borough",
            "co_mode",
            "elastic",
            "task 1",
            "task 2",
            "model a",
            "model b",
            "model c",
        )
    ):
        return IntentClassification(intent="change_constraint", rationale="heuristic")
    return IntentClassification(intent="chitchat", rationale="heuristic")


def _scenario_to_extraction(scenario: ScenarioParseResult) -> ExtractionResult:
    return ExtractionResult(
        patch=ConfigPatch(**scenario.patch),
        run_requested=scenario.run_requested,
        ambiguities=scenario.ambiguities,
        clarification_question=scenario.clarification_question,
        interpreted_summary=scenario.interpreted_summary,
        notes=scenario.notes,
    )


def _merge_extraction(heuristic: ExtractionResult, llm: ExtractionResult) -> ExtractionResult:
    merged_patch = heuristic.patch.model_dump()
    for key, value in llm.patch.model_dump().items():
        if value is not None:
            merged_patch[key] = value
    ambiguities = list(dict.fromkeys(heuristic.ambiguities + llm.ambiguities))
    notes = list(dict.fromkeys(heuristic.notes + llm.notes))
    clarification = llm.clarification_question or heuristic.clarification_question
    return ExtractionResult(
        patch=ConfigPatch(**merged_patch),
        run_requested=heuristic.run_requested or llm.run_requested,
        ambiguities=ambiguities,
        clarification_question=clarification,
        interpreted_summary=format_scenario_summary(merged_patch),
        notes=notes,
    )


def _combined_input(messages: list[BaseMessage], doc_ctx: DocumentContext | None) -> str:
    user_text = _last_user_text(messages)
    return format_prompt_with_documents(user_text, doc_ctx)


def classify_intent(
    messages: list[BaseMessage],
    doc_ctx: DocumentContext | None = None,
) -> IntentClassification:
    text = _combined_input(messages, doc_ctx)
    heuristic = _heuristic_intent(text)
    if heuristic.intent != "chitchat":
        return heuristic
    if not os.getenv("OPENAI_API_KEY"):
        return heuristic

    try:
        guidance = load_guidance()
        return structured_completion(
            IntentClassification,
            "Classify NYC taxi optimization agent user intent using the phase guide below.\n\n"
            f"{guidance}\n\n"
            "Intents: change_constraint, run_whatif, new_data, full_bakeoff, show_last_report, chitchat.",
            text,
            images=doc_ctx.images if doc_ctx else None,
        )
    except Exception:
        return heuristic


def extract_scenario(
    messages: list[BaseMessage],
    doc_ctx: DocumentContext | None = None,
    *,
    pending_patch: Optional[dict] = None,
) -> ExtractionResult:
    """Parse user text into config patch + clarification state."""
    text = _combined_input(messages, doc_ctx)

    if pending_patch:
        scenario = merge_pending_clarification(pending_patch, text)
    else:
        scenario = parse_scenario_text(text)

    heuristic = _scenario_to_extraction(scenario)

    if not os.getenv("OPENAI_API_KEY"):
        return heuristic

    try:
        llm_result = structured_completion(
            ExtractionResult,
            extraction_system_prompt(),
            text,
            images=doc_ctx.images if doc_ctx else None,
        )
        return _merge_extraction(heuristic, llm_result)
    except Exception:
        return heuristic


def extract_config_patch(
    messages: list[BaseMessage],
    doc_ctx: DocumentContext | None = None,
    *,
    pending_patch: Optional[dict] = None,
) -> ConfigPatch:
    """Backward-compatible: return patch only."""
    return extract_scenario(messages, doc_ctx, pending_patch=pending_patch).patch
