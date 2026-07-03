"""Load phased agent guidance for LLM system prompts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

GUIDANCE_DIR = Path(__file__).resolve().parent / "guidance"


@lru_cache(maxsize=4)
def load_guidance(name: str = "AGENT_PHASES.md") -> str:
    path = GUIDANCE_DIR / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def extraction_system_prompt() -> str:
    phases = load_guidance()
    return (
        "You extract NYC taxi fleet optimization config changes from user messages.\n\n"
        "## Agent phases (follow strictly)\n\n"
        f"{phases}\n\n"
        "Return JSON with config fields, run_requested, ambiguities, and clarification_question."
    )
