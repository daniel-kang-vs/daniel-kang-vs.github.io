"""Startup welcome text — objective function and tunable constraints."""

from __future__ import annotations

from engine.config import OptimizationConfig

from agent.model_cache import REFIT_CONFIG_KEYS, SOLVE_ONLY_KEYS

DOW_NAMES = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}


def _fmt_config(config: OptimizationConfig) -> str:
    dow = DOW_NAMES.get(config.day_of_week, str(config.day_of_week))
    return (
        f"- **Scenario**: {dow}, `{config.time_bucket}`, week {config.week}\n"
        f"- **Fleet size**: {config.fleet_size:,}\n"
        f"- **Bounds**: floor_alpha={config.floor_alpha}, cap_multiplier={config.cap_multiplier}\n"
        f"- **Solver**: {config.solver}"
    )


def build_welcome_message(config: OptimizationConfig | None = None) -> str:
    cfg = config or OptimizationConfig()
    solve_list = ", ".join(f"`{k}`" for k in SOLVE_ONLY_KEYS)
    refit_list = ", ".join(f"`{k}`" for k in REFIT_CONFIG_KEYS)

    return (
        "Welcome to the **NYC Yellow Taxi Fleet Optimization Agent**.\n\n"
        "### Objective (per time bucket)\n"
        "Minimize expected **newsvendor cost** across 263 zones:\n"
        "```\n"
        "min Σ_z [ Co_z × E[(q_z − D_z)⁺] + Cu_z × E[(D_z − q_z)⁺] ]\n"
        "s.t.  Σ_z q_z ≤ fleet_size\n"
        "      floor_z ≤ q_z ≤ cap_z\n"
        "```\n"
        "- **Cu** = underage (lost revenue when demand exceeds supply)\n"
        "- **Co** = overage (idle fleet cost)\n"
        "- **q_z** = taxis allocated to zone *z*; **D_z** = stochastic demand\n\n"
        "### Current baseline config\n"
        f"{_fmt_config(cfg)}\n\n"
        "### Agent demand model\n"
        "Chat and **Run allocation** use **SAA (empirical)** only — no GLM/LGBM/DFL.\n\n"
        "### Fast updates (re-solve only, ~15 s)\n"
        f"Change these without refitting SAA: {solve_list}, `co_mode`, `elastic_fleet`.\n\n"
        "**Task 1:** `co_mode=flat` (Model A) or `borough` (Model B zone-specific Co).\n"
        "**Task 2:** `elastic_fleet=true` with F₀, rental r, standdown s.\n\n"
        "**Examples (chat):**\n"
        "- `q* for Friday dow 5, 06:00-08:59, ISO week 14, fleet 11500, run`\n"
        "- `Apr 2026 week 2, Friday morning, fleet 13000 — run allocation`\n"
        "- `Monday morning week 50, fleet 13000`\n"
        "- `Set floor_alpha to 0.2 and run`\n\n"
        "### Full refit required (~2–3 min first run in session)\n"
        f"Changing these retrains demand/cost models: {refit_list}, or uploading new trip data.\n\n"
        "### Multimodal input\n"
        "Attach **PDF, Word, images, or text files** in chat alongside your message. "
        "The agent extracts constraints and scenarios from documents — not just typed text.\n\n"
        "### Stakeholder presentation (Task 1 & 2)\n"
        "Open the **Presentation Deliverables** tab for:\n"
        "- Task 1: flat vs zone-specific overage (Models A vs B)\n"
        "- Task 2: elastic fleet optimization (Models A / B / C)\n"
        "- Three slots: Monday morning, Friday evening, Saturday midday\n"
        "- Comparison tables, Q&A prep, and quick-run buttons\n\n"
        "Type a constraint change, attach a memo, or use **Quick actions** below."
    )
