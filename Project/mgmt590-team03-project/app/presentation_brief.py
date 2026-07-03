"""Stakeholder modification brief — presentation deliverables and comparison framework.

Source: NYC_taxi_stakeholder_modification.pdf (MGMT 590-037, Summer 2026).
Engine implementation is handled separately; this module drives the Streamlit UI only.
"""

from __future__ import annotations

from typing import Any

# ── Base model (pre-stakeholder-change) ───────────────────────────────────────

BASE_MODEL = {
    "id": "A",
    "label": "Model A — Flat overage cost (baseline)",
    "description": (
        "Current model: zone-uniform overage cost within each time bucket "
        "(Co_z = Co_t for all zones). Fixed fleet F = 13,000."
    ),
    "co": "Co_z = Co_t (flat bucket average)",
    "fleet": "Fixed at 13,000",
}

# ── Task 1 models ─────────────────────────────────────────────────────────────

TASK1_MODELS = [
    BASE_MODEL,
    {
        "id": "B",
        "label": "Model B — Zone-specific overage cost",
        "description": "Co_z = α_b(z) · Co_t with borough multipliers; Cu_z unchanged.",
        "co": "Co_z = α_b(z) · Co_t",
        "fleet": "Fixed at 13,000",
    },
]

# ── Task 2 models (extends Task 1) ──────────────────────────────────────────

TASK2_MODELS = [
    {
        "id": "A",
        "label": "Model A — Fixed fleet, flat overage",
        "description": "Task 1 Model A baseline.",
        "co": "Flat Co_t",
        "fleet": "Fixed F = 13,000",
    },
    {
        "id": "B",
        "label": "Model B — Fixed fleet, zone-specific overage",
        "description": "Task 1 Model B.",
        "co": "Zone-specific Co_z",
        "fleet": "Fixed F = 13,000",
    },
    {
        "id": "C",
        "label": "Model C — Elastic fleet, zone-specific overage",
        "description": (
            "F optimized jointly with q. Fleet adjustment cost: "
            "cost(F) = r·(F−F₀)⁺ − s·(F₀−F)⁺ with F₀=13,000."
        ),
        "co": "Zone-specific Co_z",
        "fleet": "F optimized jointly with q",
    },
]

FLEET_ELASTICITY = {
    "F0": 13_000,
    "rental_rate_r": 45.0,
    "standdown_saving_s": 20.0,
    "formula": "cost(F) = r·(F − F₀)⁺ − s·(F₀ − F)⁺",
}

# ── Borough overage multipliers (Task 1) ─────────────────────────────────────

BOROUGH_CO_MULTIPLIERS: list[dict[str, Any]] = [
    {"borough": "Manhattan", "zones": "—", "alpha": 1.00, "rationale": "Dense grid; short repositioning"},
    {"borough": "Brooklyn", "zones": "—", "alpha": 1.25, "rationale": "Semi-dense; moderate repositioning"},
    {"borough": "Queens (non-airport)", "zones": "—", "alpha": 1.40, "rationale": "Dispersed; higher repositioning"},
    {"borough": "Bronx", "zones": "—", "alpha": 1.50, "rationale": "Lower density; longer gaps"},
    {"borough": "Staten Island", "zones": "—", "alpha": 1.80, "rationale": "Isolated; very high repositioning"},
    {"borough": "Airport override", "zones": "132 (JFK), 138 (LGA)", "alpha": 0.70, "rationale": "High-throughput; rapid re-engagement"},
]

# ── Three required deployment slots ─────────────────────────────────────────

DEPLOYMENT_SLOTS: list[dict[str, Any]] = [
    {
        "id": "mon_morning",
        "name": "Monday morning peak",
        "day_of_week": 1,
        "time_bucket": "06:00-08:59",
        "week": 50,
        "notes": "Commute-period morning slot (adjust week to a held-out test week).",
    },
    {
        "id": "fri_evening",
        "name": "Friday evening",
        "day_of_week": 5,
        "time_bucket": "16:00-18:59",
        "week": 50,
        "notes": "Evening rush; alternative: 19:00-23:59 if your team defines evening as night.",
    },
    {
        "id": "sat_midday",
        "name": "Saturday midday",
        "day_of_week": 6,
        "time_bucket": "09:00-15:59",
        "week": 50,
        "notes": "Weekend midday leisure + errands demand profile.",
    },
]

# ── Metrics required per slot ─────────────────────────────────────────────────

TASK1_METRICS = [
    "q* summarized by borough totals",
    "Realized newsvendor cost on held-out test weeks",
    "τ_z = Cu_z / (Cu_z + Co_z) for the 5 highest-shift zones (A vs B)",
    "Solver cross-check under Model B (SLSQP, water-filling, cvxpy)",
]

TASK2_METRICS = [
    "Optimal fleet F* and direction of adjustment from 13,000 baseline",
    "Shadow price on Σ q_z ≤ F (marginal value of one additional cab)",
    "Total cost (newsvendor + fleet adjustment) vs Model B",
    "Break-even rental rate r (TLC indifferent between renting vs newsvendor loss)",
]

# ── Plain-language questions (presentation Q&A) ─────────────────────────────

TASK1_QUESTIONS = [
    "Which boroughs **gain** fleet and which **lose** it under Model B? Does the direction match economic intuition?",
    "How do **airport zones** (JFK 132, LGA 138) behave? Does lower overage cost cause more or fewer cabs, and why?",
    "Does Model B produce a **lower realized newsvendor cost** than Model A on the test set, and by how much? If not, why was the flat approximation adequate?",
    "Confirm the agent absorbs new cost parameters **without code restructuring** — what precisely changed in the pipeline?",
]

TASK2_QUESTIONS = [
    "Does the agent **expand or contract** the fleet, and does the answer vary by slot? What drives the difference?",
    "How does the **r ≠ s asymmetry** ($45 rental vs $20 stand-down) affect decisions around the 13,000 baseline?",
    "At optimal F*, is the fleet capacity constraint **binding**? What does the shadow price tell TLC about cab inventory value?",
    "How much of total cost reduction (A → C) comes from **better spatial allocation** (Task 1) vs **fleet right-sizing** (Task 2)?",
]

DELIVERABLES = [
    "**Task 1:** Updated cost pipeline (zone-specific Co_z); (A) vs (B) comparison table for 3 slots; solver cross-check under Model B.",
    "**Task 2:** Extended solver with F as decision variable; (A)/(B)/(C) comparison with fleet sizes, costs, shadow prices, break-even r.",
    "**Written explanation** covering all plain-language questions in Task 1 §1.3 and Task 2 §2.3.",
]

# ── Agent prompts (for quick-run once engine supports each model) ─────────────

def slot_config(slot: dict[str, Any]) -> dict[str, Any]:
    return {
        "day_of_week": slot["day_of_week"],
        "time_bucket": slot["time_bucket"],
        "week": slot["week"],
        "fleet_size": FLEET_ELASTICITY["F0"],
    }


def agent_prompt_task1(slot: dict[str, Any], model_id: str) -> str:
    label = "flat overage (Model A)" if model_id == "A" else "zone-specific borough overage (Model B)"
    return (
        f"Run allocation for {slot['name']}: "
        f"day_of_week={slot['day_of_week']}, time_bucket={slot['time_bucket']}, week={slot['week']}, "
        f"fleet=13000, using {label}. "
        "Report borough-level q* totals, realized NV cost, top τ shifts, and solver cross-check."
    )


def agent_prompt_task2(slot: dict[str, Any], model_id: str) -> str:
    if model_id == "A":
        mode = "fixed fleet 13000, flat overage (Task 2 Model A)"
    elif model_id == "B":
        mode = "fixed fleet 13000, zone-specific overage (Task 2 Model B)"
    else:
        mode = (
            "elastic fleet with zone-specific overage (Task 2 Model C): "
            f"optimize F jointly with q, F0=13000, r=$45, s=$20"
        )
    return (
        f"Run allocation for {slot['name']}: "
        f"day_of_week={slot['day_of_week']}, time_bucket={slot['time_bucket']}, week={slot['week']}, "
        f"{mode}. Report F*, shadow price, total cost vs B, and break-even rental rate."
    )


def empty_task1_table() -> list[dict[str, Any]]:
    rows = []
    for slot in DEPLOYMENT_SLOTS:
        for model in TASK1_MODELS:
            rows.append(
                {
                    "Slot": slot["name"],
                    "Model": model["id"],
                    "Borough q* totals": "—",
                    "Realized NV cost": "—",
                    "Top 5 τ shifts": "—",
                    "Solver agreement": "—",
                }
            )
    return rows


def empty_task2_table() -> list[dict[str, Any]]:
    rows = []
    for slot in DEPLOYMENT_SLOTS:
        for model in TASK2_MODELS:
            rows.append(
                {
                    "Slot": slot["name"],
                    "Model": model["id"],
                    "F*": "—" if model["id"] != "C" else "—",
                    "Δ from 13k": "—",
                    "Shadow price": "—",
                    "Total cost": "—",
                    "Break-even r": "—" if model["id"] == "C" else "n/a",
                }
            )
    return rows
