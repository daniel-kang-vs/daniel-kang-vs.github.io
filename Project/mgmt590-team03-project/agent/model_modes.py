"""Stakeholder model variants A / B / C for agent runs."""

from __future__ import annotations

from engine.config import OptimizationConfig


def model_variant(config: OptimizationConfig) -> str:
    """Return A_flat, B_borough, or C_elastic."""
    if config.elastic_fleet:
        return "C_elastic"
    if config.co_mode == "borough":
        return "B_borough"
    return "A_flat"


def model_label(config: OptimizationConfig) -> str:
    labels = {
        "A_flat": "Model A — flat Co, fixed fleet",
        "B_borough": "Model B — borough Co, fixed fleet (Task 1)",
        "C_elastic": "Model C — borough Co, elastic F* (Task 2)",
    }
    return labels[model_variant(config)]


def normalize_task_config(patch: dict) -> dict:
    """Task 2 (Model C) uses borough Co like stakeholder run_task2."""
    out = dict(patch)
    if out.get("elastic_fleet"):
        out["co_mode"] = "borough"
        f0 = out.get("fleet_baseline") or out.get("fleet_size")
        if f0 is not None:
            out["fleet_baseline"] = int(f0)
            out["fleet_size"] = int(f0)
    return out


def elastic_summary(elastic: dict | None, *, F0: int) -> str:
    if not elastic:
        return f"F* unavailable (F₀={F0:,})"
    F_star = float(elastic.get("F_star", F0))
    direction = str(elastic.get("direction") or elastic.get("status", "")).replace("optimal:", "")
    lam = elastic.get("breakeven_rental") or elastic.get("shadow_price")
    adj = elastic.get("fleet_adjustment_cost", 0)
    nv = elastic.get("nv_cost")
    total = elastic.get("total_cost")
    r = elastic.get("rental_rate")
    s = elastic.get("standdown_saving")

    if direction == "stay" or abs(F_star - F0) < 0.5:
        head = f"**F* = {F_star:,.0f}** (stay at F₀={F0:,})"
    elif direction == "expand":
        head = f"**F* = {F_star:,.0f}** (expand from F₀={F0:,})"
    elif direction == "contract":
        head = f"**F* = {F_star:,.0f}** (contract from F₀={F0:,})"
    else:
        head = f"**F* = {F_star:,.0f}** ({direction or 'optimal'}) · F₀={F0:,}"

    lines = [head]
    if lam is not None:
        lines.append(f"λ(F₀) = {float(lam):.2f}")
    if nv is not None:
        lines.append(f"NV = {float(nv):,.0f}")
    if adj is not None and float(adj) != 0:
        lines.append(f"fleet adj = {float(adj):+,.0f}")
    if total is not None:
        lines.append(f"**total = {float(total):,.0f}**")
    if direction == "stay" and lam is not None and r is not None and s is not None:
        lines.append(f"_Dead zone: ${float(s):.0f} ≤ λ(F₀) ≤ ${float(r):.0f}_")
    return " · ".join(lines)
