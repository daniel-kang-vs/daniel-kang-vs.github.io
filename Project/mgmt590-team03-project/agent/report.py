"""Report builder — Python template first, optional LLM executive summary."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from engine.config import OptimizationConfig

from agent.model_modes import elastic_summary


def _format_config(config: OptimizationConfig) -> str:
    d = config.model_dump()
    keep = [
        "time_bucket",
        "day_of_week",
        "week",
        "fleet_size",
        "floor_alpha",
        "cap_multiplier",
        "co_mode",
        "elastic_fleet",
        "fleet_baseline",
        "rental_rate",
        "standdown_saving",
        "proxy_aggregation",
        "solver",
    ]
    lines = [f"- **{k}**: {d[k]}" for k in keep]
    overrides = []
    if d.get("floor_overrides"):
        overrides.append(f"floor_overrides: {d['floor_overrides']}")
    if d.get("cap_overrides"):
        overrides.append(f"cap_overrides: {d['cap_overrides']}")
    if d.get("weather_override"):
        overrides.append(f"weather_override: {d['weather_override']}")
    if overrides:
        lines.extend(f"- **{o.split(':')[0]}**: {o.split(':', 1)[1].strip()}" for o in overrides)
    return "\n".join(lines)


def _elastic_report_section(config: OptimizationConfig, elastic: dict[str, Any]) -> list[str]:
    """Presentation-style Task 2 block for Model C runs."""
    F0 = int(elastic.get("F0", config.fleet_baseline))
    F_star = float(elastic.get("F_star", F0))
    direction = str(elastic.get("direction", "—"))
    lam = float(elastic.get("breakeven_rental", 0))
    lam_star = float(elastic.get("shadow_price", 0))
    nv = float(elastic.get("nv_cost", 0))
    adj = float(elastic.get("fleet_adjustment_cost", 0))
    total = float(elastic.get("total_cost", 0))
    r = float(elastic.get("rental_rate", config.rental_rate))
    s = float(elastic.get("standdown_saving", config.standdown_saving))

    lines = [
        "",
        "## Task 2 — Elastic fleet (Model C)",
        "",
        "Joint optimization of **zone allocation q*** and **fleet size F*** using borough-specific overage costs.",
        "",
        "### Fleet decision",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| F₀ (baseline) | {F0:,} |",
        f"| F* (optimal) | {F_star:,.0f} |",
        f"| Direction | {direction} |",
        f"| λ(F₀) break-even rental | {lam:.4f} |",
        f"| λ(F*) shadow price | {lam_star:.4f} |",
        f"| Rental rate r | ${r:.0f} |",
        f"| Stand-down saving s | ${s:.0f} |",
        "",
        "### Cost breakdown (Model C)",
        "",
        "| Component | Amount |",
        "|-----------|--------|",
        f"| Newsvendor (allocation) cost | {nv:,.0f} |",
        f"| Fleet adjustment r·(F−F₀)⁺ − s·(F₀−F)⁺ | {adj:+,.0f} |",
        f"| **Total cost** | **{total:,.0f}** |",
        "",
    ]

    if direction == "stay" or abs(F_star - F0) < 0.5:
        lines.extend(
            [
                f"> **Dead zone:** ${s:.0f} ≤ λ(F₀)={lam:.2f} ≤ ${r:.0f} → optimal fleet stays at **F* = F₀**.",
                "> NV cost is minimized at the baseline fleet; expanding costs more than r per cab; contracting saves less than s per cab.",
                "",
            ]
        )
    elif direction == "expand":
        lines.extend(
            [
                f"> λ(F₀)={lam:.2f} **>** r=${r:.0f} → marginal cab value exceeds rental cost → **expand** to F*={F_star:,.0f}.",
                "",
            ]
        )
    elif direction == "contract":
        lines.extend(
            [
                f"> λ(F₀)={lam:.2f} **<** s=${s:.0f} → stand-down savings exceed marginal NV benefit → **contract** to F*={F_star:,.0f}.",
                "",
            ]
        )

    lines.append(elastic_summary(elastic, F0=F0))
    return lines


def build_single_report(
    config: OptimizationConfig,
    metrics: dict[str, Any],
    run_id: str,
    config_patch: Optional[dict[str, Any]] = None,
) -> str:
    scenario = metrics.get("scenario", {})
    elastic = metrics.get("elastic")
    is_elastic = bool(metrics.get("elastic_fleet") or config.elastic_fleet)

    lines = [
        f"# Optimization Report — `{run_id}`",
        "",
        "## Scenario",
        f"- Time bucket: **{scenario.get('time_bucket', config.time_bucket)}**",
        f"- Day of week: **{scenario.get('day_of_week', config.day_of_week)}**",
        f"- Week: **{scenario.get('week', config.week)}**",
        f"- Model: **{metrics.get('model_label', '—')}**",
        "",
        "## Configuration",
        _format_config(config),
    ]
    if config_patch:
        lines.extend(["", "## Applied patch", f"```json\n{json.dumps(config_patch, indent=2)}\n```"])

    lines.extend(
        [
            "",
            "## Run summary",
            f"- Demand: **SAA (empirical)**",
            f"- Path: **{metrics.get('path', 'fast')}**"
            + (f" ({metrics.get('path_mode')})" if metrics.get("path_mode") else ""),
            f"- Runtime: **{metrics.get('runtime_s')}s**",
        ]
    )

    if is_elastic and elastic:
        lines.extend(_elastic_report_section(config, elastic))
    else:
        lines.append(
            f"- Fixed fleet: **{metrics.get('fleet_used')} / {metrics.get('fleet_size')}**"
        )
        pm = metrics.get("pipeline_metrics") or {}
        if pm.get("nv_cost") is not None:
            source = pm.get("demand_source", "saa_mean")
            source_label = {
                "actual": "actual demand for this week/year",
                "saa_mean": "SAA sample mean (no exact-week data)",
            }.get(source, f"actual demand, {source}")
            lines.append(f"- NV cost ({source_label}): **{float(pm['nv_cost']):,.0f}**")

    feas = metrics.get("feasibility") or {}
    if feas:
        lines.extend(
            [
                "",
                "## Feasibility",
                f"- Feasible: **{feas.get('feasible')}**",
                f"- Total floor: **{feas.get('total_floor')}**",
                f"- Headroom: **{feas.get('headroom')}**",
            ]
        )

    bakeoff = metrics.get("bakeoff") or []
    if bakeoff:
        lines.extend(["", "## Model bake-off (mean NV cost, lower is better)", ""])
        lines.append("| model | mean_nv_cost | mean_fill_rate | rank |")
        lines.append("|-------|--------------|----------------|------|")
        for row in bakeoff:
            lines.append(
                f"| {row.get('model')} | {row.get('mean_nv_cost', 0):,.0f} | "
                f"{row.get('mean_fill_rate', 0):.3f} | {row.get('rank')} |"
            )

    top = metrics.get("top_zones") or []
    if top:
        lines.extend(["", "## Top allocated zones (q*)", ""])
        lines.append("| Zone | q* |")
        lines.append("|------|-----|")
        for row in top[:15]:
            lines.append(f"| {row['PULocationID']} | {row['q_star']} |")

    if is_elastic and elastic:
        lines.extend(
            [
                "",
                "## Interpretation",
                "- **F*** is the fleet size that minimizes total cost (NV + fleet adjustment), not the sum of rounded zone allocations.",
                "- Compare **total cost** (not NV alone) when evaluating expand vs contract vs stay.",
                "- See the **Presentation** tab for the same slot across Tasks 1 & 2 with precomputed A/B/C benchmarks.",
            ]
        )

    solver = metrics.get("solver_check") or []
    if solver:
        lines.extend(["", "## Solver cross-check", ""])
        lines.append("| solver | objective | status | max_diff_vs_slsqp |")
        lines.append("|--------|-----------|--------|-------------------|")
        for row in solver:
            lines.append(
                f"| {row.get('solver')} | {row.get('objective', 0):,.1f} | "
                f"{row.get('status')} | {row.get('max_diff_vs_slsqp', 0)} |"
            )

    lines.extend(["", "---", "*Template report — executive summary appended when OpenAI is configured.*"])
    return "\n".join(lines)


def build_full_report(metrics: dict[str, Any], run_id: str) -> str:
    lines = [
        f"# Full Bake-off Report — `{run_id}`",
        "",
        f"- Runtime: **{metrics.get('runtime_s')}s**",
        f"- Status: **{metrics.get('status')}**",
        f"- Outputs: `{metrics.get('outputs_dir', 'outputs')}/`",
        "",
        "## Model ranking (35 scenarios)",
        "",
        "| model | mean_nv_cost | mean_fill_rate | mean_revenue | rank |",
        "|-------|--------------|----------------|--------------|------|",
    ]
    for row in metrics.get("bakeoff") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('mean_nv_cost', 0):,.0f} | "
            f"{row.get('mean_fill_rate', 0):.3f} | "
            f"{row.get('mean_revenue', row.get('mean_nv_cost', 0)):,.0f} | {row.get('rank')} |"
        )
    lines.extend(
        [
            "",
            "## Charts",
            "- `outputs/charts/10_bakeoff.png`",
            "- `outputs/charts/11_revenue_comparison.png`",
            "",
            "---",
            "*Template report — executive summary appended when OpenAI is configured.*",
        ]
    )
    return "\n".join(lines)


def append_executive_summary(report: str, metrics: dict[str, Any]) -> str:
    """Call OpenAI for a short business summary if API key is set."""
    if not os.getenv("OPENAI_API_KEY"):
        return report

    try:
        from agent.llm import text_completion
    except ImportError:
        return report

    summary_input = {
        "report_tier": metrics.get("report_tier"),
        "scenario": metrics.get("scenario"),
        "elastic_fleet": metrics.get("elastic_fleet"),
        "elastic": metrics.get("elastic"),
        "fleet_used": metrics.get("fleet_used"),
        "fleet_size": metrics.get("fleet_size"),
        "primary_model": metrics.get("primary_model"),
        "bakeoff_top": (metrics.get("bakeoff") or [])[:4],
        "top_zones": (metrics.get("top_zones") or [])[:5],
    }
    try:
        summary = text_completion(
            "You summarize NYC yellow-taxi fleet allocation results for management. "
            "Write 3-5 sentences in English only. Focus on business meaning: demand, supply, "
            "fleet sizing (F* vs F0 if elastic), cost trade-offs, and which zones gained or lost cabs. "
            "Do not invent numbers.",
            f"Metrics JSON:\n{json.dumps(summary_input, indent=2)}",
        )
    except Exception:
        return report
    return report + "\n\n## Executive Summary\n\n" + summary
