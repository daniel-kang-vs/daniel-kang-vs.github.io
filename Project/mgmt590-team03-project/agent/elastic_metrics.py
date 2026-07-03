"""Elastic fleet (Task 2 / Model C) metrics for agent runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.config import OptimizationConfig
from engine.numeric import safe_int


def elastic_dict_from_solve_result(
    elastic_result: Any,
    config: OptimizationConfig,
) -> dict[str, Any]:
    direction = str(elastic_result.status).replace("optimal:", "")
    return {
        "F0": float(config.fleet_baseline),
        "F_star": float(elastic_result.fleet_optimal),
        "direction": direction,
        "shadow_price": float(elastic_result.shadow_price),
        "breakeven_rental": float(elastic_result.breakeven_rental),
        "nv_cost": float(elastic_result.objective),
        "fleet_adjustment_cost": float(elastic_result.fleet_adjustment_cost or 0),
        "total_cost": float(elastic_result.total_cost or 0),
        "status": elastic_result.status,
        "rental_rate": float(config.rental_rate),
        "standdown_saving": float(config.standdown_saving),
    }


def ensure_elastic_payload(
    config: OptimizationConfig,
    result: dict[str, Any],
) -> dict[str, Any] | None:
    """Return elastic metrics dict; compute from engine if pipeline omitted it."""
    if not config.elastic_fleet:
        return None

    existing = result.get("elastic")
    if existing:
        return existing

    from engine import optimize as opt_mod

    fleet_min = config.fleet_min
    if fleet_min is None:
        fleet_min = int(result["floors"].sum())

    elastic_result = opt_mod.solve_elastic_fleet(
        result["emp_samples"],
        result["Cu"],
        result["Co"],
        result["floors"],
        result["caps"],
        F0=float(config.fleet_baseline),
        r=config.rental_rate,
        s=config.standdown_saving,
        zones=result["zones"],
        fleet_min=float(fleet_min),
    )
    payload = elastic_dict_from_solve_result(elastic_result, config)
    result["elastic"] = payload
    result["q_star"] = elastic_result.q_star_int.astype(float)
    return payload


def apply_elastic_to_metrics(
    metrics: dict[str, Any],
    config: OptimizationConfig,
    elastic: dict[str, Any],
    q_star: np.ndarray,
) -> None:
    """Mutate metrics in place with Task 2 fleet / cost fields."""
    metrics["elastic"] = elastic
    metrics["elastic_fleet"] = True
    metrics["fleet_F0"] = int(config.fleet_baseline)
    metrics["fleet_F_star"] = float(elastic["F_star"])
    metrics["fleet_used"] = safe_int(round(elastic["F_star"]))
    metrics["fleet_size"] = int(config.fleet_baseline)
    metrics["nv_cost"] = elastic.get("nv_cost")
    metrics["total_cost"] = elastic.get("total_cost")
    metrics["fleet_adj_cost"] = elastic.get("fleet_adjustment_cost")
    metrics["model_variant"] = "C_elastic"
    metrics["model_label"] = "Model C — borough Co, elastic F* (Task 2)"
