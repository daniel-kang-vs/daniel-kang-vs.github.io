"""Merge ConfigPatch into OptimizationConfig with validation."""

from __future__ import annotations

from typing import Any

from engine.config import OptimizationConfig
from engine import bounds as bounds_mod
from engine import data as data_mod
from engine import proxy as proxy_mod


def default_config() -> OptimizationConfig:
    return OptimizationConfig()


def merge_config(base: OptimizationConfig, patch: dict[str, Any]) -> OptimizationConfig:
    data = base.model_dump()
    for key, value in patch.items():
        if value is not None:
            data[key] = value
    # In elastic-fleet (Model C) mode, F0 is `fleet_baseline`, not `fleet_size`.
    # If the user's patch only changed `fleet_size` (e.g. "fleet reduced to 9000"),
    # carry that into `fleet_baseline` too so F0 actually moves.
    if data.get("elastic_fleet") and patch.get("fleet_size") is not None and patch.get("fleet_baseline") is None:
        data["fleet_baseline"] = patch["fleet_size"]
    return OptimizationConfig.model_validate(data)


def validate_merged_config(
    config: OptimizationConfig,
    session_id: str = "default",
) -> tuple[bool, list[str], dict]:
    """Return (ok, errors, feasibility_info)."""
    errors: list[str] = []
    try:
        OptimizationConfig.model_validate(config.model_dump())
    except Exception as exc:
        errors.append(str(exc))
        return False, errors, {}

    try:
        from agent.data_source import load_prepared_dataframe

        df_full = load_prepared_dataframe(
            session_id,
            n_clusters=config.n_clusters,
            seed=config.random_seed,
        )
        df = proxy_mod.add_demand(df_full, aggregation=config.proxy_aggregation)
        train, _, _ = data_mod.temporal_split(df)
        bounds_df = bounds_mod.compute_bounds(
            train,
            time_bucket=config.time_bucket,
            day_of_week=config.day_of_week,
            floor_alpha=config.floor_alpha,
            cap_multiplier=config.cap_multiplier,
            floor_overrides=config.floor_overrides,
            cap_overrides=config.cap_overrides,
            fleet_size=config.fleet_size,
        )
        feasibility = bounds_mod.check_feasibility(bounds_df, config.fleet_size)
        if not feasibility["feasible"]:
            errors.extend(feasibility.get("issues", []))
        return len(errors) == 0, errors, feasibility
    except Exception as exc:
        errors.append(f"Feasibility check failed: {exc}")
        return False, errors, {}
