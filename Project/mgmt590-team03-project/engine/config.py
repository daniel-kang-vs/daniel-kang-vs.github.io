"""Pydantic configuration schema — single source of truth for the optimization engine.

All engine functions accept an OptimizationConfig and are pure functions of it, which is
what lets the Phase-2 agent re-run with a validated config diff without re-training.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field, model_validator

_DEFAULT_BOROUGH_MULTIPLIERS: Dict[int, float] = {
    1: 1.00,   # Manhattan
    2: 1.50,   # Bronx
    3: 1.25,   # Brooklyn
    4: 1.40,   # Queens
    5: 1.80,   # Staten Island
}
_DEFAULT_ZONE_OVERRIDES: Dict[int, float] = {
    132: 0.70,  # JFK
    138: 0.70,  # LGA
}


class OptimizationConfig(BaseModel):
    # ── Scenario selector ────────────────────────────────────────────────────
    time_bucket: Literal[
        "00:00-05:59", "06:00-08:59", "09:00-15:59", "16:00-18:59", "19:00-23:59"
    ] = "09:00-15:59"
    day_of_week: int = Field(default=1, ge=1, le=7, description="1=Monday … 7=Sunday")
    week: int = Field(default=26, ge=1, le=52, description="ISO week number (1=first week of Jan)")
    year: Optional[int] = Field(
        default=None, ge=2024, le=2026,
        description="Year for realized-demand evaluation (2024-2026). None = average across available years.",
    )

    # ── Fleet ────────────────────────────────────────────────────────────────
    fleet_size: int = Field(default=13000, ge=1)

    # ── Constraint multipliers (agent-tunable) ───────────────────────────────
    floor_alpha: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="floor_z = ceil(alpha * median_demand_z)",
    )
    cap_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        description="cap_z = ceil(multiplier * p95_demand_z)",
    )

    # Per-zone overrides (zone_id -> value); empty = use data-derived defaults
    floor_overrides: Dict[int, int] = Field(default_factory=dict)
    cap_overrides: Dict[int, int] = Field(default_factory=dict)

    # ── Cost multipliers (agent-tunable) ─────────────────────────────────────
    cu_multiplier: float = Field(default=1.0, ge=0.0, description="Scale Cu (underage cost)")
    co_multiplier: float = Field(default=1.0, ge=0.0, description="Scale Co (overage cost)")

    # ── Demand proxy ─────────────────────────────────────────────────────────
    proxy_aggregation: Literal["borough", "cluster", "global"] = "borough"
    n_clusters: int = Field(
        default=5,
        ge=2,
        le=20,
        description="KMeans clusters used when proxy_aggregation='cluster'",
    )
    demand_multiplier: float = Field(
        default=1.0, ge=0.0, description="Uniform demand scaling (e.g. for scenario analysis)"
    )

    # ── Weather scenario override ─────────────────────────────────────────────
    weather_override: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Override weather features for what-if scenarios. "
            "Keys: temperature_2m, precipitation, relative_humidity_2m, apparent_temperature"
        ),
    )

    # ── Zone-specific overage cost (Task 1) ──────────────────────────────────
    co_mode: Literal["flat", "borough"] = Field(
        default="flat",
        description="flat = uniform Co per bucket; borough = Co_z = alpha_{b(z)} * Co_t",
    )
    co_borough_multipliers: Dict[int, float] = Field(
        default_factory=lambda: dict(_DEFAULT_BOROUGH_MULTIPLIERS),
        description="borough_id -> alpha multiplier for Co_z (used when co_mode='borough')",
    )
    co_zone_overrides: Dict[int, float] = Field(
        default_factory=lambda: dict(_DEFAULT_ZONE_OVERRIDES),
        description="zone_id -> alpha override, beats borough (e.g. airports)",
    )

    # ── Elastic fleet (Task 2) ────────────────────────────────────────────────
    elastic_fleet: bool = Field(
        default=False,
        description="When True, fleet size F is jointly optimized with q*",
    )
    fleet_baseline: int = Field(
        default=13000,
        ge=1,
        description="F0: baseline fleet around which adjustment cost is measured",
    )
    rental_rate: float = Field(
        default=45.0,
        ge=0.0,
        description="r: cost per extra cab per slot when expanding fleet",
    )
    standdown_saving: float = Field(
        default=20.0,
        ge=0.0,
        description="s: saving per cab per slot when contracting fleet",
    )
    fleet_min: Optional[int] = Field(
        default=None,
        description="Hard minimum fleet (TLC-mandated). None => computed as sum(floors)",
    )

    # ── Solver ───────────────────────────────────────────────────────────────
    solver: Literal["slsqp", "water_filling", "cvxpy"] = "slsqp"
    random_seed: int = 42

    @model_validator(mode="after")
    def _check_fleet_feasibility(self) -> "OptimizationConfig":
        if self.floor_alpha < 0 or self.floor_alpha > 1:
            raise ValueError("floor_alpha must be in [0, 1]")
        if self.standdown_saving > self.rental_rate:
            raise ValueError("standdown_saving must be <= rental_rate (dead-zone requires r >= s)")
        return self
