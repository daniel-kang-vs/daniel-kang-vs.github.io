"""Stakeholder model presets — explicit Cu/Co and fleet settings per task.

Model A (baseline): Cu_z from data × 1.0, Co_z = Co_t (flat, α=1 everywhere).
Model B (Task 1):   same Cu; Co_z = α_b(z) · Co_t with borough α table.
Model C (Task 2): same costs as B; elastic fleet F* with r/s adjustment.
"""

from __future__ import annotations

from engine.config import OptimizationConfig, _DEFAULT_BOROUGH_MULTIPLIERS, _DEFAULT_ZONE_OVERRIDES


def model_a_flat(
    *,
    fleet_size: int = 13000,
    time_bucket: str = "09:00-15:59",
    day_of_week: int = 1,
    week: int = 50,
) -> OptimizationConfig:
    """Baseline — no Cu/Co multipliers, flat overage within bucket."""
    return OptimizationConfig(
        time_bucket=time_bucket,
        day_of_week=day_of_week,
        week=week,
        fleet_size=fleet_size,
        cu_multiplier=1.0,
        co_multiplier=1.0,
        co_mode="flat",
        elastic_fleet=False,
    )


def model_b_borough(
    *,
    fleet_size: int = 13000,
    time_bucket: str = "09:00-15:59",
    day_of_week: int = 1,
    week: int = 50,
) -> OptimizationConfig:
    """Task 1 — zone-specific overage via borough α multipliers."""
    return OptimizationConfig(
        time_bucket=time_bucket,
        day_of_week=day_of_week,
        week=week,
        fleet_size=fleet_size,
        cu_multiplier=1.0,
        co_multiplier=1.0,
        co_mode="borough",
        co_borough_multipliers=dict(_DEFAULT_BOROUGH_MULTIPLIERS),
        co_zone_overrides=dict(_DEFAULT_ZONE_OVERRIDES),
        elastic_fleet=False,
    )


def model_c_elastic(
    *,
    fleet_baseline: int = 13000,
    rental_rate: float = 45.0,
    standdown_saving: float = 20.0,
    time_bucket: str = "09:00-15:59",
    day_of_week: int = 1,
    week: int = 50,
) -> OptimizationConfig:
    """Task 2 — borough Co + elastic fleet."""
    cfg = model_b_borough(
        fleet_size=fleet_baseline,
        time_bucket=time_bucket,
        day_of_week=day_of_week,
        week=week,
    )
    return cfg.model_copy(
        update={
            "elastic_fleet": True,
            "fleet_baseline": fleet_baseline,
            "rental_rate": rental_rate,
            "standdown_saving": standdown_saving,
        }
    )


COST_SPEC = {
    "A_flat": "Cu_z = avg fare (×1.0); Co_z = Co_t bucket average (×1.0, α=1)",
    "B_borough": "Cu_z unchanged; Co_z = α_b(z)·Co_t (Manhattan 1.0 … SI 1.8; airports 0.7)",
    "C_elastic": "Same Cu/Co as B; fleet F* optimized with r=$45, s=$20",
}
