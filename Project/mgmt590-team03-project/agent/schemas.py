"""Structured outputs for the agent LLM and state serialization."""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


IntentType = Literal[
    "change_constraint",
    "run_whatif",
    "new_data",
    "full_bakeoff",
    "show_last_report",
    "clarify",
    "chitchat",
]

ReportTier = Literal["single", "full"]


class IntentClassification(BaseModel):
    intent: IntentType = Field(description="Classified user intent for routing.")
    rationale: str = Field(default="", description="One-sentence reason for the classification.")


class ConfigPatch(BaseModel):
    """Partial update for OptimizationConfig — only set fields the user changed."""

    time_bucket: Optional[
        Literal["00:00-05:59", "06:00-08:59", "09:00-15:59", "16:00-18:59", "19:00-23:59"]
    ] = None
    day_of_week: Optional[int] = Field(default=None, ge=1, le=7)
    week: Optional[int] = Field(default=None, ge=1, le=52)
    year: Optional[int] = Field(default=None, ge=2024, le=2026)
    fleet_size: Optional[int] = Field(default=None, ge=1)
    floor_alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    cap_multiplier: Optional[float] = Field(default=None, ge=1.0)
    floor_overrides: Optional[Dict[int, int]] = None
    cap_overrides: Optional[Dict[int, int]] = None
    cu_multiplier: Optional[float] = Field(default=None, ge=0.0)
    co_multiplier: Optional[float] = Field(default=None, ge=0.0)
    co_mode: Optional[Literal["flat", "borough"]] = None
    elastic_fleet: Optional[bool] = None
    fleet_baseline: Optional[int] = Field(default=None, ge=1)
    rental_rate: Optional[float] = Field(default=None, ge=0.0)
    standdown_saving: Optional[float] = Field(default=None, ge=0.0)
    proxy_aggregation: Optional[Literal["borough", "cluster", "global"]] = None
    n_clusters: Optional[int] = Field(default=None, ge=2, le=20)
    demand_multiplier: Optional[float] = Field(default=None, ge=0.0)
    weather_override: Optional[Dict[str, float]] = None
    solver: Optional[Literal["slsqp", "water_filling", "cvxpy"]] = None
    random_seed: Optional[int] = None

    def non_empty(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ExtractionResult(BaseModel):
    """Full extraction output including clarification state."""

    patch: ConfigPatch = Field(default_factory=ConfigPatch)
    run_requested: bool = False
    ambiguities: list[str] = Field(default_factory=list)
    clarification_question: Optional[str] = None
    interpreted_summary: str = ""
    notes: list[str] = Field(default_factory=list)

    def non_empty_patch(self) -> dict:
        return self.patch.non_empty()

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification_question)


class JobEstimate(BaseModel):
    job_type: Literal["slow_refit", "full_bakeoff"]
    changed_files: list[str] = Field(default_factory=list)
    estimated_minutes: float = Field(description="Rough runtime estimate.")
    message: str = Field(description="User-facing confirmation message.")
