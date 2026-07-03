"""In-process model cache for fast re-solve (constraint/scenario changes without refit)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Optional

from engine.config import OptimizationConfig

from agent.catalog import file_fingerprint
from agent.overlays import get_pending_overlay, resolve_agg_parquet_path

REFIT_CONFIG_KEYS = (
    "n_clusters",
    "random_seed",
    "proxy_aggregation",
    "demand_multiplier",
)

SOLVE_ONLY_KEYS = (
    "fleet_size",
    "floor_alpha",
    "cap_multiplier",
    "floor_overrides",
    "cap_overrides",
    "time_bucket",
    "day_of_week",
    "week",
    "weather_override",
    "solver",
    "cu_multiplier",
    "co_multiplier",
    "co_mode",
    "elastic_fleet",
    "fleet_baseline",
    "rental_rate",
    "standdown_saving",
    "fleet_min",
)


@dataclass
class PipelineModelCache:
    refit_fingerprint: str
    models: dict[str, Any]


_CACHE: dict[str, PipelineModelCache] = {}


GLOBAL_MODEL_CACHE_ID = "pipeline_models"


def data_fingerprint(session_id: str) -> str:
    return file_fingerprint(resolve_agg_parquet_path(session_id))


def refit_fingerprint(session_id: str, config: OptimizationConfig) -> str:
    payload: dict[str, Any] = {k: getattr(config, k) for k in REFIT_CONFIG_KEYS}
    payload["data_fp"] = data_fingerprint(session_id)
    overlay = get_pending_overlay(session_id)
    if overlay.get("has_data_upload"):
        payload["data_upload"] = overlay.get("meta", {}).get("data_upload")
    if overlay.get("has_agg_overlay"):
        payload["agg_overlay"] = overlay.get("meta", {}).get("agg_overlay")
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def get_cached_models(
    session_id: str,
    config: OptimizationConfig,
) -> tuple[Optional[dict[str, Any]], bool]:
    fp = refit_fingerprint(session_id, config)
    for key in (session_id, GLOBAL_MODEL_CACHE_ID):
        entry = _CACHE.get(key)
        if entry is None:
            continue
        if entry.refit_fingerprint != fp:
            continue
        return entry.models, True
    return None, False


def store_cached_models(
    session_id: str,
    config: OptimizationConfig,
    models: dict[str, Any],
) -> None:
    entry = PipelineModelCache(
        refit_fingerprint=refit_fingerprint(session_id, config),
        models=models,
    )
    _CACHE[session_id] = entry
    _CACHE[GLOBAL_MODEL_CACHE_ID] = entry


def invalidate_session_cache(session_id: str) -> None:
    _CACHE.pop(session_id, None)


def invalidate_global_model_cache() -> None:
    _CACHE.pop(GLOBAL_MODEL_CACHE_ID, None)
