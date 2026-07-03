"""Deterministic tools — the LLM never touches raw data here."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from engine.config import OptimizationConfig
from engine.numeric import safe_int, sanitize_allocation

from agent.catalog import PROJECT_ROOT, check_data_freshness, get_data_catalog
from agent.config_merge import merge_config, validate_merged_config
from agent.elastic_metrics import apply_elastic_to_metrics, ensure_elastic_payload
from agent.model_modes import model_label, model_variant, normalize_task_config
from agent.data_source import load_prepared_dataframe
from agent.model_cache import get_cached_models, invalidate_session_cache, store_cached_models
from agent.overlays import get_pending_overlay, save_temp_config_patch
from agent.registry import append_run
from agent.schemas import JobEstimate

RUN_RESULTS_SCRIPT = PROJECT_ROOT / "run_results.py"


def run_fast_pipeline(
    config: OptimizationConfig,
    session_id: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Single-scenario optimization (Fast Path). Uses cached models when possible."""
    try:
        from engine.pipeline import run_pipeline
    except OSError as exc:
        raise RuntimeError(
            "Optimization engine unavailable (LightGBM/OpenMP). "
            "Use Anaconda Python or install libomp. "
            f"Original error: {exc}"
        ) from exc

    t0 = time.perf_counter()
    df_full = load_prepared_dataframe(
        session_id,
        n_clusters=config.n_clusters,
        seed=config.random_seed,
    )

    cached_models, solve_only = get_cached_models(session_id, config)
    pipeline_kwargs: dict[str, Any] = {
        "df_full": df_full,
        "verbose": verbose,
        "skip_solver_check": True,
        "skip_proxy_sensitivity": True,
        "empirical_only": True,
        "include_dfl": False,
    }

    if solve_only and cached_models and cached_models.get("empirical") is not None:
        pipeline_kwargs["empirical"] = cached_models["empirical"]

    result = run_pipeline(config, **pipeline_kwargs)
    store_cached_models(session_id, config, {"empirical": result["models"].get("empirical")})
    runtime = round(time.perf_counter() - t0, 2)

    zones = result["zones"]
    elastic = ensure_elastic_payload(config, result)
    q_star = sanitize_allocation(result["q_star"])
    top_idx = sorted(range(len(zones)), key=lambda i: q_star[i], reverse=True)[:15]

    variant = model_variant(config)
    metrics = {
        "path": "fast",
        "path_mode": "solve_only" if solve_only else "refit",
        "demand_model": "empirical",
        "model_variant": variant,
        "model_label": model_label(config),
        "report_tier": "single",
        "runtime_s": runtime,
        "fleet_used": safe_int(q_star.sum()),
        "fleet_size": config.fleet_size,
        "fleet_F0": config.fleet_baseline if config.elastic_fleet else config.fleet_size,
        "primary_model": "empirical",
        "co_mode": config.co_mode,
        "elastic_fleet": config.elastic_fleet,
        "scenario": {
            "time_bucket": config.time_bucket,
            "day_of_week": config.day_of_week,
            "week": config.week,
            "co_mode": config.co_mode,
            "elastic_fleet": config.elastic_fleet,
        },
        "feasibility": result.get("feasibility"),
        "pipeline_metrics": result.get("metrics"),
        "top_zones": [
            {"PULocationID": safe_int(zones[i]), "q_star": safe_int(q_star[i])}
            for i in top_idx
        ],
        "q_star_table": [
            {"PULocationID": safe_int(zones[i]), "q_star": float(q_star[i])}
            for i in range(len(zones))
        ],
        "bakeoff": result["bakeoff"].to_dict(orient="records") if not result["bakeoff"].empty else [],
        "solver_check": result["solver_check"].to_dict(orient="records")
        if not result["solver_check"].empty
        else [],
        "data_source": get_data_catalog()["primary_data"],
        "session_overlay": get_pending_overlay(session_id),
    }
    if elastic:
        apply_elastic_to_metrics(metrics, config, elastic, q_star)
    return {"result": result, "metrics": metrics}


def estimate_slow_job(
    job_type: str,
    changed_files: list[str] | None = None,
) -> JobEstimate:
    changed = changed_files or []
    if job_type == "new_data":
        minutes = 15.0 if changed else 10.0
        msg = (
            f"New data overlay detected ({', '.join(changed) or 'uploaded/session data'}). "
            f"Estimated runtime ~{minutes:.0f} min (aggregation + model refit + full bake-off). "
            "Confirm to proceed. Canonical cleaned_trips_2025.parquet is never overwritten."
        )
    else:
        minutes = 8.0
        msg = (
            f"Full bake-off will retrain all models and evaluate 35 scenarios. "
            f"Estimated runtime ~{minutes:.0f} min. Confirm to proceed."
        )
    return JobEstimate(
        job_type="slow_refit" if job_type == "new_data" else "full_bakeoff",
        changed_files=changed,
        estimated_minutes=minutes,
        message=msg,
    )


def run_slow_bakeoff(timeout_seconds: int = 3600) -> dict[str, Any]:
    """Run full bake-off via run_results.py (Slow Path)."""
    if not RUN_RESULTS_SCRIPT.exists():
        raise FileNotFoundError(f"Missing {RUN_RESULTS_SCRIPT}")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(RUN_RESULTS_SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    runtime = round(time.perf_counter() - t0, 2)

    outputs_dir = PROJECT_ROOT / "outputs"
    bakeoff_path = outputs_dir / "bakeoff_results.csv"
    bakeoff_records: list[dict[str, Any]] = []
    if bakeoff_path.exists():
        import pandas as pd

        bakeoff_records = pd.read_csv(bakeoff_path).to_dict(orient="records")

    metrics = {
        "path": "slow",
        "report_tier": "full",
        "runtime_s": runtime,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
        "bakeoff": bakeoff_records,
        "outputs_dir": str(outputs_dir.relative_to(PROJECT_ROOT)),
    }
    if proc.returncode != 0:
        metrics["status"] = "failed"
        raise RuntimeError(metrics["stderr_tail"] or "run_results.py failed")
    metrics["status"] = "success"
    return {"metrics": metrics, "stdout": proc.stdout}


def apply_and_validate(
    base: OptimizationConfig,
    patch: dict[str, Any],
    session_id: str = "default",
) -> tuple[OptimizationConfig, bool, list[str], dict]:
    merged = merge_config(base, normalize_task_config(patch))
    ok, errors, feasibility = validate_merged_config(merged, session_id=session_id)
    return merged, ok, errors, feasibility


def persist_config_overlay(session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Save config patch to session overlay (temporary until user promotes)."""
    return save_temp_config_patch(session_id, patch)


def invalidate_pipeline_cache(session_id: str) -> None:
    """Drop cached models after new data upload or permanent promote."""
    invalidate_session_cache(session_id)
