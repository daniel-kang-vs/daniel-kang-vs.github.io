"""Temporary overlay storage — never mutates canonical project files until promoted."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from engine.config import OptimizationConfig

from agent.config_merge import default_config, merge_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OVERLAY_ROOT = PROJECT_ROOT / "agent_runs" / "overlays"
CANONICAL_ROOT = PROJECT_ROOT / "agent_runs" / "canonical"
CACHE_ROOT = PROJECT_ROOT / "agent_runs" / "cache"

CANONICAL_CONFIG = CANONICAL_ROOT / "config.json"
CANONICAL_AGG = CANONICAL_ROOT / "agg_active.parquet"
CANONICAL_SOURCE = PROJECT_ROOT / "cleaned_trips_2025.parquet"


def session_dir(session_id: str) -> Path:
    return OVERLAY_ROOT / session_id


def save_temp_config_patch(session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge patch onto baseline and save to session overlay only."""
    base = load_baseline_config()
    merged = merge_config(base, patch)
    d = session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "config.json"
    path.write_text(json.dumps(merged.model_dump(), indent=2), encoding="utf-8")
    meta = {
        "type": "config",
        "path": str(path.relative_to(PROJECT_ROOT)),
        "patch": patch,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (d / "overlay_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_baseline_config() -> OptimizationConfig:
    if CANONICAL_CONFIG.exists():
        return OptimizationConfig.model_validate_json(CANONICAL_CONFIG.read_text())
    return default_config()


def load_session_config(session_id: str) -> OptimizationConfig:
    path = session_dir(session_id) / "config.json"
    if path.exists():
        return OptimizationConfig.model_validate_json(path.read_text())
    return load_baseline_config()


def save_temp_upload(session_id: str, uploaded_path: Path, original_name: str) -> dict[str, Any]:
    """Store uploaded data under session overlay (does not touch cleaned_trips_2025)."""
    d = session_dir(session_id) / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    dest = d / original_name
    shutil.copy2(uploaded_path, dest)
    meta = {
        "type": "data_upload",
        "path": str(dest.relative_to(PROJECT_ROOT)),
        "original_name": original_name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    existing = _read_overlay_meta(session_id)
    existing["data_upload"] = meta
    (session_dir(session_id) / "overlay_meta.json").write_text(
        json.dumps(existing, indent=2), encoding="utf-8"
    )
    return meta


def set_session_agg_path(session_id: str, agg_path: Path) -> None:
    meta = _read_overlay_meta(session_id)
    meta["agg_overlay"] = {
        "path": str(agg_path.relative_to(PROJECT_ROOT)),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (session_dir(session_id) / "overlay_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def _read_overlay_meta(session_id: str) -> dict[str, Any]:
    path = session_dir(session_id) / "overlay_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_pending_overlay(session_id: str) -> dict[str, Any]:
    meta = _read_overlay_meta(session_id)
    pending = {
        "has_config_overlay": (session_dir(session_id) / "config.json").exists(),
        "has_data_upload": "data_upload" in meta,
        "has_agg_overlay": "agg_overlay" in meta,
        "meta": meta,
    }
    pending["has_any"] = any(
        [pending["has_config_overlay"], pending["has_data_upload"], pending["has_agg_overlay"]]
    )
    return pending


def resolve_agg_parquet_path(session_id: str) -> Path:
    """Priority: session agg overlay > canonical promoted agg > cached build from cleaned_trips."""
    meta = _read_overlay_meta(session_id)
    if "agg_overlay" in meta:
        p = PROJECT_ROOT / meta["agg_overlay"]["path"]
        if p.exists():
            return p
    if CANONICAL_AGG.exists():
        return CANONICAL_AGG
    from agent.data_source import ensure_cached_agg_from_cleaned_trips

    return ensure_cached_agg_from_cleaned_trips()


def promote_session_overlay(session_id: str) -> dict[str, Any]:
    """Copy session overlay to canonical store (permanent) — never overwrites cleaned_trips_2025."""
    CANONICAL_ROOT.mkdir(parents=True, exist_ok=True)
    promoted: dict[str, Any] = {"session_id": session_id, "items": []}

    cfg = session_dir(session_id) / "config.json"
    if cfg.exists():
        shutil.copy2(cfg, CANONICAL_CONFIG)
        promoted["items"].append("config")

    meta = _read_overlay_meta(session_id)
    if "agg_overlay" in meta:
        src = PROJECT_ROOT / meta["agg_overlay"]["path"]
        if src.exists():
            shutil.copy2(src, CANONICAL_AGG)
            promoted["items"].append("agg_active")

    promoted["promoted_at"] = datetime.now(timezone.utc).isoformat()
    return promoted


def discard_session_overlay(session_id: str) -> None:
    d = session_dir(session_id)
    if d.exists():
        shutil.rmtree(d)


def list_overlay_summary(session_id: str) -> str:
    pending = get_pending_overlay(session_id)
    if not pending["has_any"]:
        return "No temporary overlays."
    lines = ["**Temporary overlays (not yet permanent):**"]
    if pending["has_config_overlay"]:
        lines.append("- Config patch saved in session overlay")
    if pending["has_data_upload"]:
        lines.append(f"- Data upload: `{pending['meta'].get('data_upload', {}).get('path', '')}`")
    if pending["has_agg_overlay"]:
        lines.append(f"- Aggregated data: `{pending['meta'].get('agg_overlay', {}).get('path', '')}`")
    lines.append("\nUse **Save permanently** to promote, or **Discard** to drop temp changes.")
    return "\n".join(lines)
