"""Data catalog — metadata only, never reads row-level data."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

PRIMARY_DATA = "cleaned_trips_2025.parquet"
CACHE_AGG = "agent_runs/cache/agg_from_cleaned_trips_2025.parquet"

TRACKED_FILES = [
    PRIMARY_DATA,
    CACHE_AGG,
    "agent_runs/canonical/agg_active.parquet",
    "cleaned_trips_2025_sample.csv",
    "updated_2025_agg.parquet",
]


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    meta: dict[str, Any] = {
        "exists": True,
        "path": _relative_path(path),
        "size_mb": round(stat.st_size / 1e6, 2),
        "last_modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if path.suffix == ".parquet":
        try:
            pf = pq.ParquetFile(path)
            meta["rows"] = pf.metadata.num_rows
            meta["columns"] = pf.schema_arrow.names
        except Exception as exc:
            meta["parquet_error"] = str(exc)
    return meta


def file_fingerprint(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(str(path.stat().st_mtime_ns).encode())
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:16]


def get_data_catalog() -> dict[str, Any]:
    files = {name: _file_meta(PROJECT_ROOT / name) for name in TRACKED_FILES}
    primary = PROJECT_ROOT / PRIMARY_DATA
    cache = PROJECT_ROOT / CACHE_AGG
    return {
        "primary_data": PRIMARY_DATA,
        "primary_ready": primary.exists(),
        "cache_agg_ready": cache.exists(),
        "files": files,
        "overlay_root": "agent_runs/overlays/",
        "canonical_root": "agent_runs/canonical/",
        "engine_config": "engine/config.py:OptimizationConfig",
        "models": ["empirical", "glm_pto", "lgbm_pto", "dfl"],
        "constraints": [
            "fleet_size",
            "floor_alpha",
            "cap_multiplier",
            "floor_overrides",
            "cap_overrides",
            "cu_multiplier",
            "co_multiplier",
            "weather_override",
        ],
    }


def check_data_freshness(previous_fingerprints: dict[str, str] | None = None) -> dict[str, Any]:
    current = {name: file_fingerprint(PROJECT_ROOT / name) for name in TRACKED_FILES}
    previous = previous_fingerprints or {}
    changed = [name for name, fp in current.items() if fp and fp != previous.get(name)]
    return {
        "changed_files": changed,
        "has_changes": bool(changed),
        "fingerprints": current,
    }
