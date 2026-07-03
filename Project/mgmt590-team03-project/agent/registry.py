"""Append-only run registry for traceability."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "agent_runs" / "registry.jsonl"


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def append_run(record: dict[str, Any]) -> str:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_id = record.get("run_id") or new_run_id()
    record = {
        **record,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with REGISTRY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return run_id


def load_runs(limit: int = 20) -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    lines = REGISTRY_PATH.read_text(encoding="utf-8").strip().splitlines()
    runs = [json.loads(line) for line in lines if line.strip()]
    return list(reversed(runs[-limit:]))


def get_run(run_id: str) -> Optional[dict[str, Any]]:
    for run in load_runs(limit=10_000):
        if run.get("run_id") == run_id:
            return run
    return None
