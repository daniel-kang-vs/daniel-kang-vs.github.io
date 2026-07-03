"""Force-reload engine modules so a long-lived Streamlit process picks up code fixes."""

from __future__ import annotations

import importlib
import inspect
import re
import sys

# Only reload engine code — never agent.tools/graph/model_cache (would wipe in-memory model cache).
_ENGINE_MODULES = (
    "engine.costs",
    "engine.optimize",
    "engine.evaluate",
    "engine.pipeline",
    "engine.bounds",
    "engine.models.linear_demand",
    "engine.models.qr_lgbm",
    "engine.models.dfl_spo",
    "agent.data_source",
)


def reload_engine_modules() -> str | None:
    """Reload engine stack. Returns error message if costs.py still has the old bug."""
    for name in _ENGINE_MODULES:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
    import engine.costs as costs_mod

    source = inspect.getsource(costs_mod.compute_costs)
    # Flag only the old pandas groupby.apply bug — not row-wise costs.apply(_alpha, axis=1).
    if "include_groups" in source or re.search(r"groupby\([^)]+\)\.apply", source):
        return (
            "Stale engine/costs.py is still loaded (contains groupby.apply). "
            "Stop ALL Streamlit processes and restart with ./run_streamlit.sh"
        )
    return None
