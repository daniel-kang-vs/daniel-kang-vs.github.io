"""Check whether the optimization engine (lightgbm etc.) can be imported."""

from __future__ import annotations


def check_engine_import() -> str | None:
    """Return an error message if the engine cannot load, else None."""
    try:
        import lightgbm  # noqa: F401
    except OSError as exc:
        return (
            "LightGBM failed to load (missing OpenMP runtime). "
            f"Details: {exc}\n\n"
            "Fix: run Streamlit with Anaconda Python, or install libomp "
            "(macOS: `brew install libomp`)."
        )
    except ImportError as exc:
        return f"Engine dependency missing: {exc}"
    return None
