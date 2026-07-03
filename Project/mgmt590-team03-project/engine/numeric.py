"""Safe numeric conversions for allocation vectors."""

from __future__ import annotations

import numpy as np


def safe_int(value, default: int = 0) -> int:
    """Convert to int, mapping NaN/inf to default."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return int(round(v))


def sanitize_allocation(q) -> np.ndarray:
    """Replace NaN/inf in an allocation vector with 0."""
    arr = np.asarray(q, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
