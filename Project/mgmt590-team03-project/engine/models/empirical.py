"""Empirical quantile allocator — zero-ML reference baseline.

For each (zone, bucket, dow) pools demand across all months in the training set
and reads off the tau-quantile directly.  This is the reference that Track A (QR)
and Track B (DFL) must beat on realized newsvendor cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class EmpiricalAllocator:
    """Stores per-(zone, bucket, dow) demand distributions from training data."""

    def __init__(self):
        self._samples: dict[tuple, np.ndarray] = {}  # (zone, bucket, dow) -> demand array

    def fit(self, train_df: pd.DataFrame) -> "EmpiricalAllocator":
        """Index training demand samples by (PULocationID, time_bucket, day_of_week)."""
        if "demand" not in train_df.columns:
            raise ValueError("Run proxy.add_demand() before fitting EmpiricalAllocator")

        for (zone, bucket, dow), grp in train_df.groupby(
            ["PULocationID", "time_bucket", "day_of_week"]
        ):
            self._samples[(int(zone), str(bucket), int(dow))] = grp["demand"].values
        return self

    def get_samples(
        self,
        zones: np.ndarray,
        time_bucket: str,
        day_of_week: int,
        fallback_to_bucket: bool = True,
    ) -> np.ndarray:
        """Return demand sample matrix (n_samples × n_zones).

        When a (zone, bucket, dow) cell has no training data, falls back to the
        bucket-only distribution (pooling all dow).
        """
        cols = []
        for zone in zones:
            key = (int(zone), time_bucket, day_of_week)
            arr = self._samples.get(key)
            if arr is None and fallback_to_bucket:
                # Pool all dow for this zone×bucket
                arr_parts = [
                    v for (z, b, d), v in self._samples.items()
                    if z == int(zone) and b == time_bucket
                ]
                arr = np.concatenate(arr_parts) if arr_parts else np.array([0])
            if arr is None or len(arr) == 0:
                arr = np.array([0])
            cols.append(arr)

        # Align lengths and cap at MAX_SAMPLES for solver speed
        MAX_SAMPLES = 50
        max_len = min(max(len(c) for c in cols), MAX_SAMPLES)
        rng = np.random.default_rng(42)
        aligned = np.column_stack([
            rng.choice(c, size=max_len, replace=True)
            for c in cols
        ])
        return aligned.astype(float)

    def predict_quantile(
        self,
        zones: np.ndarray,
        time_bucket: str,
        day_of_week: int,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Return the tau-quantile of demand per zone (unconstrained q*)."""
        samples = self.get_samples(zones, time_bucket, day_of_week)
        return np.array([
            float(np.quantile(samples[:, i], tau[i]))
            for i in range(len(zones))
        ])
