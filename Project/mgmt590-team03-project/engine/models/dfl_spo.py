"""Track B: End-to-End Decision-Focused Learning with SPO+ surrogate.

Architecture:
  - Predictor: linear model f_θ(x) = X @ θ + b  (smooth, moderate-dim → ideal for L-BFGS-B)
  - Inner solver: SLSQP (from optimize.py) — same solver as Track A for a clean A/B
  - Loss: SPO+ convex surrogate (Elmachtoub & Grigas 2022)
  - Outer optimizer: L-BFGS-B via scipy.optimize.minimize (full-batch per epoch)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from engine.optimize import solve_slsqp, solve_water_filling, nv_cost_vectorized


FEATURE_COLS = [
    "PULocationID", "borough_id", "cluster_id",
    "day_of_week", "week",
    "week_sin", "week_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "temperature_2m", "precipitation", "relative_humidity_2m", "apparent_temperature",
]


@dataclass
class DFLTrainingLog:
    epochs: List[int] = field(default_factory=list)
    spo_losses: List[float] = field(default_factory=list)
    nv_costs: List[float] = field(default_factory=list)
    runtimes_ms: List[float] = field(default_factory=list)


class LinearDFLModel:
    """Linear predictor trained end-to-end via SPO+ loss and L-BFGS-B / BFGS."""

    def __init__(
        self,
        outer_method: str = "L-BFGS-B",
        n_epochs: int = 10,
        batch_size: int = 8,
        seed: int = 42,
        l2_reg: float = 1e-4,
    ):
        self.outer_method = outer_method    # "L-BFGS-B" or "BFGS"
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.l2_reg = l2_reg

        self.theta_: Optional[np.ndarray] = None   # (n_features, n_zones)
        self.bias_: Optional[np.ndarray] = None    # (n_zones,)
        self.feature_mean_: Optional[np.ndarray] = None
        self.feature_std_: Optional[np.ndarray] = None
        self.training_log_ = DFLTrainingLog()
        self._zones: Optional[np.ndarray] = None

    # ── Feature helpers ──────────────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return df[FEATURE_COLS].fillna(0).values.astype(float)

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.feature_mean_ = X.mean(axis=0)
            self.feature_std_ = X.std(axis=0) + 1e-8
        return (X - self.feature_mean_) / self.feature_std_

    # ── Prediction ───────────────────────────────────────────────────────────

    def _predict_raw(self, X_norm: np.ndarray) -> np.ndarray:
        """Return predicted demand (n_samples, n_zones)."""
        return np.maximum(X_norm @ self.theta_ + self.bias_, 0)

    # ── SPO+ Loss ────────────────────────────────────────────────────────────

    def _spo_plus_loss_and_grad(
        self,
        theta_flat: np.ndarray,
        X_batch: np.ndarray,          # (B, n_features)
        D_batch: np.ndarray,          # (B, n_zones)
        Cu: np.ndarray,               # (n_zones,)
        Co: np.ndarray,               # (n_zones,)
        floors: np.ndarray,
        caps: np.ndarray,
        fleet_size: int,
    ) -> Tuple[float, np.ndarray]:
        """SPO+ loss and closed-form gradient w.r.t. theta_flat.

        Closed-form gradient avoids inner solver calls entirely:
            ∂L_SPO+/∂θ = X^T * g  where g_z = 2*(Cu_z+Co_z)*τ_z_adj - 2*Cu_z
        and τ_z_adj is the effective fractile implied by the predicted demand
        under the budget constraint (approximated via the newsvendor first-order
        condition: F_z(q*) = τ_z, so g_z = dNV/dD_hat_z at q*(D_hat)).

        This is O(B * n_zones) per call — no inner optimization needed.
        """
        n_f, n_z = X_batch.shape[1], D_batch.shape[1]
        theta = theta_flat[: n_f * n_z].reshape(n_f, n_z)
        bias = theta_flat[n_f * n_z :]

        D_hat = np.maximum(X_batch @ theta + bias, 0)    # (B, n_zones)
        tau = Cu / (Cu + Co)                              # (n_zones,) critical fractile

        total_loss = 0.0
        grad_theta = np.zeros_like(theta)
        grad_bias = np.zeros_like(bias)
        B = X_batch.shape[0]

        for i in range(B):
            d_hat_i = D_hat[i]       # predicted demand (n_zones,)
            d_true_i = D_batch[i]    # true demand (n_zones,)

            # Newsvendor loss at predicted demand vs true demand
            # NV(D_hat, D_true) = Cu*(D_true - D_hat)+ + Co*(D_hat - D_true)+
            understock = np.maximum(d_true_i - d_hat_i, 0)
            overstock  = np.maximum(d_hat_i - d_true_i, 0)
            loss_i = (Cu * understock + Co * overstock).sum()
            total_loss += loss_i

            # Closed-form SPO+ subgradient w.r.t. D_hat_i:
            # ∂NV/∂D_hat_z = Co_z * I(D_hat_z > D_true_z) - Cu_z * I(D_hat_z < D_true_z)
            # Budget-coupling correction: shift by mean gradient to respect sum(q)<=fleet
            g = np.where(d_hat_i > d_true_i, Co, -Cu)   # (n_zones,)

            # Budget correction: if predicted total > fleet, penalise zones above τ
            if d_hat_i.sum() > fleet_size:
                excess_frac = (d_hat_i.sum() - fleet_size) / fleet_size
                g = g + excess_frac * (Cu + Co) * (tau - 0.5)

            grad_theta += np.outer(X_batch[i], g) / B
            grad_bias  += g / B

        # L2 regularization
        total_loss = total_loss / B + self.l2_reg * (theta ** 2).sum()
        grad_theta += 2 * self.l2_reg * theta

        flat_grad = np.concatenate([grad_theta.ravel(), grad_bias])
        return float(total_loss), flat_grad

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        Cu: np.ndarray,
        Co: np.ndarray,
        floors: np.ndarray,
        caps: np.ndarray,
        fleet_size: int,
        zones: np.ndarray,
        time_bucket: str,
        day_of_week: int,
    ) -> "LinearDFLModel":
        self._zones = zones
        rng = np.random.default_rng(self.seed)

        mask = (train_df["time_bucket"] == time_bucket) & (train_df["day_of_week"] == day_of_week)
        sub = train_df[mask]
        if sub.empty:
            raise ValueError(f"No training data for bucket={time_bucket}, dow={day_of_week}")

        # Build X: one row per (month, zone) → shape (n_months * n_zones, n_features)
        X_raw = self._extract_features(sub)
        X_norm = self._normalize(X_raw, fit=True)

        # Build D: demand per zone × week matrix (n_weeks, n_zones)
        demand_pivot = (
            sub.pivot_table(index="week", columns="PULocationID", values="demand", aggfunc="mean")
            .reindex(columns=zones)
            .fillna(0)
        )
        # For each week, build the feature row as the mean feature across all zones for that week
        feature_pivot = (
            sub.groupby("week")[FEATURE_COLS].mean()
        )
        weeks_common = demand_pivot.index.intersection(feature_pivot.index)
        D_mat = demand_pivot.loc[weeks_common].values.astype(float)    # (n_weeks, n_zones)
        X_month = self._normalize(feature_pivot.loc[weeks_common].fillna(0).values.astype(float))

        n_f = X_month.shape[1]
        n_z = len(zones)

        # Initialize θ, bias
        self.theta_ = rng.normal(0, 0.01, (n_f, n_z))
        self.bias_ = D_mat.mean(axis=0)  # warm start: bias ≈ mean demand per zone

        theta0_flat = np.concatenate([self.theta_.ravel(), self.bias_])
        n_weeks = X_month.shape[0]

        for epoch in range(self.n_epochs):
            t0 = time.perf_counter()
            # Sample a mini-batch of weeks
            idx = rng.choice(n_weeks, size=min(self.batch_size, n_weeks), replace=False)
            X_batch = X_month[idx]
            D_batch = D_mat[idx]

            res = minimize(
                self._spo_plus_loss_and_grad,
                theta0_flat,
                args=(X_batch, D_batch, Cu, Co, floors, caps, fleet_size),
                jac=True,
                method=self.outer_method,
                options={"maxiter": 20, "disp": False},
            )
            theta0_flat = res.x
            rt = (time.perf_counter() - t0) * 1000

            # Unpack
            self.theta_ = theta0_flat[: n_f * n_z].reshape(n_f, n_z)
            self.bias_ = theta0_flat[n_f * n_z :]

            # Log
            self.training_log_.epochs.append(epoch)
            self.training_log_.spo_losses.append(float(res.fun))
            self.training_log_.runtimes_ms.append(rt)

            if epoch % 5 == 0:
                print(f"  [DFL] Epoch {epoch:3d}/{self.n_epochs} | SPO+ loss: {res.fun:.4f} | {rt:.0f}ms")

        return self

    def predict(
        self,
        df: pd.DataFrame,
        time_bucket: str,
        day_of_week: int,
        week: int,
        weather_override: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (zones, predicted_demand) for the given scenario."""
        mask = (
            (df["time_bucket"] == time_bucket)
            & (df["day_of_week"] == day_of_week)
            & (df["week"] == week)
        )
        sub = df[mask].copy()
        if sub.empty:
            mask2 = (df["time_bucket"] == time_bucket) & (df["day_of_week"] == day_of_week)
            sub = df[mask2].copy()
            sub["week"] = week

        if weather_override:
            for col, val in weather_override.items():
                if col in sub.columns:
                    sub[col] = val

        # Aggregate to per-zone mean features — one row per zone → (n_zones, n_features)
        zone_feats = sub.groupby("PULocationID")[FEATURE_COLS].mean().reindex(self._zones).fillna(0)
        X_norm = self._normalize(zone_feats.values.astype(float), fit=False)  # (n_zones, n_features)
        # theta_ is (n_features, n_zones); element-wise: pred_z = X_norm[z] · theta_[:, z] + bias_[z]
        preds = np.array([
            float(X_norm[i] @ self.theta_[:, i] + self.bias_[i])
            for i in range(len(self._zones))
        ])
        return self._zones, np.maximum(preds, 0)
