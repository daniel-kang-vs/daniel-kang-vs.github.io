"""Track A: Log-linear (Poisson) demand model via sklearn — fast, interpretable.

Model:  log(E[demand_{z,t,d,m}]) = α_z + β_t + γ_d + δ_m + θ·weather
Fit via sklearn PoissonRegressor (coordinate descent, sparse-friendly).
Fits in ~5 seconds on 91k rows vs statsmodels IRLS which is slow on wide matrices.

For the newsvendor we need F_z^{-1}(τ_z).
With Poisson(λ_z): q* = scipy.stats.poisson.ppf(τ_z, mu=λ_z).
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings


WEATHER_COLS = ["temperature_2m", "precipitation", "relative_humidity_2m", "apparent_temperature"]
CAT_COLS = ["PULocationID", "time_bucket", "day_of_week", "week"]


class LogLinearDemandModel:
    """Poisson log-linear model with zone/bucket/dow/week fixed effects + weather."""

    def __init__(self, alpha: float = 1e-4, seed: int = 42):
        self.alpha = alpha   # L2 regularization
        self.seed = seed
        self._pipeline: Optional[Pipeline] = None
        self._fitted = False

    def _make_pipeline(self) -> Pipeline:
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, drop="first")
        num_scaler = StandardScaler()
        preprocessor = ColumnTransformer([
            ("cat", cat_enc, CAT_COLS),
            ("num", num_scaler, WEATHER_COLS),
        ])
        model = PoissonRegressor(alpha=self.alpha, max_iter=300, verbose=0)
        return Pipeline([("prep", preprocessor), ("model", model)])

    def _prep_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        for col in WEATHER_COLS:
            if col not in d.columns:
                d[col] = 0.0
            else:
                d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)
        d = d[CAT_COLS + WEATHER_COLS].copy()
        for col in CAT_COLS:
            d[col] = d[col].astype(str)
        return d

    def fit(self, train_df: pd.DataFrame) -> "LogLinearDemandModel":
        if "demand" not in train_df.columns:
            raise ValueError("demand column required; run proxy.add_demand() first")

        X = self._prep_df(train_df)
        y = train_df["demand"].values.astype(float).clip(min=0)

        self._pipeline = self._make_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipeline.fit(X, y)

        self._fitted = True
        train_pred = self._pipeline.predict(X)
        resid = y - train_pred
        print(f"  [LogLinear] Fitted | train MAE: {np.abs(resid).mean():.1f} | "
              f"train RMSE: {np.sqrt((resid**2).mean()):.1f}")
        return self

    def predict_mu(
        self,
        df: pd.DataFrame,
        weather_override: Optional[dict] = None,
    ) -> np.ndarray:
        """Return predicted Poisson rate λ_z per row."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        d = df.copy()
        if weather_override:
            for col, val in weather_override.items():
                if col in d.columns:
                    d[col] = val
        X = self._prep_df(d)
        return np.maximum(self._pipeline.predict(X), 0)

    def predict_for_scenario(
        self,
        df: pd.DataFrame,
        time_bucket: str,
        day_of_week: int,
        week: int,
        tau: np.ndarray,
        zones: np.ndarray,
        weather_override: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (zones, q_tau) — τ-quantile demand for each of the 263 zones."""
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

        # One representative row per zone (mean of any duplicates)
        for col in WEATHER_COLS:
            if col not in sub.columns:
                sub[col] = 0.0
            else:
                sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(0)
        sub = sub.groupby("PULocationID").mean(numeric_only=True).reset_index()
        sub = sub.set_index("PULocationID").reindex(zones).reset_index()
        sub["PULocationID"] = zones
        sub["time_bucket"] = time_bucket
        sub["day_of_week"] = day_of_week
        sub["week"] = week

        for col in WEATHER_COLS:
            if col in sub.columns:
                sub[col] = sub[col].fillna(sub[col].mean()).fillna(0)

        mu = self.predict_mu(sub, weather_override=weather_override)
        q = np.array([float(stats.poisson.ppf(tau[i], mu=max(mu[i], 0.5))) for i in range(len(zones))])
        return zones, np.maximum(q, 0).astype(float)
