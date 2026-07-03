"""Track A2: LightGBM Quantile Regression with per-sample τ (Predict-then-Optimize).

Uses a custom LightGBM objective where τ_{z,t} is the per-row pinball loss coefficient.
Gradient = -τ_i if underpredicting (residual > 0), else (1 - τ_i) if overpredicting.
τ is joined onto training rows by (PULocationID, time_bucket) — 1,315 unique values
(263 zones × 5 buckets). Does NOT vary by DOW (fare economics are DOW-invariant).

Features: PULocationID, borough_id, cluster_id, time_bucket (label-encoded),
          day_of_week, week, cyclical encodings, weather.
"""

from __future__ import annotations

import pickle
import pathlib
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "PULocationID", "borough_id", "cluster_id",
    "day_of_week", "week",
    "week_sin", "week_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "temperature_2m", "precipitation", "relative_humidity_2m", "apparent_temperature",
    "tb_encoded",   # label-encoded time_bucket
]


class LGBMQuantileModel:
    """LightGBM with per-sample τ custom pinball objective over all buckets/zones/days."""

    def __init__(self, n_estimators: int = 300, num_leaves: int = 63, seed: int = 42):
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.seed = seed
        self._model: Optional[lgb.Booster] = None
        self._tb_encoder = LabelEncoder()
        self._mean_tau: float = 0.5   # kept for save/load compatibility
        self._fitted = False

    def _encode_buckets(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        df = df.copy()
        if fit:
            df["tb_encoded"] = self._tb_encoder.fit_transform(df["time_bucket"])
        else:
            df["tb_encoded"] = self._tb_encoder.transform(df["time_bucket"])
        return df

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        out = df[FEATURE_COLS].copy()
        for col in FEATURE_COLS:
            if col not in ("PULocationID", "borough_id", "cluster_id", "day_of_week", "week", "tb_encoded"):
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        return out.fillna(0)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame],
        tau_df: pd.DataFrame,       # columns: PULocationID, time_bucket, tau
    ) -> "LGBMQuantileModel":
        if "demand" not in train_df.columns:
            raise ValueError("demand column required; run proxy.add_demand() first")

        self._mean_tau = float(tau_df["tau"].mean())

        train_df = self._encode_buckets(train_df, fit=True)

        # Join per-row τ by (PULocationID, time_bucket); fallback to mean τ
        tau_joined = train_df[["PULocationID", "time_bucket"]].merge(
            tau_df[["PULocationID", "time_bucket", "tau"]],
            on=["PULocationID", "time_bucket"],
            how="left",
        )
        tau_per_row = tau_joined["tau"].fillna(self._mean_tau).values.astype(float)

        X_tr = self._build_features(train_df)
        y_tr = train_df["demand"].values.astype(float)

        # Scale labels to unit range so tree leaf values (bounded by τ ∈ [0,1]) can
        # reach targets. Predictions are scaled back up in predict().
        self._demand_scale = float(y_tr.mean()) if y_tr.mean() > 0 else 1.0
        y_tr_scaled = y_tr / self._demand_scale

        # Custom per-sample pinball objective on scaled labels
        def _pinball_objective(preds: np.ndarray, dataset: lgb.Dataset):
            labels = dataset.get_label()           # scaled demand
            residual = labels - preds
            grad = np.where(residual > 0, -tau_per_row, 1.0 - tau_per_row)
            hess = np.ones_like(preds)
            return grad, hess

        params = {
            "objective": _pinball_objective,
            "num_leaves": self.num_leaves,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": self.seed,
            "verbose": -1,
            "num_threads": -1,
        }

        d_tr = lgb.Dataset(X_tr, label=y_tr_scaled)
        callbacks = [lgb.log_evaluation(-1)]
        evals = []

        if val_df is not None and isinstance(val_df, pd.DataFrame) and not val_df.empty:
            val_df = self._encode_buckets(val_df, fit=False)
            X_vl = self._build_features(val_df)
            y_vl = val_df["demand"].values.astype(float) / self._demand_scale
            d_vl = lgb.Dataset(X_vl, label=y_vl, reference=d_tr)
            evals = [d_vl]

        self._model = lgb.train(
            params, d_tr,
            num_boost_round=self.n_estimators,
            valid_sets=evals or None,
            callbacks=callbacks,
        )
        self._fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        weather_override: Optional[dict] = None,
    ) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted; call fit() first")
        df = self._encode_buckets(df, fit=False)
        X = self._build_features(df)
        if weather_override:
            for col, val in weather_override.items():
                if col in X.columns:
                    X[col] = val
        raw = self._model.predict(X)
        return np.maximum(raw * self._demand_scale, 0)

    def predict_for_scenario(
        self,
        df: pd.DataFrame,
        time_bucket: str,
        day_of_week: int,
        week: int,
        weather_override: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (zones, predicted_demand) arrays for the given scenario."""
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

        if sub.empty:
            raise ValueError(f"No data for bucket={time_bucket}, dow={day_of_week}")

        preds = self.predict(sub, weather_override=weather_override)
        sub = sub.copy()
        sub["_pred"] = preds
        zone_pred = sub.groupby("PULocationID")["_pred"].mean().reset_index()
        return zone_pred["PULocationID"].values, zone_pred["_pred"].values

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "lgbm_model.txt"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump({
                "mean_tau": self._mean_tau,
                "demand_scale": getattr(self, "_demand_scale", 1.0),
                "n_estimators": self.n_estimators,
                "seed": self.seed,
                "tb_encoder_classes": self._tb_encoder.classes_.tolist(),
            }, f)

    @classmethod
    def load(cls, path: pathlib.Path) -> "LGBMQuantileModel":
        path = pathlib.Path(path)
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(n_estimators=meta["n_estimators"], seed=meta["seed"])
        obj._mean_tau = meta["mean_tau"]
        obj._demand_scale = meta.get("demand_scale", 1.0)
        obj._tb_encoder.classes_ = np.array(meta["tb_encoder_classes"])
        obj._model = lgb.Booster(model_file=str(path / "lgbm_model.txt"))
        obj._fitted = True
        return obj
