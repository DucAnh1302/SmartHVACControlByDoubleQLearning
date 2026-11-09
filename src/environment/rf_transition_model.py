"""
Random Forest Transition Model (Legacy Data-Driven Simulation).
Demonstrates observational machine-learning transition modeling and exposes
closed-loop confounding where action features receive negligible importance.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.environment.transition_base import BaseTransitionModel
from src.data.provenance import ProvenanceType


class RandomForestTransitionModel(BaseTransitionModel):
    """
    Random Forest Regression for next-state prediction:
    (T_t, RH_t, P_hvac, P_int, T_amb, hour, is_workhour) -> (T_t+1, RH_t+1)
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 8, seed: int = 42):
        super().__init__(name="Random Forest (Legacy)")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.model_temp: Optional[RandomForestRegressor] = None
        self.model_rh: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = [
            "temp_c", "humidity_rh", "hvac_kw", "internal_heat_kw",
            "ambient_temp_est", "hour", "is_workhour", "window_open"
        ]
        self.temp_feature_importances_: Dict[str, float] = {}
        self.rh_feature_importances_: Dict[str, float] = {}

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare X, y_temp, y_rh with next-step shifts."""
        df_shift = df.copy()
        df_shift["next_temp"] = df_shift["temp_c"].shift(-1)
        df_shift["next_rh"] = df_shift["humidity_rh"].shift(-1)
        if "window_open" not in df_shift.columns:
            df_shift["window_open"] = 0

        valid = df_shift.dropna(subset=["next_temp", "next_rh"] + [f for f in self.feature_names if f in df_shift.columns])

        X = valid[self.feature_names].values
        y_temp = valid["next_temp"].values
        y_rh = valid["next_rh"].values
        return X, y_temp, y_rh

    def fit(self, df_train: pd.DataFrame) -> "RandomForestTransitionModel":
        X, y_temp, y_rh = self._prepare_features(df_train)

        self.model_temp = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1
        ).fit(X, y_temp)

        self.model_rh = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.seed,
            n_jobs=-1
        ).fit(X, y_rh)

        self.temp_feature_importances_ = {
            f: round(float(imp), 4)
            for f, imp in zip(self.feature_names, self.model_temp.feature_importances_)
        }
        self.rh_feature_importances_ = {
            f: round(float(imp), 4)
            for f, imp in zip(self.feature_names, self.model_rh.feature_importances_)
        }

        self.is_fitted = True
        return self

    def predict_step(
        self,
        current_temp: float,
        current_rh: float,
        hvac_kw: float,
        internal_heat_kw: float,
        ambient_temp: float,
        ambient_rh: float,
        hour: int,
        window_open: int = 0
    ) -> Tuple[float, float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_step!")

        is_workhour = 1 if (8 <= hour < 18) else 0
        x = np.array([[
            current_temp, current_rh, hvac_kw, internal_heat_kw,
            ambient_temp, hour, is_workhour, window_open
        ]])

        pred_t = float(self.model_temp.predict(x)[0])
        pred_rh = float(self.model_rh.predict(x)[0])

        return (
            float(np.clip(pred_t, 16.0, 38.0)),
            float(np.clip(pred_rh, 20.0, 95.0))
        )

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_batch!")
        df_eval = df.copy()
        if "window_open" not in df_eval.columns:
            df_eval["window_open"] = 0
        if "is_workhour" not in df_eval.columns:
            df_eval["is_workhour"] = (
                (df_eval["datetime"].dt.dayofweek < 5) &
                (df_eval["hour"] >= 8) & (df_eval["hour"] < 18)
            ).astype(int)

        X = df_eval[self.feature_names].values
        pred_t = np.clip(self.model_temp.predict(X), 16.0, 38.0)
        pred_rh = np.clip(self.model_rh.predict(X), 20.0, 95.0)
        return pred_t, pred_rh
