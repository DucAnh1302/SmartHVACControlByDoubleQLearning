"""
Persistence Baseline Transition Model.
Predicts T(t+1) = T(t) and RH(t+1) = RH(t).
Serves as the zero-intelligence benchmark for transition model validity.
"""

from typing import Tuple
import numpy as np
import pandas as pd

from src.environment.transition_base import BaseTransitionModel


class PersistenceTransitionModel(BaseTransitionModel):
    """
    Persistence model: assumes the state does not change in the next step.
    Any valid physical or machine learning model MUST outperform this baseline.
    """

    def __init__(self):
        super().__init__(name="Persistence Baseline")
        self.is_fitted = True

    def fit(self, df_train: pd.DataFrame) -> "PersistenceTransitionModel":
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
        return (float(current_temp), float(current_rh))

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return df["temp_c"].values.copy(), df["humidity_rh"].values.copy()
