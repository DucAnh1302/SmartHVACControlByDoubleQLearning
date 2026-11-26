"""
Base Interface for Transition Models in Smart HVAC Simulation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class BaseTransitionModel(ABC):
    """
    Abstract base class for all next-state transition models:
    Given (T_t, RH_t, P_hvac, P_int, T_amb, RH_amb, dt), predict (T_t+1, RH_t+1).
    """

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, df_train: pd.DataFrame) -> "BaseTransitionModel":
        """Fit or calibrate model parameters using training data."""
        pass

    @abstractmethod
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
        """
        Predict next indoor temperature (°C) and relative humidity (%RH).
        """
        pass

    @abstractmethod
    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized 1-step prediction across an entire dataframe.
        """
        pass

    def rollout(
        self,
        initial_temp: float,
        initial_rh: float,
        df_trajectory: pd.DataFrame,
        hvac_powers_kw: np.ndarray,
        window_opens: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multi-step autoregressive rollout given an environmental trajectory.
        """
        n_steps = len(df_trajectory)
        pred_temps = np.zeros(n_steps)
        pred_rhs = np.zeros(n_steps)

        curr_t = initial_temp
        curr_rh = initial_rh

        for i in range(n_steps):
            row = df_trajectory.iloc[i]
            p_hvac = float(hvac_powers_kw[i])
            w_open = int(window_opens[i]) if window_opens is not None else 0
            p_int = float(row["internal_heat_kw"])
            t_amb = float(row["ambient_temp_est"])
            rh_amb = float(row.get("ambient_rh_est", 70.0))
            hour = int(row["hour"])

            next_t, next_rh = self.predict_step(
                current_temp=curr_t,
                current_rh=curr_rh,
                hvac_kw=p_hvac,
                internal_heat_kw=p_int,
                ambient_temp=t_amb,
                ambient_rh=rh_amb,
                hour=hour,
                window_open=w_open
            )
            pred_temps[i] = next_t
            pred_rhs[i] = next_rh
            curr_t = next_t
            curr_rh = next_rh

        return pred_temps, pred_rhs
