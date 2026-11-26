"""
Smart HVAC Simulation Environment for Reinforcement Learning.
Compatible with standard RL interfaces (step, reset, state_space, action_space).
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.environment.thermal_model import BuildingThermalModel
from src.reward.reward_function import HVACRewardFunction
from src.data.preprocess import discretize_state, state_to_index, get_state_space_size


# Discrete HVAC actions mapped to power consumption in kW
ACTION_POWER_MAP = {
    0: 0.0,   # OFF / Standby
    1: 10.0,  # LOW (Economy cooling)
    2: 22.0,  # MEDIUM (Standard cooling)
    3: 38.0   # HIGH (Boost cooling)
}

ACTION_NAMES = {
    0: "OFF (0 kW)",
    1: "LOW (10 kW)",
    2: "MED (22 kW)",
    3: "HIGH (38 kW)"
}


class SmartHVACEnv:
    """
    Simulation Environment for Smart HVAC Building Energy Management.
    Driven by real CU-BEMS load profiles and thermal dynamics.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        episode_length: int = 24,       # 24 hours per episode
        initial_temp_range: Tuple[float, float] = (25.0, 28.5),
        thermal_model: Optional[BuildingThermalModel] = None,
        reward_fn: Optional[HVACRewardFunction] = None,
        seed: Optional[int] = None
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.episode_length = episode_length
        self.initial_temp_range = initial_temp_range
        self.thermal_model = thermal_model or BuildingThermalModel()
        self.reward_fn = reward_fn or HVACRewardFunction()
        
        if seed is not None:
            np.random.seed(seed)

        # Action & State Spaces
        self.action_space_size = len(ACTION_POWER_MAP)
        self.n_states = get_state_space_size(n_temp_bins=5, n_load_bins=3, n_time_bins=3)

        # Environment runtime state
        self.current_step = 0
        self.data_idx = 0
        self.current_temp = 25.0
        self.current_state_tuple = (2, 0, 0)
        self.current_state_idx = 0

    def reset(
        self,
        start_idx: Optional[int] = None,
        initial_temp: Optional[float] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Parameters:
        -----------
        start_idx: Specific index into data_df, or None to sample randomly
        initial_temp: Specific starting temperature, or None to sample from initial_temp_range
        
        Returns:
        --------
        (state_idx, info)
        """
        self.current_step = 0

        # Choose start index with enough room for episode_length steps
        max_start = len(self.data_df) - self.episode_length - 1
        if max_start <= 0:
            raise ValueError(f"Dataset too short for episode length {self.episode_length}!")

        if start_idx is None:
            self.data_idx = np.random.randint(0, max_start)
        else:
            self.data_idx = min(max(0, start_idx), max_start)

        # Sample initial indoor temperature
        if initial_temp is None:
            self.current_temp = float(np.random.uniform(
                self.initial_temp_range[0], self.initial_temp_range[1]
            ))
        else:
            self.current_temp = float(initial_temp)

        row = self.data_df.iloc[self.data_idx]
        internal_heat = float(row.get("internal_heat_kw", row.get("plug_kw", 1.0) + row.get("light_kw", 0.5)))
        hour = int(row.get("hour", 0))

        # Compute initial state
        self.current_state_tuple = discretize_state(self.current_temp, internal_heat, hour)
        self.current_state_idx = state_to_index(self.current_state_tuple)

        info = {
            "step": self.current_step,
            "temp_c": self.current_temp,
            "state_tuple": self.current_state_tuple,
            "datetime": str(row.get("datetime", "")),
            "hour": hour,
            "is_workhour": int(row.get("is_workhour", 0)),
            "ambient_temp": float(row.get("ambient_temp_est", 28.0)),
            "internal_heat_kw": internal_heat
        }

        return self.current_state_idx, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of HVAC control action.
        
        Parameters:
        -----------
        action: 0 (OFF), 1 (LOW), 2 (MED), 3 (HIGH)
        
        Returns:
        --------
        (next_state_idx, reward, done, truncated, info)
        """
        if action not in ACTION_POWER_MAP:
            raise ValueError(f"Invalid action {action}! Valid actions are {list(ACTION_POWER_MAP.keys())}")

        hvac_kw = ACTION_POWER_MAP[action]
        row = self.data_df.iloc[self.data_idx]

        internal_heat = float(row.get("internal_heat_kw", 1.5))
        ambient_temp = float(row.get("ambient_temp_est", 28.5))
        hour = int(row.get("hour", 12))
        is_workhour = int(row.get("is_workhour", 1))

        # 1. Update physics (Thermal dynamics)
        next_temp = self.thermal_model.step(
            current_temp=self.current_temp,
            hvac_kw=hvac_kw,
            internal_heat_kw=internal_heat,
            ambient_temp=ambient_temp,
            add_noise=True
        )

        # 2. Compute multi-objective reward
        reward, r_breakdown = self.reward_fn.calculate_reward(
            temp_c=next_temp,
            hvac_kw=hvac_kw,
            is_workhour=is_workhour,
            hour=hour
        )

        # 3. Advance to next data row & step
        self.current_temp = next_temp
        self.current_step += 1
        self.data_idx += 1

        done = self.current_step >= self.episode_length
        truncated = False

        # Next state observation
        next_row = self.data_df.iloc[self.data_idx]
        next_internal_heat = float(next_row.get("internal_heat_kw", 1.5))
        next_hour = int(next_row.get("hour", (hour + 1) % 24))

        self.current_state_tuple = discretize_state(self.current_temp, next_internal_heat, next_hour)
        self.current_state_idx = state_to_index(self.current_state_tuple)

        info = {
            "step": self.current_step,
            "temp_c": self.current_temp,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "hvac_kw": hvac_kw,
            "ambient_temp": ambient_temp,
            "internal_heat_kw": internal_heat,
            "hour": hour,
            "is_workhour": is_workhour,
            "state_tuple": self.current_state_tuple,
            **r_breakdown
        }

        return self.current_state_idx, reward, done, truncated, info
