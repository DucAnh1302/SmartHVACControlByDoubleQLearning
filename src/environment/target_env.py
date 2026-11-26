"""
Approved Target Smart HVAC Reinforcement Learning Environment.
Specifications:
- 600 Discrete States: 10 Temperature × 4 Humidity × 3 Heat Load × 5 Time/Tariff Bins
- 5 Discrete Actions: SET_25, SET_23, SET_27, AC_OFF, WINDOW_OPEN
- Domain-calibrated HVAC equipment stages (EquipmentStageCalibration)
- Coupled thermal and humidity physics (GreyBoxThermalModel)
- Multi-objective reward (AdvancedHVACRewardFunction)
- Initial window state is strictly closed
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.environment.greybox_model import GreyBoxThermalModel, PhysicalParameters
from src.reward.advanced_reward import AdvancedHVACRewardFunction
from src.data.calibration import EquipmentStageCalibration
from src.data.provenance import ProvenanceType


TARGET_ACTIONS = {
    0: "SET_25",       # Normal Comfort Target: 25.0°C
    1: "SET_23",       # Max Comfort / Cool Target: 23.0°C
    2: "SET_27",       # Economy Target: 27.0°C
    3: "AC_OFF",       # Turn HVAC compressor OFF (0.0 kW), window closed
    4: "WINDOW_OPEN"   # Open window for natural ventilation, AC OFF (0.0 kW)
}

TARGET_ACTION_SETPOINTS = {
    0: 25.0,
    1: 23.0,
    2: 27.0,
    3: None,
    4: None
}


def discretize_state_600(
    temp_c: float,
    humidity_rh: float,
    internal_heat_kw: float,
    hour: int,
    dayofweek: int
) -> Tuple[int, int, int, int]:
    """
    Discretize continuous environmental variables into 4 discrete indices:
    1. Temperature Bin (10 bins)
    2. Humidity Bin (4 bins)
    3. Heat Load Bin (3 bins)
    4. Time / Tariff Bin (5 bins)
    Returns: (t_bin, rh_bin, load_bin, time_bin)
    """
    # 1. Temperature (10 bins)
    temp_cutoffs = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.5]
    t_bin = int(np.digitize(temp_c, temp_cutoffs))

    # 2. Humidity (4 bins)
    rh_cutoffs = [45.0, 55.0, 65.0]
    rh_bin = int(np.digitize(humidity_rh, rh_cutoffs))

    # 3. Internal Heat Load (3 bins)
    load_cutoffs = [2.0, 6.0]
    load_bin = int(np.digitize(internal_heat_kw, load_cutoffs))

    # 4. Time / Tariff (5 bins)
    is_weekend = (dayofweek >= 5)
    if is_weekend:
        time_bin = 4  # Weekend
    elif 0 <= hour < 6 or hour >= 22:
        time_bin = 0  # Night Off-Peak
    elif 6 <= hour < 9:
        time_bin = 1  # Morning Ramp
    elif 9 <= hour < 17:
        time_bin = 2  # Midday Peak Tariff
    else:
        time_bin = 3  # Evening Shoulder

    return (t_bin, rh_bin, load_bin, time_bin)


def state_tuple_to_index_600(state_tuple: Tuple[int, int, int, int]) -> int:
    """Encode 4D discrete state tuple into a flat integer in [0, 599]."""
    t_bin, rh_bin, load_bin, time_bin = state_tuple
    t_bin = min(max(t_bin, 0), 9)
    rh_bin = min(max(rh_bin, 0), 3)
    load_bin = min(max(load_bin, 0), 2)
    time_bin = min(max(time_bin, 0), 4)

    return (((t_bin * 4) + rh_bin) * 3 + load_bin) * 5 + time_bin


def index_to_state_tuple_600(state_idx: int) -> Tuple[int, int, int, int]:
    """Decode a flat integer in [0, 599] into 4D discrete state tuple."""
    idx = int(np.clip(state_idx, 0, 599))
    time_bin = idx % 5
    rem1 = idx // 5
    load_bin = rem1 % 3
    rem2 = rem1 // 3
    rh_bin = rem2 % 4
    t_bin = rem2 // 4
    return (t_bin, rh_bin, load_bin, time_bin)


class TargetHVACEnv:
    """
    Standard target simulation environment for Double Q-Learning in Smart HVAC.
    """

    def __init__(
        self,
        data_df: pd.DataFrame,
        episode_length: int = 24,
        calibration: Optional[EquipmentStageCalibration] = None,
        thermal_model: Optional[GreyBoxThermalModel] = None,
        reward_fn: Optional[AdvancedHVACRewardFunction] = None,
        seed: Optional[int] = 42
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.episode_length = episode_length
        self.thermal_model = thermal_model or GreyBoxThermalModel()
        self.reward_fn = reward_fn or AdvancedHVACRewardFunction()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Default equipment calibration if none provided (2018 Zone 1 baseline)
        self.calibration = calibration or EquipmentStageCalibration(
            domain_id="default_2018", year=2018, floor=2, zone="z1",
            off_power_kw=0.0, low_power_kw=12.50, med_power_kw=21.64, high_power_kw=31.81,
            off_frequency=0.68, total_samples=1000, active_samples=320,
            cluster_centers=[12.50, 21.64, 31.81], cluster_boundaries=[17.07, 26.72], inertia=50.0
        )

        # Space sizes
        self.n_states = 600
        self.n_actions = 5

        # Runtime state
        self.current_step = 0
        self.data_idx = 0
        self.current_temp = 25.5
        self.current_rh = 60.0
        self.window_state = 0   # 0: Closed, 1: Open (Strictly closed initially)
        self.last_action: Optional[int] = None
        self.current_state_idx = 0
        self.prev_hvac_kw: float = 0.0
        self.prev_setpoint: Optional[float] = None
        self.episode_energy_kwh: float = 0.0
        self.episode_switching_count: int = 0

    def get_model_config(self) -> Dict[str, Any]:
        """Return exact model configuration and provenance used by this environment."""
        return {
            "thermal_model_name": self.thermal_model.name,
            "is_fitted": getattr(self.thermal_model, "is_fitted", False),
            "thermal_model_is_fitted": getattr(self.thermal_model, "is_fitted", False),
            "thermal_model_provenance": getattr(self.thermal_model, "provenance", ProvenanceType.SIMULATED_ASSUMED).value,
            "parameters": self.thermal_model.params.to_dict() if hasattr(self.thermal_model, "params") else {},
            "calibration_domain": self.calibration.domain_id,
            "calibration_stages": self.calibration.stage_power_map,
            "episode_length": self.episode_length
        }

    def reset(
        self,
        start_idx: Optional[int] = None,
        initial_temp: Optional[float] = None,
        initial_rh: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment to a starting state with strict episode isolation.
        Restores all mutable state variables to initial conditions.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.window_state = 0  # Initialized closed
        self.last_action = None  # Crucial: None prevents spurious switching penalty on step 0
        self.prev_hvac_kw = 0.0
        self.prev_setpoint = None
        self.episode_energy_kwh = 0.0
        self.episode_switching_count = 0

        max_start = len(self.data_df) - self.episode_length - 2
        if max_start <= 0:
            raise ValueError(f"Data length {len(self.data_df)} is too short for episode length {self.episode_length}!")

        if start_idx is None:
            self.data_idx = int(self.rng.integers(0, max_start))
        else:
            self.data_idx = min(max(0, start_idx), max_start)

        row = self.data_df.iloc[self.data_idx]
        self.current_temp = float(initial_temp) if initial_temp is not None else float(row.get("temp_c", 25.5))
        self.current_rh = float(initial_rh) if initial_rh is not None else float(row.get("humidity_rh", 60.0))

        p_int = float(row["internal_heat_kw"])
        hour = int(row["hour"])
        dow = int(row.get("dayofweek", 0))

        state_tuple = discretize_state_600(self.current_temp, self.current_rh, p_int, hour, dow)
        self.current_state_idx = state_tuple_to_index_600(state_tuple)

        info = {
            "step": self.current_step,
            "temp_c": self.current_temp,
            "humidity_rh": self.current_rh,
            "state_tuple": state_tuple,
            "window_open": self.window_state,
            "hour": hour,
            "datetime": str(row.get("datetime", "")),
            "internal_heat_kw": p_int,
            "ambient_temp": float(row["ambient_temp_est"]),
            "episode_energy_kwh": self.episode_energy_kwh,
            "episode_switching_count": self.episode_switching_count
        }
        return self.current_state_idx, info

    def _determine_hvac_power(self, action: int, current_temp: float) -> Tuple[float, int]:
        """
        Thermostat-driven compressor staging from Action:
        Returns: (hvac_kw, window_state)
        """
        cal = self.calibration

        if action == 3:  # AC_OFF
            return (cal.off_power_kw, 0)
        elif action == 4:  # WINDOW_OPEN
            return (cal.off_power_kw, 1)

        # Setpoint actions: 0 -> 25°C, 1 -> 23°C, 2 -> 27°C
        setpoint = TARGET_ACTION_SETPOINTS[action]
        thermal_gap = current_temp - setpoint

        if thermal_gap <= -0.5:
            # Room is colder than target -> cutoff compressor
            p_hvac = cal.off_power_kw
        elif -0.5 < thermal_gap <= 0.8:
            p_hvac = cal.low_power_kw
        elif 0.8 < thermal_gap <= 2.0:
            p_hvac = cal.med_power_kw
        else:
            p_hvac = cal.high_power_kw

        return (float(p_hvac), 0)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step."""
        if action not in TARGET_ACTIONS:
            raise ValueError(f"Invalid action {action}! Valid actions are {list(TARGET_ACTIONS.keys())}")

        # Action switching penalty is strictly 0 on step 0 (when last_action is None)
        action_changed = (action != self.last_action) if self.last_action is not None else False
        if action_changed:
            self.episode_switching_count += 1
        self.last_action = action
        self.prev_setpoint = TARGET_ACTION_SETPOINTS.get(action, None)

        row = self.data_df.iloc[self.data_idx]
        p_int = float(row["internal_heat_kw"])
        amb_t = float(row["ambient_temp_est"])
        amb_rh = float(row.get("ambient_rh_est", 70.0))
        hour = int(row["hour"])
        dow = int(row.get("dayofweek", 0))
        is_workhour = int(row.get("is_workhour", 1))

        # 1. Map action to physical HVAC power and window state
        hvac_kw, window_state = self._determine_hvac_power(action, self.current_temp)
        self.window_state = window_state
        self.prev_hvac_kw = hvac_kw
        self.episode_energy_kwh += hvac_kw * 1.0

        # 2. Physics transition forward
        next_temp, next_rh = self.thermal_model.predict_step(
            current_temp=self.current_temp,
            current_rh=self.current_rh,
            hvac_kw=hvac_kw,
            internal_heat_kw=p_int,
            ambient_temp=amb_t,
            ambient_rh=amb_rh,
            hour=hour,
            window_open=self.window_state
        )

        # 3. Calculate reward
        reward, r_breakdown = self.reward_fn.calculate_reward(
            temp_c=next_temp,
            humidity_rh=next_rh,
            hvac_kw=hvac_kw,
            is_workhour=is_workhour,
            hour=hour,
            action_changed=action_changed
        )

        # 4. Advance clock
        self.current_temp = next_temp
        self.current_rh = next_rh
        self.current_step += 1
        self.data_idx += 1

        done = (self.current_step >= self.episode_length)
        truncated = False

        # Next state discretization
        next_row = self.data_df.iloc[self.data_idx]
        next_p_int = float(next_row["internal_heat_kw"])
        next_hour = int(next_row["hour"])
        next_dow = int(next_row.get("dayofweek", 0))

        state_tuple = discretize_state_600(self.current_temp, self.current_rh, next_p_int, next_hour, next_dow)
        self.current_state_idx = state_tuple_to_index_600(state_tuple)

        info = {
            "step": self.current_step,
            "temp_c": self.current_temp,
            "humidity_rh": self.current_rh,
            "action": action,
            "action_name": TARGET_ACTIONS[action],
            "hvac_kw": hvac_kw,
            "window_open": self.window_state,
            "ambient_temp": amb_t,
            "ambient_rh": amb_rh,
            "internal_heat_kw": p_int,
            "hour": hour,
            "state_tuple": state_tuple,
            "episode_energy_kwh": self.episode_energy_kwh,
            "episode_switching_count": self.episode_switching_count,
            "provenance": {
                "temp_c": ProvenanceType.SIMULATED_ASSUMED.value,
                "humidity_rh": ProvenanceType.SIMULATED_ASSUMED.value,
                "hvac_kw": ProvenanceType.FITTED_WITH_CAVEAT.value,
                "window_open": ProvenanceType.SIMULATED_ASSUMED.value,
                "internal_heat_kw": ProvenanceType.OBSERVED.value
            },
            **r_breakdown
        }

        return self.current_state_idx, reward, done, truncated, info
