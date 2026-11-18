"""
Baseline Policies for Multi-Device HVAC Benchmark Comparison.
Implements:
1. FixedSetpointMultiDevice: Always maintains COMFORT_COOL (Action 1)
2. ConventionalThermostatMultiDevice: Simple deadband thermostat (Cool/Heat/Off)
3. RuleBasedSmartHVACMultiDevice: Multi-variable rule-based heuristic controller
4. RandomMultiDevicePolicy: Pure random exploration baseline
"""

from typing import Dict, Any, Optional
import numpy as np

from src.agent.base_agent import BaseAgent
from src.environment.multi_device_env import index_to_state_tuple_720


class FixedSetpointMultiDevice(BaseAgent):
    """Always commands COMFORT_COOL (Action 1: 25°C Setpoint, low vent, low filter)."""

    def __init__(self):
        super().__init__(n_states=720, n_actions=12, name="Fixed Setpoint (25°C Cool)")
        self.is_learning_agent = False
        self.last_action_is_exploring = False

    def select_action(self, state: int, evaluate: bool = False) -> int:
        return 1

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class ConventionalThermostatMultiDevice(BaseAgent):
    """
    Standard building thermostat with deadband [23.5°C, 26.5°C].
    Cools when hot, heats when cold, idle otherwise. Ignores CO2 and PM2.5.
    """

    def __init__(self):
        super().__init__(n_states=720, n_actions=12, name="Conventional Thermostat")
        self.is_learning_agent = False
        self.last_action_is_exploring = False

    def select_action(self, state: int, evaluate: bool = False) -> int:
        t_bin, rh_bin, co2_bin, pm_bin, load_bin, time_bin = index_to_state_tuple_720(state)
        # t_bin: 0: <21, 1: [21, 24), 2: [24, 26], 3: (26, 28.5), 4: >=28.5
        if t_bin == 0:
            return 7  # MAX_HEAT
        elif t_bin == 1:
            return 6  # COMFORT_HEAT
        elif t_bin == 2:
            return 0  # ALL_OFF_STANDBY (deadband)
        elif t_bin == 3:
            return 1  # COMFORT_COOL
        else:
            return 2  # MAX_COOL

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class RuleBasedSmartHVACMultiDevice(BaseAgent):
    """
    Expert heuristic rule-based controller monitoring Temp, Humidity, CO2, and PM2.5.
    """

    def __init__(self):
        super().__init__(n_states=720, n_actions=12, name="Rule-Based Smart HVAC")
        self.is_learning_agent = False
        self.last_action_is_exploring = False

    def select_action(self, state: int, evaluate: bool = False) -> int:
        t_bin, rh_bin, co2_bin, pm_bin, load_bin, time_bin = index_to_state_tuple_720(state)

        # 1. Thermal safety priority
        if t_bin >= 4:
            return 2  # MAX_COOL
        if t_bin == 0:
            return 7  # MAX_HEAT

        # 2. Severe Air Quality / CO2 priority
        if co2_bin >= 2:
            return 5  # ACTIVE_VENT_PURIFY
        if pm_bin == 1 and t_bin == 2:
            return 10  # ECO_AIR_PURIFY

        # 3. Severe Humidity priority
        if rh_bin == 2 and t_bin in [2, 3]:
            return 8  # DEHUMIDIFY_ACTIVE
        if rh_bin == 0 and t_bin in [1, 2]:
            return 9  # HUMIDIFY_COMFORT

        # 4. Standard comfort cooling / heating
        if t_bin == 3:
            return 3 if time_bin == 2 else 1  # ECONOMY_COOL on peak, else COMFORT_COOL
        if t_bin == 1:
            return 6  # COMFORT_HEAT

        # 5. Night setback
        if time_bin == 0:
            return 11  # NIGHT_SETBACK

        # Default optimal comfort
        return 1  # COMFORT_COOL

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class RandomMultiDevicePolicy(BaseAgent):
    """Pure random uniform baseline."""

    def __init__(self, seed: int = 42):
        super().__init__(n_states=720, n_actions=12, name="Random Uniform Policy")
        self.rng = np.random.default_rng(seed)
        self.is_learning_agent = False
        self.last_action_is_exploring = True

    def select_action(self, state: int, evaluate: bool = False) -> int:
        return int(self.rng.integers(0, self.n_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass
