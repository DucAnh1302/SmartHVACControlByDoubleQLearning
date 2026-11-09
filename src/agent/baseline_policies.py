"""
Baseline Policies for Smart HVAC Control Comparison.
Provides standardized, reproducible baselines:
1. RandomPolicy: Uniform stochastic baseline
2. FixedSetpointPolicy: Constant 25.0°C setpoint
3. ComfortFirstPolicy: Aggressive cooling (SET_23) whenever warm
4. EnergySavingPolicy: Economy setpoint (SET_27) and night setback
5. LegacyDoubleQLearningAdapter: Wraps legacy 45-state DQL model for comparison
"""

from typing import Optional
import numpy as np

from src.agent.base_agent import BaseAgent
from src.environment.target_env import index_to_state_tuple_600


class RandomPolicy(BaseAgent):
    """Selects an action uniformly at random."""

    def __init__(self, n_states: int = 600, n_actions: int = 5, seed: int = 42):
        super().__init__(n_states, n_actions, name="Random Policy")
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, evaluate: bool = False) -> int:
        return int(self.rng.integers(0, self.n_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class FixedSetpointPolicy(BaseAgent):
    """Always maintains a constant setpoint of 25.0°C (Action 0: SET_25)."""

    def __init__(self, n_states: int = 600, n_actions: int = 5):
        super().__init__(n_states, n_actions, name="Fixed Setpoint (25°C)")

    def select_action(self, state: int, evaluate: bool = False) -> int:
        return 0  # SET_25

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class ComfortFirstPolicy(BaseAgent):
    """
    Prioritizes thermal comfort above all energy costs:
    - If temp >= 24.0°C: Boost cool (SET_23, Action 1)
    - If temp 22.0 - 24.0°C: Maintain (SET_25, Action 0)
    - If temp < 22.0°C: Turn OFF (AC_OFF, Action 3)
    """

    def __init__(self, n_states: int = 600, n_actions: int = 5):
        super().__init__(n_states, n_actions, name="Comfort-First Policy")

    def select_action(self, state: int, evaluate: bool = False) -> int:
        t_bin, rh_bin, load_bin, time_bin = index_to_state_tuple_600(state)
        # t_bin: 0:<20, 1:20-21, 2:21-22, 3:22-23, 4:23-24, 5:24-25, 6:25-26, 7:26-27, 8:27-28.5, 9:>28.5
        if t_bin >= 5:
            return 1  # SET_23 (aggressive cool)
        elif t_bin >= 3:
            return 0  # SET_25 (maintain comfort)
        else:
            return 3  # AC_OFF (prevent overcooling)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class EnergySavingPolicy(BaseAgent):
    """
    Prioritizes electricity reduction:
    - Night off-peak or weekend (time_bin 0 or 4): AC_OFF (Action 3)
    - Daytime occupied: High setpoint (SET_27, Action 2)
    - If extremely warm (> 28.5°C, t_bin 9): SET_25 (Action 0)
    """

    def __init__(self, n_states: int = 600, n_actions: int = 5):
        super().__init__(n_states, n_actions, name="Energy-Saving Policy")

    def select_action(self, state: int, evaluate: bool = False) -> int:
        t_bin, rh_bin, load_bin, time_bin = index_to_state_tuple_600(state)

        # Unoccupied night or weekend
        if time_bin == 0 or time_bin == 4:
            return 3  # AC_OFF

        # During workday:
        if t_bin >= 9:
            return 0  # SET_25 when dangerously hot
        elif t_bin >= 6:
            return 2  # SET_27 (economy)
        else:
            return 3  # AC_OFF if already mild

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class ACOffPolicy(BaseAgent):
    """
    Lower-bound baseline: Turns AC completely OFF (Action 3: AC_OFF) at all times.
    Measures passive building thermal inertia and provides the zero-energy cost reference.
    """

    def __init__(self, n_states: int = 600, n_actions: int = 5):
        super().__init__(n_states, n_actions, name="AC-Always-Off Baseline")

    def select_action(self, state: int, evaluate: bool = False) -> int:
        return 3  # AC_OFF

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class RuleBasedThermostatPolicy(BaseAgent):
    """
    Standard commercial deadband thermostat controller:
    - If indoor temp > 25.5°C (t_bin >= 6): Cooling boost (SET_23, Action 1)
    - If indoor temp < 23.5°C (t_bin <= 3): Overcooling protection (AC_OFF, Action 3)
    - Otherwise within deadband (23.5 - 25.5°C): Maintain (SET_25, Action 0)
    """

    def __init__(self, n_states: int = 600, n_actions: int = 5):
        super().__init__(n_states, n_actions, name="Rule-Based Thermostat")

    def select_action(self, state: int, evaluate: bool = False) -> int:
        t_bin, rh_bin, load_bin, time_bin = index_to_state_tuple_600(state)
        if t_bin >= 6:
            return 1  # SET_23
        elif t_bin <= 3:
            return 3  # AC_OFF
        else:
            return 0  # SET_25

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass


class LegacyDQLAdapter(BaseAgent):
    """
    Adapts legacy 45-state Double Q-Learning agent to the 600-state environment.
    Maps legacy action output (0: OFF, 1: LOW, 2: MED, 3: HIGH) to target actions.
    """

    def __init__(self, legacy_agent: BaseAgent):
        super().__init__(n_states=600, n_actions=5, name="Legacy DQL (45 states)")
        self.legacy_agent = legacy_agent

    def select_action(self, state: int, evaluate: bool = False) -> int:
        t_bin_600, rh_bin, load_bin_600, time_bin_600 = index_to_state_tuple_600(state)
        # Map 10 temp bins to 5 legacy bins
        t_bin_45 = min(4, t_bin_600 // 2)
        load_bin_45 = min(2, load_bin_600)
        time_bin_45 = 1 if time_bin_600 in [1, 2] else (2 if time_bin_600 == 3 else 0)

        legacy_state = t_bin_45 * 9 + load_bin_45 * 3 + time_bin_45
        legacy_action = self.legacy_agent.select_action(legacy_state, evaluate=evaluate)

        # Map legacy action (0: OFF, 1: LOW, 2: MED, 3: HIGH) to target actions (0: SET_25, 1: SET_23, 2: SET_27, 3: AC_OFF)
        mapping = {
            0: 3,  # AC_OFF
            1: 2,  # SET_27 (Economy)
            2: 0,  # SET_25 (Standard)
            3: 1   # SET_23 (Boost)
        }
        return mapping.get(legacy_action, 0)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        pass
