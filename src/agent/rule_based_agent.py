"""
Rule-Based Thermostat Controller (Baseline) for Smart HVAC.
"""

from typing import Tuple
from src.agent.base_agent import BaseAgent
from src.data.preprocess import index_to_state


class RuleBasedAgent(BaseAgent):
    """
    Standard Commercial Thermostat / Bang-Bang Rule-Based Controller.
    Decides HVAC power according to temperature setpoints and occupancy hours.
    """

    def __init__(self, n_states: int, n_actions: int, name: str = "Rule-Based"):
        super().__init__(n_states, n_actions, name)

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Rule-based decision:
        - Temp > 26.0°C (bin 4): HIGH cooling (Action 3)
        - Temp 24.5 - 26.0°C (bin 3): MED cooling (Action 2)
        - Temp 22.5 - 24.5°C (bin 2):
            - If daytime/workhour (time_bin 1 or 2): LOW cooling (Action 1)
            - If night (time_bin 0): OFF (Action 0)
        - Temp < 22.5°C (bin 0 or 1): OFF (Action 0)
        """
        temp_bin, load_bin, time_bin = index_to_state(state)

        if temp_bin >= 4:
            return 3  # HIGH (38 kW)
        elif temp_bin == 3:
            return 2  # MED (22 kW)
        elif temp_bin == 2:
            return 1 if time_bin > 0 else 0  # LOW (10 kW) during day, OFF at night
        else:
            return 0  # OFF (0 kW)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """Static rule-based agent does not require updates."""
        pass
