"""
Safety Gateway and Deployment Readiness Validation Layer.
Implements hard physical safety constraints, short-cycling prevention,
emergency overrides, and explicit deployment readiness certification.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import json
import numpy as np


class DeploymentReadinessLevel(str, Enum):
    """
    Tiered deployment classification to prevent unauthorized live actuation.
    """
    RESEARCH_SIMULATION_ONLY = "RESEARCH_SIMULATION_ONLY"
    OFFLINE_VALIDATED = "OFFLINE_VALIDATED"
    HARDWARE_IN_LOOP_VALIDATED = "HARDWARE_IN_LOOP_VALIDATED"
    SHADOW_MODE_VALIDATED = "SHADOW_MODE_VALIDATED"
    DEPLOYMENT_CANDIDATE = "DEPLOYMENT_CANDIDATE"


class SafetyIntervention(NamedTuple):
    timestamp_step: int
    original_action: int
    overridden_action: int
    rule_violated: str
    indoor_temp: float
    ambient_temp: float


class SafetyGateway:
    """
    Independent deterministic safety interlock operating downstream of the RL Agent.
    Possesses veto power to override unsafe actions before command reaches equipment.
    """

    def __init__(
        self,
        min_cycles_between_toggles: int = 2,  # Compressor short-cycling protection (at least 2 hours)
        emergency_freeze_temp: float = 18.0,  # Below 18°C -> Force heater / AC_OFF
        emergency_overheat_temp: float = 29.5, # Above 29.5°C -> Force Max Cooling (SET_23)
        window_max_ambient_temp: float = 33.0, # Cannot open window if outdoor is scorching (>33°C)
        window_max_ambient_rh: float = 85.0    # Cannot open window if raining / tropical storm (>85% RH)
    ):
        self.min_cycles = min_cycles_between_toggles
        self.emergency_freeze_temp = emergency_freeze_temp
        self.emergency_overheat_temp = emergency_overheat_temp
        self.window_max_ambient_temp = window_max_ambient_temp
        self.window_max_ambient_rh = window_max_ambient_rh

        # State tracking
        self.last_action: int = 0
        self.steps_in_current_state: int = 0
        self.interventions: List[SafetyIntervention] = []

    def filter_action(
        self,
        agent_action: int,
        indoor_temp: float,
        ambient_temp: float,
        ambient_rh: float,
        hour: int,
        step: int
    ) -> Tuple[int, Optional[str]]:
        """
        Evaluate agent action against hard safety constraints.
        Returns: (safe_action, reason_if_overridden)
        """
        # 1. Extreme Overheat Emergency: If room > 29.5°C, force SET_23 (Action 1)
        if indoor_temp >= self.emergency_overheat_temp and agent_action in [3, 4, 2]:
            self.interventions.append(
                SafetyIntervention(step, agent_action, 1, "EMERGENCY_OVERHEAT_OVERRIDE", indoor_temp, ambient_temp)
            )
            self._update_state(1)
            return 1, "EMERGENCY_OVERHEAT: Room exceeds 29.5°C; forcing SET_23 max cooling."

        # 2. Extreme Cold Emergency: If room < 18°C, force AC_OFF (Action 3)
        if indoor_temp <= self.emergency_freeze_temp and agent_action in [0, 1, 2]:
            self.interventions.append(
                SafetyIntervention(step, agent_action, 3, "EMERGENCY_FREEZE_PREVENTION", indoor_temp, ambient_temp)
            )
            self._update_state(3)
            return 3, "EMERGENCY_FREEZE: Room below 18.0°C; forcing AC_OFF."

        # 3. Window Open Safety: Cannot open window during scorching heat, rain, or night security hours
        if agent_action == 4:  # WINDOW_OPEN
            # Security hours: 22:00 to 06:00
            if (hour >= 22 or hour < 6):
                safe_act = 3  # Fallback to AC_OFF with window closed
                self.interventions.append(
                    SafetyIntervention(step, agent_action, safe_act, "NIGHT_SECURITY_WINDOW_PROHIBITION", indoor_temp, ambient_temp)
                )
                self._update_state(safe_act)
                return safe_act, "WINDOW_SAFETY: Night security hours; window opening prohibited."

            # Weather checks
            if ambient_temp > self.window_max_ambient_temp or ambient_rh > self.window_max_ambient_rh:
                safe_act = 0  # Fallback to SET_25 closed
                self.interventions.append(
                    SafetyIntervention(step, agent_action, safe_act, "WEATHER_HAZARD_WINDOW_PROHIBITION", indoor_temp, ambient_temp)
                )
                self._update_state(safe_act)
                return safe_act, f"WINDOW_SAFETY: Outdoor condition ({ambient_temp:.1f}°C, {ambient_rh:.1f}%RH) hazardous."

        # 4. Compressor Short-Cycle Protection: Avoid toggling AC_OFF <-> Cooling every hour
        is_agent_off = (agent_action in [3, 4])
        is_last_off = (self.last_action in [3, 4])
        if is_agent_off != is_last_off and self.steps_in_current_state < self.min_cycles:
            # Maintain previous operational state to protect physical compressor
            retained_action = self.last_action
            self.interventions.append(
                SafetyIntervention(step, agent_action, retained_action, "COMPRESSOR_SHORT_CYCLE_PREVENTION", indoor_temp, ambient_temp)
            )
            self.steps_in_current_state += 1
            return retained_action, "SHORT_CYCLE_PROTECTION: Action change deferred to prevent compressor damage."

        # Action accepted
        self._update_state(agent_action)
        return agent_action, None

    def _update_state(self, action: int) -> None:
        if action == self.last_action:
            self.steps_in_current_state += 1
        else:
            self.last_action = action
            self.steps_in_current_state = 1

    def reset(self) -> None:
        self.last_action = 0
        self.steps_in_current_state = 0
        self.interventions.clear()


class ShadowModeRunner:
    """
    Shadow Mode / Passive Recommendation Engine.
    Reads live/historical BMS observation stream, runs agent policy and safety gateway,
    and logs recommendation without commanding physical actuators.
    """

    def __init__(self, agent: Any, safety_gateway: Optional[SafetyGateway] = None):
        self.agent = agent
        self.safety_gateway = safety_gateway or SafetyGateway()
        self.recommendations: List[Dict[str, Any]] = []

    def process_bms_reading(
        self,
        timestamp: str,
        state_idx: int,
        indoor_temp: float,
        ambient_temp: float,
        ambient_rh: float,
        historical_hvac_kw: float,
        hour: int,
        step: int
    ) -> Dict[str, Any]:
        """Process one incoming BMS telemetry frame in shadow mode."""
        # 1. Agent policy query (greedy)
        raw_action = self.agent.select_action(state_idx, evaluate=True)

        # 2. Safety gateway check
        safe_action, override_reason = self.safety_gateway.filter_action(
            agent_action=raw_action,
            indoor_temp=indoor_temp,
            ambient_temp=ambient_temp,
            ambient_rh=ambient_rh,
            hour=hour,
            step=step
        )

        rec = {
            "timestamp": timestamp,
            "step": step,
            "indoor_temp": round(indoor_temp, 2),
            "historical_hvac_kw": round(historical_hvac_kw, 2),
            "agent_recommended_action": raw_action,
            "safety_approved_action": safe_action,
            "was_overridden": bool(raw_action != safe_action),
            "override_reason": override_reason,
            "shadow_mode_readiness": DeploymentReadinessLevel.SHADOW_MODE_VALIDATED.value
        }
        self.recommendations.append(rec)
        return rec
