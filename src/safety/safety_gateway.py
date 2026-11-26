"""
Safety Gateway and Deployment Readiness Validation Layer.
Implements hard physical safety constraints, short-cycling prevention,
emergency overrides, and explicit deployment readiness certification.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
import math
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


@dataclass(frozen=True)
class SafetyConfig:
    """
    Authoritative, immutable safety configuration for the HVAC Safety Gateway.
    Defines hard physical constraints and sensor validation boundaries.
    Note: These algorithmic rules operate at the application/software layer and do
    not replace certified physical hardware safety interlocks (e.g. UL 60730-1, NFPA 90A).
    """
    min_cycles_between_toggles: int = 2      # Anti-short-cycling: compressor toggle locked for at least 2 steps (2 hours)
    emergency_freeze_temp: float = 18.0       # T < 18°C -> Force AC_OFF
    emergency_overheat_temp: float = 29.5     # T > 29.5°C -> Force Max Cooling (SET_23)
    window_max_ambient_temp: float = 33.0     # Outdoor T > 33°C -> Window prohibit
    window_max_ambient_rh: float = 85.0       # Outdoor RH > 85% (rain/storm) -> Window prohibit
    night_security_start_hour: int = 22       # 22:00 -> Window prohibit
    night_security_end_hour: int = 6          # 06:00 -> Window prohibit
    sensor_valid_min_temp: float = 10.0       # Hardware reading < 10°C is treated as sensor fault
    sensor_valid_max_temp: float = 45.0       # Hardware reading > 45°C is treated as sensor fault


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
    Possesses veto power to override unsafe actions before commands reach equipment.
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        min_cycles_between_toggles: Optional[int] = None,
        emergency_freeze_temp: Optional[float] = None,
        emergency_overheat_temp: Optional[float] = None,
        window_max_ambient_temp: Optional[float] = None,
        window_max_ambient_rh: Optional[float] = None
    ):
        if config is not None:
            self.config = config
        else:
            # Build from provided parameters or defaults
            self.config = SafetyConfig(
                min_cycles_between_toggles=min_cycles_between_toggles if min_cycles_between_toggles is not None else 2,
                emergency_freeze_temp=emergency_freeze_temp if emergency_freeze_temp is not None else 18.0,
                emergency_overheat_temp=emergency_overheat_temp if emergency_overheat_temp is not None else 29.5,
                window_max_ambient_temp=window_max_ambient_temp if window_max_ambient_temp is not None else 33.0,
                window_max_ambient_rh=window_max_ambient_rh if window_max_ambient_rh is not None else 85.0
            )

        # Expose properties for backwards compatibility
        self.min_cycles = self.config.min_cycles_between_toggles
        self.emergency_freeze_temp = self.config.emergency_freeze_temp
        self.emergency_overheat_temp = self.config.emergency_overheat_temp
        self.window_max_ambient_temp = self.config.window_max_ambient_temp
        self.window_max_ambient_rh = self.config.window_max_ambient_rh

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
        # 0. Sensor Fault / Anomaly Detection Fail-Safe
        if math.isnan(indoor_temp) or indoor_temp < self.config.sensor_valid_min_temp or indoor_temp > self.config.sensor_valid_max_temp:
            safe_act = 0  # Fallback to safe moderate setpoint 25°C
            self.interventions.append(
                SafetyIntervention(step, agent_action, safe_act, "SENSOR_ANOMALY_FAILSAFE", indoor_temp, ambient_temp)
            )
            self._update_state(safe_act)
            return safe_act, f"SENSOR_ANOMALY_FAILSAFE: Indoor temperature reading ({indoor_temp}) is NaN or out of physical bounds [{self.config.sensor_valid_min_temp}, {self.config.sensor_valid_max_temp}]°C."

        # 1. Extreme Overheat Emergency: If room > emergency_overheat_temp, force SET_23 (Action 1)
        if indoor_temp >= self.config.emergency_overheat_temp and agent_action in [3, 4, 2]:
            self.interventions.append(
                SafetyIntervention(step, agent_action, 1, "EMERGENCY_OVERHEAT_OVERRIDE", indoor_temp, ambient_temp)
            )
            self._update_state(1)
            return 1, f"EMERGENCY_OVERHEAT: Room exceeds {self.config.emergency_overheat_temp}°C; forcing SET_23 max cooling."

        # 2. Extreme Cold Emergency: If room < emergency_freeze_temp, force AC_OFF (Action 3)
        if indoor_temp <= self.config.emergency_freeze_temp and agent_action in [0, 1, 2]:
            self.interventions.append(
                SafetyIntervention(step, agent_action, 3, "EMERGENCY_FREEZE_PREVENTION", indoor_temp, ambient_temp)
            )
            self._update_state(3)
            return 3, f"EMERGENCY_FREEZE: Room below {self.config.emergency_freeze_temp}°C; forcing AC_OFF."

        # 3. Window Open Safety: Cannot open window during scorching heat, rain, or night security hours
        if agent_action == 4:  # WINDOW_OPEN
            # Security hours: 22:00 to 06:00
            if hour >= self.config.night_security_start_hour or hour < self.config.night_security_end_hour:
                safe_act = 3  # Fallback to AC_OFF with window closed
                self.interventions.append(
                    SafetyIntervention(step, agent_action, safe_act, "NIGHT_SECURITY_WINDOW_PROHIBITION", indoor_temp, ambient_temp)
                )
                self._update_state(safe_act)
                return safe_act, "WINDOW_SAFETY: Night security hours; window opening prohibited."

            # Weather checks
            if ambient_temp > self.config.window_max_ambient_temp or ambient_rh > self.config.window_max_ambient_rh:
                safe_act = 0  # Fallback to SET_25 closed
                self.interventions.append(
                    SafetyIntervention(step, agent_action, safe_act, "WEATHER_HAZARD_WINDOW_PROHIBITION", indoor_temp, ambient_temp)
                )
                self._update_state(safe_act)
                return safe_act, f"WINDOW_SAFETY: Outdoor condition ({ambient_temp:.1f}°C, {ambient_rh:.1f}%RH) hazardous."

        # 4. Compressor Short-Cycle Protection: Avoid toggling AC_OFF <-> Cooling every hour
        is_agent_off = (agent_action in [3, 4])
        is_last_off = (self.last_action in [3, 4])
        if is_agent_off != is_last_off and self.steps_in_current_state < self.config.min_cycles_between_toggles:
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
