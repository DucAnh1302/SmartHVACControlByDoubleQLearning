"""
Multi-Device Safety Gateway and Interlock Layer for Smart HVAC Control.
Enforces hard physical constraints, anti-short-cycling compressor locks,
emergency hazard overrides (temperature, CO2, outdoor particulate pollution),
and interlock protections independent of the RL learning agent.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
import math
import numpy as np

from src.environment.multi_device_env import MULTI_DEVICE_ACTION_TEMPLATES, ActionTemplate


@dataclass(frozen=True)
class MultiDeviceSafetyConfig:
    """Authoritative physical safety boundaries and operational constraints."""
    min_compressor_dwell_steps: int = 2        # Prevent compressor toggling within 2 steps
    emergency_freeze_temp: float = 18.0         # T <= 18°C -> Force Heating / shutdown cooling
    emergency_overheat_temp: float = 29.5       # T >= 29.5°C -> Force MAX_COOL
    emergency_co2_ppm: float = 1600.0           # CO2 >= 1600 ppm -> Force ACTIVE_VENT_PURIFY
    window_max_ambient_temp: float = 33.0       # T_amb > 33°C -> Window veto
    window_max_ambient_rh: float = 85.0         # RH_amb > 85% (rain/storm) -> Window veto
    window_max_ambient_pm25: float = 60.0       # PM2.5_amb > 60 µg/m³ (smog) -> Window veto
    night_security_start_hour: int = 22         # 22:00 -> Window veto
    night_security_end_hour: int = 6            # 06:00 -> Window veto
    sensor_valid_min_temp: float = 5.0          # Sensor fault detection
    sensor_valid_max_temp: float = 50.0


class MultiDeviceIntervention(NamedTuple):
    timestamp_step: int
    original_action: int
    overridden_action: int
    rule_violated: str
    reason: str
    indoor_temp: float
    indoor_co2: float
    ambient_temp: float


class MultiDeviceSafetyGateway:
    """
    Downstream safety gateway with unilateral override authority over agent actions.
    Ensures safe operation across all 7 HVAC equipment categories.
    """

    def __init__(self, config: Optional[MultiDeviceSafetyConfig] = None):
        self.config = config or MultiDeviceSafetyConfig()
        self.last_action: Optional[int] = None
        self.steps_in_current_state: int = 0
        self.interventions: List[MultiDeviceIntervention] = []

    def reset(self) -> None:
        """Reset gateway state for a new episode."""
        self.last_action = None
        self.steps_in_current_state = 0
        self.interventions.clear()

    def filter_action(
        self,
        agent_action: int,
        indoor_temp: float,
        indoor_rh: float,
        indoor_co2: float,
        indoor_pm25: float,
        ambient_temp: float,
        ambient_rh: float,
        ambient_pm25: float,
        hour: int,
        step: int
    ) -> Tuple[int, Optional[str]]:
        """
        Evaluate agent proposed action against hard safety rules and return:
        (executed_safe_action, override_reason_or_None)
        """
        # 0. Sensor Anomaly Fail-safe
        if (
            math.isnan(indoor_temp) or
            indoor_temp < self.config.sensor_valid_min_temp or
            indoor_temp > self.config.sensor_valid_max_temp or
            math.isnan(indoor_co2) or
            indoor_co2 < 200.0 or
            math.isnan(indoor_pm25)
        ):
            safe_act = 1  # COMFORT_COOL safe default
            reason = (
                f"SENSOR_ANOMALY_FAILSAFE: Transducer reading out of plausible bounds "
                f"(T={indoor_temp}, CO2={indoor_co2}, PM2.5={indoor_pm25}). Reverting to COMFORT_COOL."
            )
            self._record(step, agent_action, safe_act, "SENSOR_ANOMALY_FAILSAFE", reason, indoor_temp, indoor_co2, ambient_temp)
            self._update_state(safe_act)
            return safe_act, reason

        # 1. Emergency Overheat Override (T >= 29.5°C) -> Force MAX_COOL (Action 2)
        if indoor_temp >= self.config.emergency_overheat_temp and agent_action != 2:
            safe_act = 2  # MAX_COOL
            reason = (
                f"EMERGENCY_OVERHEAT: Room temperature {indoor_temp:.1f}°C exceeds safe threshold "
                f"{self.config.emergency_overheat_temp}°C; overriding to MAX_COOL."
            )
            self._record(step, agent_action, safe_act, "EMERGENCY_OVERHEAT_OVERRIDE", reason, indoor_temp, indoor_co2, ambient_temp)
            self._update_state(safe_act)
            return safe_act, reason

        # 2. Emergency Freeze Override (T <= 18.0°C) -> Force COMFORT_HEAT (Action 6) or Standby
        if indoor_temp <= self.config.emergency_freeze_temp and agent_action in [1, 2, 3]:
            safe_act = 6  # COMFORT_HEAT
            reason = (
                f"EMERGENCY_FREEZE: Room temperature {indoor_temp:.1f}°C is critically cold (<= {self.config.emergency_freeze_temp}°C); "
                f"prohibiting cooling and forcing COMFORT_HEAT."
            )
            self._record(step, agent_action, safe_act, "EMERGENCY_FREEZE_PREVENTION", reason, indoor_temp, indoor_co2, ambient_temp)
            self._update_state(safe_act)
            return safe_act, reason

        # 3. Emergency High CO2 Flush (CO2 >= 1600 ppm) -> Force ACTIVE_VENT_PURIFY (Action 5)
        if indoor_co2 >= self.config.emergency_co2_ppm and agent_action not in [2, 5]:
            safe_act = 5  # ACTIVE_VENT_PURIFY
            reason = (
                f"EMERGENCY_CO2_FLUSH: Indoor CO2 concentration {indoor_co2:.0f} ppm exceeds hazard limit "
                f"{self.config.emergency_co2_ppm:.0f} ppm; forcing ACTIVE_VENT_PURIFY."
            )
            self._record(step, agent_action, safe_act, "EMERGENCY_CO2_FLUSH", reason, indoor_temp, indoor_co2, ambient_temp)
            self._update_state(safe_act)
            return safe_act, reason

        # 4. Window Environmental Hazards (Action 4: NATURAL_VENTILATION)
        if agent_action == 4:
            # 4a. Night security
            if hour >= self.config.night_security_start_hour or hour < self.config.night_security_end_hour:
                safe_act = 11  # NIGHT_SETBACK with window closed
                reason = "WINDOW_SAFETY: Night security hours (22:00-06:00); window opening prohibited."
                self._record(step, agent_action, safe_act, "NIGHT_SECURITY_WINDOW_VETO", reason, indoor_temp, indoor_co2, ambient_temp)
                self._update_state(safe_act)
                return safe_act, reason

            # 4b. Ambient temperature too hot
            if ambient_temp > self.config.window_max_ambient_temp:
                safe_act = 1  # COMFORT_COOL
                reason = f"WINDOW_SAFETY: Outdoor temperature {ambient_temp:.1f}°C exceeds {self.config.window_max_ambient_temp}°C; window opening prohibited."
                self._record(step, agent_action, safe_act, "WEATHER_HEAT_WINDOW_VETO", reason, indoor_temp, indoor_co2, ambient_temp)
                self._update_state(safe_act)
                return safe_act, reason

            # 4c. Ambient high humidity / rain
            if ambient_rh > self.config.window_max_ambient_rh:
                safe_act = 1  # COMFORT_COOL
                reason = f"WINDOW_SAFETY: Outdoor humidity {ambient_rh:.1f}% exceeds {self.config.window_max_ambient_rh}%; rain hazard window veto."
                self._record(step, agent_action, safe_act, "WEATHER_RAIN_WINDOW_VETO", reason, indoor_temp, indoor_co2, ambient_temp)
                self._update_state(safe_act)
                return safe_act, reason

            # 4d. Ambient outdoor PM2.5 pollution
            if ambient_pm25 > self.config.window_max_ambient_pm25:
                safe_act = 10  # ECO_AIR_PURIFY
                reason = f"WINDOW_SAFETY: Outdoor PM2.5 {ambient_pm25:.1f} µg/m³ exceeds {self.config.window_max_ambient_pm25} µg/m³; smog hazard window veto."
                self._record(step, agent_action, safe_act, "POLLUTION_SMOG_WINDOW_VETO", reason, indoor_temp, indoor_co2, ambient_temp)
                self._update_state(safe_act)
                return safe_act, reason

        # 5. Anti-Short-Cycling Protection: Lock compressor switching between cooling and off within min dwell time
        if self.last_action is not None and agent_action != self.last_action:
            prev_tmpl = MULTI_DEVICE_ACTION_TEMPLATES[self.last_action]
            curr_tmpl = MULTI_DEVICE_ACTION_TEMPLATES[agent_action]

            prev_active_ac = prev_tmpl.ac_mode != "OFF"
            curr_active_ac = curr_tmpl.ac_mode != "OFF"

            # Toggling compressor on/off state
            if prev_active_ac != curr_active_ac and self.steps_in_current_state < self.config.min_compressor_dwell_steps:
                safe_act = self.last_action
                reason = (
                    f"SHORT_CYCLE_PROTECTION: Compressor transition between active and idle requested after only "
                    f"{self.steps_in_current_state} step(s); locked in state {self.last_action} "
                    f"for minimum {self.config.min_compressor_dwell_steps} steps."
                )
                self._record(step, agent_action, safe_act, "SHORT_CYCLE_PROTECTION", reason, indoor_temp, indoor_co2, ambient_temp)
                self._update_state(safe_act)
                return safe_act, reason

        # Passed all safety checks
        self._update_state(agent_action)
        return agent_action, None

    def _update_state(self, action: int) -> None:
        if self.last_action == action:
            self.steps_in_current_state += 1
        else:
            self.last_action = action
            self.steps_in_current_state = 1

    def _record(
        self,
        step: int,
        orig_act: int,
        over_act: int,
        rule: str,
        reason: str,
        indoor_temp: float,
        indoor_co2: float,
        ambient_temp: float
    ) -> None:
        self.interventions.append(
            MultiDeviceIntervention(
                timestamp_step=step,
                original_action=orig_act,
                overridden_action=over_act,
                rule_violated=rule,
                reason=reason,
                indoor_temp=indoor_temp,
                indoor_co2=indoor_co2,
                ambient_temp=ambient_temp
            )
        )
