"""
Multi-Objective Decomposed Reward Function for Multi-Device HVAC Control.
Decomposes total reward into 7 independent signed physical components:
R_total = R_temp + R_humidity + R_co2 + R_air_quality + R_energy + R_switching + R_safety

Supports configurable stakeholder preference profiles:
- "balanced" (Default)
- "comfort_priority"
- "energy_priority"
- "air_quality_priority"
"""

from typing import Dict, Tuple, Literal, Optional, Any
import numpy as np


class MultiDeviceRewardFunction:
    """
    Multi-objective reward function with verifiable signed additive decomposition
    spanning thermal comfort, psychrometric humidity, CO2 ventilation, PM2.5 filtration,
    multi-device power costs, equipment wear/switching, and safety interlocks.
    """

    def __init__(
        self,
        t_pref_min: float = 24.0,
        t_pref_max: float = 26.0,
        t_safe_min: float = 21.0,
        t_safe_max: float = 29.0,
        rh_pref_min: float = 40.0,
        rh_pref_max: float = 65.0,
        co2_pref_max: float = 800.0,
        co2_safe_max: float = 1200.0,
        pm25_pref_max: float = 25.0,
        pm25_safe_max: float = 50.0,
        w_temp: float = 1.00,
        w_humidity: float = 0.25,
        w_co2: float = 0.30,
        w_pm25: float = 0.25,
        w_energy: float = 0.12,
        w_switch: float = 0.35,
        w_safety: float = 2.00,
        workhour_mult: float = 2.0,
        peak_tariff_mult: float = 1.4,
        preference: str = "balanced"
    ):
        self.t_pref_min = t_pref_min
        self.t_pref_max = t_pref_max
        self.t_safe_min = t_safe_min
        self.t_safe_max = t_safe_max
        self.rh_pref_min = rh_pref_min
        self.rh_pref_max = rh_pref_max
        self.co2_pref_max = co2_pref_max
        self.co2_safe_max = co2_safe_max
        self.pm25_pref_max = pm25_pref_max
        self.pm25_safe_max = pm25_safe_max

        self.w_temp = w_temp
        self.w_humidity = w_humidity
        self.w_co2 = w_co2
        self.w_pm25 = w_pm25
        self.w_energy = w_energy
        self.w_switch = w_switch
        self.w_safety = w_safety
        self.workhour_mult = workhour_mult
        self.peak_tariff_mult = peak_tariff_mult
        self.preference = preference

        if preference != "balanced":
            self.configure_weights(preference)

    def configure_weights(
        self,
        preference: Literal["balanced", "comfort_priority", "energy_priority", "air_quality_priority"]
    ) -> None:
        """Configure multi-objective weights according to stakeholder preference profile."""
        self.preference = preference
        if preference == "comfort_priority":
            self.w_temp = 2.00
            self.w_humidity = 0.40
            self.w_co2 = 0.35
            self.w_pm25 = 0.30
            self.w_energy = 0.04
            self.w_switch = 0.20
            self.w_safety = 2.00
        elif preference == "energy_priority":
            self.w_temp = 0.60
            self.w_humidity = 0.15
            self.w_co2 = 0.20
            self.w_pm25 = 0.15
            self.w_energy = 0.25
            self.w_switch = 0.50
            self.w_safety = 2.00
        elif preference == "air_quality_priority":
            self.w_temp = 0.90
            self.w_humidity = 0.25
            self.w_co2 = 0.70
            self.w_pm25 = 0.60
            self.w_energy = 0.08
            self.w_switch = 0.30
            self.w_safety = 2.00
        elif preference == "balanced":
            self.w_temp = 1.00
            self.w_humidity = 0.25
            self.w_co2 = 0.30
            self.w_pm25 = 0.25
            self.w_energy = 0.12
            self.w_switch = 0.35
            self.w_safety = 2.00
        else:
            raise ValueError(f"Unknown preference profile '{preference}'!")

    def calculate_reward(
        self,
        temp_c: float,
        humidity_rh: float,
        co2_ppm: float,
        pm25_ug_m3: float,
        total_power_kw: float,
        hour: int,
        dayofweek: int = 0,
        is_workhour: Optional[int] = None,
        action_changed: bool = False,
        safety_violation: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total scalar reward and decomposed signed components:
        R_total = R_temp + R_humidity + R_co2 + R_air_quality + R_energy + R_switching + R_safety
        """
        # Determine office work hours: Monday-Friday (0..4), 08:00 - 18:00 if not explicitly supplied
        if is_workhour is None:
            is_work = 1 if (dayofweek < 5 and 8 <= hour < 18) else 0
        else:
            is_work = int(is_workhour)

        occ_mult = self.workhour_mult if is_work == 1 else 0.5

        # 1. Thermal Comfort Component
        if self.t_pref_min <= temp_c <= self.t_pref_max:
            # Optimal comfort zone bonus
            r_temp = float(0.50 * occ_mult)
        elif self.t_safe_min <= temp_c < self.t_pref_min:
            dev = self.t_pref_min - temp_c
            r_temp = float(- self.w_temp * (0.4 * dev) * occ_mult)
        elif self.t_pref_max < temp_c <= self.t_safe_max:
            dev = temp_c - self.t_pref_max
            r_temp = float(- self.w_temp * (0.4 * dev) * occ_mult)
        else:
            # Extreme thermal discomfort (capped within safe range for r_temp, quadratic penalty goes to r_safety)
            r_temp = float(- self.w_temp * 2.0 * occ_mult)

        # 2. Psychrometric Humidity Component
        if self.rh_pref_min <= humidity_rh <= self.rh_pref_max:
            r_humidity = float(0.10 * occ_mult)
        elif humidity_rh < self.rh_pref_min:
            dev_rh = self.rh_pref_min - humidity_rh
            r_humidity = float(- self.w_humidity * (0.05 * dev_rh) * occ_mult)
        else:
            dev_rh = humidity_rh - self.rh_pref_max
            r_humidity = float(- self.w_humidity * (0.05 * dev_rh) * occ_mult)

        # 3. Indoor CO2 Ventilation Component (ASHRAE 62.1)
        if co2_ppm <= self.co2_pref_max:
            r_co2 = float(0.20 * occ_mult)
        elif co2_ppm <= self.co2_safe_max:
            dev_co2 = (co2_ppm - self.co2_pref_max) / 100.0
            r_co2 = float(- self.w_co2 * (0.25 * dev_co2) * occ_mult)
        else:
            dev_co2 = (co2_ppm - self.co2_safe_max) / 100.0
            r_co2 = float(- self.w_co2 * (1.0 + 0.6 * dev_co2) * occ_mult)

        # 4. Air Quality / PM2.5 Component (WHO guidelines)
        if pm25_ug_m3 <= 15.0:
            r_air_quality = float(0.20 * occ_mult)
        elif pm25_ug_m3 <= self.pm25_pref_max:
            r_air_quality = 0.0
        elif pm25_ug_m3 <= self.pm25_safe_max:
            dev_pm = pm25_ug_m3 - self.pm25_pref_max
            r_air_quality = float(- self.w_pm25 * (0.04 * dev_pm) * occ_mult)
        else:
            dev_pm = pm25_ug_m3 - self.pm25_safe_max
            r_air_quality = float(- self.w_pm25 * (1.0 + 0.08 * dev_pm) * occ_mult)

        # 5. Energy Consumption Component
        is_peak = (9 <= hour < 17) and (dayofweek < 5) and (is_workhour == 1)
        tariff = self.peak_tariff_mult if is_peak else 1.0
        energy_cost = total_power_kw * tariff
        r_energy = float(- self.w_energy * energy_cost)

        # 6. Switching / Hunting Component
        r_switching = float(- self.w_switch * (1.0 if action_changed else 0.0))

        # 7. Safety Component (Non-linear penalties for hazardous extremes)
        r_safety = 0.0
        if temp_c < self.t_safe_min:
            dev_freeze = self.t_safe_min - temp_c
            r_safety += float(-(5.0 + 2.5 * (dev_freeze ** 2.0)) * occ_mult)
        elif temp_c > self.t_safe_max:
            dev_heat = temp_c - self.t_safe_max
            r_safety += float(-(5.0 + 2.5 * (dev_heat ** 2.0)) * occ_mult)

        if co2_ppm > 1800.0:
            dev_c = (co2_ppm - 1800.0) / 100.0
            r_safety += float(-(4.0 + 2.0 * dev_c) * occ_mult)

        if pm25_ug_m3 > 100.0:
            dev_p = (pm25_ug_m3 - 100.0) / 10.0
            r_safety += float(-(3.0 + 1.5 * dev_p) * occ_mult)

        if safety_violation:
            r_safety += float(-5.0 * self.w_safety)

        # Additive decomposition sum
        total_reward = float(
            r_temp + r_humidity + r_co2 + r_air_quality + r_energy + r_switching + r_safety
        )

        breakdown = {
            "reward_temp": round(r_temp, 4),
            "reward_humidity": round(r_humidity, 4),
            "reward_co2": round(r_co2, 4),
            "reward_air_quality": round(r_air_quality, 4),
            "reward_energy": round(r_energy, 4),
            "reward_switching": round(r_switching, 4),
            "reward_safety": round(r_safety, 4),
            "total_reward": round(total_reward, 4),
            "energy_cost": round(energy_cost, 4),
            "is_comfortable_temp": float(1.0 if self.t_pref_min <= temp_c <= self.t_pref_max else 0.0),
            "is_comfortable_humidity": float(1.0 if self.rh_pref_min <= humidity_rh <= self.rh_pref_max else 0.0),
            "is_fresh_co2": float(1.0 if co2_ppm <= self.co2_pref_max else 0.0),
            "is_clean_pm25": float(1.0 if pm25_ug_m3 <= self.pm25_pref_max else 0.0),
            "is_safety_violation": float(1.0 if r_safety < 0.0 else 0.0)
        }

        return total_reward, breakdown
