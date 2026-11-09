"""
Advanced Multi-Objective Reward Function for Smart HVAC Control.
Implements the approved scientific specification:
- 24.0°C - 26.0°C preferred comfort region
- Gentle penalties in 23-24°C and 26-27°C
- Non-linear severe penalties below 21°C or above 29°C (uncompromised by energy saving)
- Humidity comfort penalties (ASHRAE 55)
- Grid tariff peak energy pricing
- Equipment cycling / switching penalty
"""

from typing import Dict, Tuple


class AdvancedHVACRewardFunction:
    """
    Reward = - (w_comfort * ComfortPenalty + w_humidity * HumidityPenalty +
                w_energy * EnergyCost + w_switch * SwitchPenalty) + ComfortBonus
    """

    def __init__(
        self,
        t_pref_min: float = 24.0,
        t_pref_max: float = 26.0,
        t_safe_min: float = 21.0,
        t_safe_max: float = 29.0,
        rh_pref_min: float = 40.0,
        rh_pref_max: float = 65.0,
        w_comfort: float = 1.0,
        w_humidity: float = 0.2,
        w_energy: float = 0.08,
        w_switch: float = 0.30,
        workhour_mult: float = 2.0,
        peak_tariff_mult: float = 1.4
    ):
        self.t_pref_min = t_pref_min
        self.t_pref_max = t_pref_max
        self.t_safe_min = t_safe_min
        self.t_safe_max = t_safe_max
        self.rh_pref_min = rh_pref_min
        self.rh_pref_max = rh_pref_max
        self.w_comfort = w_comfort
        self.w_humidity = w_humidity
        self.w_energy = w_energy
        self.w_switch = w_switch
        self.workhour_mult = workhour_mult
        self.peak_tariff_mult = peak_tariff_mult

    def calculate_reward(
        self,
        temp_c: float,
        humidity_rh: float,
        hvac_kw: float,
        is_workhour: int,
        hour: int,
        action_changed: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total scalar reward and component diagnostics.
        """
        # 1. Temperature Comfort Penalty
        comfort_bonus = 0.0
        if self.t_pref_min <= temp_c <= self.t_pref_max:
            # Preferred comfort zone: no penalty, positive stability bonus
            t_penalty = 0.0
            comfort_bonus = 0.50
        elif 23.0 <= temp_c < self.t_pref_min:
            # Mild cool: 23-24°C
            dev = self.t_pref_min - temp_c
            t_penalty = 0.5 * dev
        elif self.t_pref_max < temp_c <= 27.0:
            # Mild warm: 26-27°C
            dev = temp_c - self.t_pref_max
            t_penalty = 0.5 * dev
        elif self.t_safe_min <= temp_c < 23.0:
            # Moderately cold: 21-23°C
            dev = 23.0 - temp_c
            t_penalty = 0.5 + 1.2 * (dev ** 1.4)
        elif 27.0 < temp_c <= self.t_safe_max:
            # Moderately warm: 27-29°C
            dev = temp_c - 27.0
            t_penalty = 0.5 + 1.2 * (dev ** 1.4)
        elif temp_c < self.t_safe_min:
            # Severe cold violation (< 21°C)
            dev = self.t_safe_min - temp_c
            t_penalty = 5.0 + 3.0 * (dev ** 2.0)
        else:
            # Severe hot violation (> 29°C)
            dev = temp_c - self.t_safe_max
            t_penalty = 5.0 + 3.0 * (dev ** 2.0)

        # Scale thermal comfort by occupancy/workhour
        scaled_t_penalty = t_penalty * (self.workhour_mult if is_workhour == 1 else 0.5)

        # 2. Humidity Comfort Penalty
        rh_penalty = 0.0
        if humidity_rh > self.rh_pref_max:
            rh_penalty = 0.05 * (humidity_rh - self.rh_pref_max)
        elif humidity_rh < self.rh_pref_min:
            rh_penalty = 0.05 * (self.rh_pref_min - humidity_rh)

        # 3. Energy Consumption Cost
        # Peak tariff between 09:00 and 17:00 on weekdays
        is_peak = (9 <= hour < 17) and (is_workhour == 1)
        tariff = self.peak_tariff_mult if is_peak else 1.0
        energy_cost = hvac_kw * tariff

        # 4. Action Switching Cost (prevents compressor hunting/wear)
        switch_cost = 1.0 if action_changed else 0.0

        # Total multi-objective reward
        total_penalty = (
            self.w_comfort * scaled_t_penalty +
            self.w_humidity * rh_penalty +
            self.w_energy * energy_cost +
            self.w_switch * switch_cost
        )
        total_reward = -total_penalty + comfort_bonus

        breakdown = {
            "comfort_penalty": float(scaled_t_penalty),
            "humidity_penalty": float(rh_penalty),
            "energy_cost": float(energy_cost),
            "switch_cost": float(switch_cost),
            "comfort_bonus": float(comfort_bonus),
            "total_reward": float(total_reward),
            "is_comfortable": float(1.0 if self.t_pref_min <= temp_c <= self.t_pref_max else 0.0),
            "is_severe_violation": float(1.0 if (temp_c < self.t_safe_min or temp_c > self.t_safe_max) else 0.0)
        }

        return float(total_reward), breakdown
