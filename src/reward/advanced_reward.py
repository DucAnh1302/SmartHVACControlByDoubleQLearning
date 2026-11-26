"""
Advanced Multi-Objective Reward Function for Smart HVAC Control.
Implements the approved scientific specification with signed component decomposition:
- Preferred zone [24.0°C, 26.0°C]: positive bonus (+0.50)
- Mild deviation [23-24°C] and [26-27°C]: gentle linear penalties
- Moderate deviation [21-23°C] and [27-29°C]: non-linear penalty
- Severe thermal safety violations (<21°C or >29°C): isolated quadratic safety penalty
- Humidity comfort penalties (ASHRAE 55)
- Grid tariff peak energy pricing
- Equipment cycling / switching penalty

Enforces signed additive decomposition:
R_total = R_comfort + R_safety + R_energy + R_switching + R_humidity
"""

from typing import Dict, Tuple, Literal
import numpy as np


class AdvancedHVACRewardFunction:
    """
    Multi-objective reward function with verifiable signed additive decomposition.
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
        peak_tariff_mult: float = 1.4,
        preference: str = "balanced"
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
        self.preference = preference

        if preference != "balanced":
            self.configure_weights(preference)

    def configure_weights(self, preference: Literal["balanced", "comfort_priority", "energy_priority"]) -> None:
        """
        Configure multi-objective weights according to stakeholder preference profile.
        Safety penalties remain immutable across all profiles.
        """
        self.preference = preference
        if preference == "comfort_priority":
            self.w_comfort = 2.00
            self.w_energy = 0.03
            self.w_switch = 0.20
            self.w_humidity = 0.30
        elif preference == "energy_priority":
            self.w_comfort = 0.50
            self.w_energy = 0.18
            self.w_switch = 0.40
            self.w_humidity = 0.10
        elif preference == "balanced":
            self.w_comfort = 1.00
            self.w_energy = 0.08
            self.w_switch = 0.30
            self.w_humidity = 0.20
        else:
            raise ValueError(f"Unknown preference profile '{preference}'! Valid profiles: 'balanced', 'comfort_priority', 'energy_priority'.")

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
        Calculate total scalar reward and decomposed signed components:
        R_total = R_comfort + R_safety + R_energy + R_switching + R_humidity
        """
        occ_mult = self.workhour_mult if is_workhour == 1 else 0.5

        # 1. Comfort Component (within safe range [21, 29])
        comfort_bonus = 0.0
        mild_t_penalty = 0.0

        if self.t_pref_min <= temp_c <= self.t_pref_max:
            # Preferred comfort zone: positive stability bonus
            comfort_bonus = 0.50
            mild_t_penalty = 0.0
        elif 23.0 <= temp_c < self.t_pref_min:
            dev = self.t_pref_min - temp_c
            mild_t_penalty = 0.5 * dev
        elif self.t_pref_max < temp_c <= 27.0:
            dev = temp_c - self.t_pref_max
            mild_t_penalty = 0.5 * dev
        elif self.t_safe_min <= temp_c < 23.0:
            dev = 23.0 - temp_c
            mild_t_penalty = 0.5 + 1.2 * (dev ** 1.4)
        elif 27.0 < temp_c <= self.t_safe_max:
            dev = temp_c - 27.0
            mild_t_penalty = 0.5 + 1.2 * (dev ** 1.4)
        else:
            # Outside safe range: maximum standard discomfort penalty applies
            mild_t_penalty = 0.5 + 1.2 * (2.0 ** 1.4)

        r_comfort = float(comfort_bonus - self.w_comfort * mild_t_penalty * occ_mult)

        # 2. Safety Component (isolated non-linear penalty for severe excursions)
        r_safety = 0.0
        if temp_c < self.t_safe_min:
            dev = self.t_safe_min - temp_c
            # Heavy quadratic penalty for freezing hazard
            r_safety = float(-(5.0 + 3.0 * (dev ** 2.0)) * occ_mult)
        elif temp_c > self.t_safe_max:
            dev = temp_c - self.t_safe_max
            # Heavy quadratic penalty for overheating hazard
            r_safety = float(-(5.0 + 3.0 * (dev ** 2.0)) * occ_mult)

        # 3. Humidity Component
        rh_penalty = 0.0
        if humidity_rh > self.rh_pref_max:
            rh_penalty = 0.05 * (humidity_rh - self.rh_pref_max)
        elif humidity_rh < self.rh_pref_min:
            rh_penalty = 0.05 * (self.rh_pref_min - humidity_rh)
        r_humidity = float(- self.w_humidity * rh_penalty)

        # 4. Energy Component
        is_peak = (9 <= hour < 17) and (is_workhour == 1)
        tariff = self.peak_tariff_mult if is_peak else 1.0
        energy_cost = hvac_kw * tariff
        r_energy = float(- self.w_energy * energy_cost)

        # 5. Switching Component
        switch_cost = 1.0 if action_changed else 0.0
        r_switching = float(- self.w_switch * switch_cost)

        # Total signed sum
        total_reward = float(r_comfort + r_safety + r_energy + r_switching + r_humidity)

        breakdown = {
            # Signed components
            "reward_comfort": r_comfort,
            "reward_safety": r_safety,
            "reward_energy": r_energy,
            "reward_switching": r_switching,
            "reward_humidity": r_humidity,
            # Backwards-compatible legacy diagnostic keys
            "comfort_penalty": float(mild_t_penalty * occ_mult),
            "humidity_penalty": float(rh_penalty),
            "energy_cost": float(energy_cost),
            "switch_cost": float(switch_cost),
            "comfort_bonus": float(comfort_bonus),
            "total_reward": total_reward,
            "is_comfortable": float(1.0 if self.t_pref_min <= temp_c <= self.t_pref_max else 0.0),
            "is_severe_violation": float(1.0 if (temp_c < self.t_safe_min or temp_c > self.t_safe_max) else 0.0)
        }

        return total_reward, breakdown
