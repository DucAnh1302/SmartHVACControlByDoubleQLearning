"""
Reward function module for SmartHVACControlByDoubleQLearning.
Balances Thermal Comfort (ASHRAE 55) and Energy Conservation.
"""

from typing import Dict, Tuple


class HVACRewardFunction:
    """
    Multi-objective reward calculator for HVAC control.
    
    Reward = - (w_comfort * ComfortPenalty + w_energy * EnergyCost)
    """

    def __init__(
        self,
        t_comfort_min: float = 22.0,
        t_comfort_max: float = 24.5,
        t_target: float = 23.2,
        w_comfort: float = 1.2,
        w_energy: float = 0.35,
        workhour_penalty_multiplier: float = 2.0,
        peak_energy_multiplier: float = 1.3
    ):
        self.t_comfort_min = t_comfort_min
        self.t_comfort_max = t_comfort_max
        self.t_target = t_target
        self.w_comfort = w_comfort
        self.w_energy = w_energy
        self.workhour_penalty_multiplier = workhour_penalty_multiplier
        self.peak_energy_multiplier = peak_energy_multiplier

    def calculate_reward(
        self,
        temp_c: float,
        hvac_kw: float,
        is_workhour: int,
        hour: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total reward and detailed breakdown.
        
        Parameters:
        -----------
        temp_c: Current indoor temperature (°C)
        hvac_kw: HVAC power applied in current step (kW)
        is_workhour: 1 if work hours, 0 otherwise
        hour: Current hour of the day (0-23)
        
        Returns:
        --------
        (total_reward, breakdown_dict)
        """
        # 1. Thermal Comfort Penalty
        # If within comfort range [22.0, 24.5], penalty is 0 and agent receives a small stability bonus
        if self.t_comfort_min <= temp_c <= self.t_comfort_max:
            comfort_penalty = 0.0
            comfort_bonus = 0.5
            is_comfortable = 1.0
        else:
            if temp_c < self.t_comfort_min:
                dev = self.t_comfort_min - temp_c
            else:
                dev = temp_c - self.t_comfort_max
            
            # Quadratic penalty for larger deviations
            comfort_penalty = (dev ** 1.8)
            comfort_bonus = 0.0
            is_comfortable = 0.0

        # Occupancy weight: during work hours, comfort penalty is magnified
        occ_mult = self.workhour_penalty_multiplier if is_workhour == 1 else 0.5
        scaled_comfort_penalty = comfort_penalty * occ_mult

        # 2. Energy Consumption Cost
        # Peak electricity hours: 12:00 - 18:00
        is_peak = 1.0 if (12 <= hour < 18 and is_workhour == 1) else 0.0
        rate_mult = self.peak_energy_multiplier if is_peak else 1.0

        # Normalized energy cost (scaled so ~20kW is comparable to comfort penalty)
        energy_cost = (hvac_kw / 10.0) * rate_mult

        # 3. Total Weighted Reward
        total_reward = -(self.w_comfort * scaled_comfort_penalty + self.w_energy * energy_cost) + comfort_bonus

        breakdown = {
            "comfort_penalty": float(scaled_comfort_penalty),
            "energy_cost": float(energy_cost),
            "comfort_bonus": float(comfort_bonus),
            "is_comfortable": float(is_comfortable),
            "total_reward": float(total_reward)
        }

        return total_reward, breakdown
