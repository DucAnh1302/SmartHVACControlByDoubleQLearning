"""
Building Indoor Thermal Dynamic Model for CU-BEMS Smart HVAC simulation.
Based on 1st-order Resistance-Capacitance (RC) heat transfer calibrated with empirical data.
"""

from typing import Tuple
import numpy as np


class BuildingThermalModel:
    """
    Calibrated RC Thermal Model:
    dT/dt = alpha_amb * (T_amb - T_in) + beta_int * P_int - gamma_cool * P_hvac + noise
    """

    def __init__(
        self,
        alpha_amb: float = 0.20,      # Ambient thermal conductance / building thermal capacitance
        beta_int: float = 0.052,      # Internal heat gain coefficient (°C/kW)
        gamma_cool: float = 0.038,    # HVAC cooling capacity coefficient (°C/kW)
        noise_std: float = 0.12,      # Stochastic noise std (°C)
        min_temp: float = 16.0,       # Physical lower bound (°C)
        max_temp: float = 38.0        # Physical upper bound (°C)
    ):
        self.alpha_amb = alpha_amb
        self.beta_int = beta_int
        self.gamma_cool = gamma_cool
        self.noise_std = noise_std
        self.min_temp = min_temp
        self.max_temp = max_temp

    def step(
        self,
        current_temp: float,
        hvac_kw: float,
        internal_heat_kw: float,
        ambient_temp: float,
        add_noise: bool = True
    ) -> float:
        """
        Compute next indoor temperature after applying HVAC power for one time step (1 hour).
        
        Parameters:
        -----------
        current_temp: T_in at current step (°C)
        hvac_kw: HVAC power output (kW)
        internal_heat_kw: Total plug + lighting power (kW)
        ambient_temp: Outdoor / sol-air ambient temperature (°C)
        add_noise: Whether to add random thermal disturbance
        
        Returns:
        --------
        next_temp: Predicted indoor temperature at step t+1 (°C)
        """
        # 1. Heat transfer from outdoor ambient
        heat_from_ambient = self.alpha_amb * (ambient_temp - current_temp)

        # 2. Internal heat gain (equipment + occupants)
        heat_from_internal = self.beta_int * internal_heat_kw

        # 3. Cooling heat extraction from HVAC
        cooling_extracted = self.gamma_cool * hvac_kw

        # 4. Temperature variation
        delta_t = heat_from_ambient + heat_from_internal - cooling_extracted

        if add_noise and self.noise_std > 0:
            delta_t += np.random.normal(0.0, self.noise_std)

        next_temp = current_temp + delta_t
        return float(np.clip(next_temp, self.min_temp, self.max_temp))
