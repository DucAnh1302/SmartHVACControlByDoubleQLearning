"""
Robustness and Uncertainty Suite for Smart HVAC Control.
Evaluates a frozen policy under 7 plant perturbations across multiple seeds:
1. Nominal Environment
2. High Thermal Conductance (k_env +35%)
3. Low Thermal Conductance (k_env -35%)
4. Degraded HVAC Cooling Efficiency (k_ac -25%)
5. Severe Sensor Noise (std = 0.50°C)
6. Outdoor Weather Heatwave Bias (+2.5°C ambient shift)
7. Equipment Calibration Drift (+25% power consumption)
"""

from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.agent.base_agent import BaseAgent
from src.environment.target_env import TargetHVACEnv
from src.environment.greybox_model import GreyBoxThermalModel, PhysicalParameters
from src.data.calibration import EquipmentStageCalibration
from src.evaluation.benchmark_suite import evaluate_policy_on_env


class RobustnessExperimentSuite:
    """
    Executes plant configuration perturbations on a frozen trained policy.
    """

    def __init__(self, agent: BaseAgent, base_env_data: pd.DataFrame, base_calibration: EquipmentStageCalibration):
        self.agent = agent
        self.data_df = base_env_data
        self.cal = base_calibration

    def run_robustness_test(
        self,
        seeds: List[int] = [42, 123, 999],
        n_episodes_per_seed: int = 20
    ) -> pd.DataFrame:
        """
        Run the complete perturbation matrix across all random seeds.
        """
        perturbations = {
            "Nominal Plant": {},
            "High Envelope Leakage (+35% k_env)": {"k_env": 0.18 * 1.35},
            "Tight Insulation (-35% k_env)": {"k_env": 0.18 * 0.65},
            "Degraded HVAC Efficiency (-25% k_ac)": {"k_ac": 0.035 * 0.75},
            "High Sensor Noise (0.5°C std)": {"noise_std_temp": 0.50},
            "Heatwave Bias (+2.5°C Ambient)": {"ambient_bias": 2.5},
            "Equipment Power Drift (+25% kW)": {"power_drift": 1.25}
        }

        results = []

        for p_name, p_kwargs in perturbations.items():
            seed_rewards = []
            seed_violations = []
            seed_energies = []
            seed_severes = []

            for s in seeds:
                # Build perturbed thermal model
                p_params = PhysicalParameters()
                if "k_env" in p_kwargs:
                    p_params.k_env = p_kwargs["k_env"]
                if "k_ac" in p_kwargs:
                    p_params.k_ac = p_kwargs["k_ac"]

                noise_t = p_kwargs.get("noise_std_temp", 0.10)
                t_model = GreyBoxThermalModel(params=p_params, noise_std_temp=noise_t, seed=s)

                # Perturbed data df (e.g. ambient bias)
                df_eval = self.data_df.copy()
                if "ambient_bias" in p_kwargs:
                    df_eval["ambient_temp_est"] += p_kwargs["ambient_bias"]

                # Perturbed calibration (equipment drift)
                cal_eval = deepcopy(self.cal)
                if "power_drift" in p_kwargs:
                    mult = p_kwargs["power_drift"]
                    cal_eval.low_power_kw *= mult
                    cal_eval.med_power_kw *= mult
                    cal_eval.high_power_kw *= mult

                env = TargetHVACEnv(
                    data_df=df_eval,
                    episode_length=24,
                    calibration=cal_eval,
                    thermal_model=t_model,
                    seed=s
                )

                m = evaluate_policy_on_env(self.agent, env, n_episodes=n_episodes_per_seed, seed=s)
                seed_rewards.append(m["mean_episode_reward"])
                seed_violations.append(m["comfort_violation_pct"])
                seed_energies.append(m["avg_energy_kwh_per_day"])
                seed_severes.append(m["severe_violation_pct"])

            results.append({
                "Perturbation Scenario": p_name,
                "Mean Reward": round(float(np.mean(seed_rewards)), 2),
                "Reward Std Across Seeds": round(float(np.std(seed_rewards)), 2),
                "Worst-Case Reward": round(float(np.min(seed_rewards)), 2),
                "Comfort Violation (%)": round(float(np.mean(seed_violations)), 2),
                "Severe Violations (%)": round(float(np.mean(seed_severes)), 2),
                "Daily Energy (kWh)": round(float(np.mean(seed_energies)), 2),
                "Robustness Status": "ROBUST" if np.mean(seed_severes) < 5.0 else "VULNERABLE"
            })

        df_res = pd.DataFrame(results)
        return df_res
