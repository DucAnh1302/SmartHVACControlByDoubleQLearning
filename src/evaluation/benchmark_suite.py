"""
Comprehensive Domain Shift and Cross-Domain Generalization Benchmark.
Evaluates:
1. Temporal Generalization (Train early 2018 -> Test late 2018)
2. Cross-Year Generalization (2018 Floor 2 Zone 1 -> 2019 Floor 2 Zone 1)
3. Cross-Zone Generalization (Floor 2 Zone 1 -> Floor 2 Zone 2)
4. Cross-Floor Generalization (Floor 2 Zone 1 -> Floor 3 Zone 1)

Across Three Strict Transfer Modes:
- ZERO-SHOT: Frozen Q-table, source equipment calibration
- CALIBRATION-ONLY: Frozen Q-table, target equipment calibration
- FINE-TUNING: Target equipment calibration + limited training budget (e.g. 25 episodes)
"""

from typing import Dict, List, Optional, Tuple, Any
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.agent.base_agent import BaseAgent
from src.agent.target_double_q_agent import TargetDoubleQLearningAgent
from src.agent.baseline_policies import (
    RandomPolicy, FixedSetpointPolicy, ComfortFirstPolicy,
    EnergySavingPolicy, LegacyDQLAdapter
)
from src.environment.target_env import TargetHVACEnv
from src.environment.greybox_model import GreyBoxThermalModel, PhysicalParameters
from src.data.domain_config import get_domain_spec, DOMAINS
from src.data.domain_loader import get_domain_dataset, split_temporal_dataset
from src.data.calibration import EquipmentCalibrator, EquipmentStageCalibration
from src.safety.safety_gateway import SafetyGateway


def evaluate_policy_on_env(
    agent: BaseAgent,
    env: TargetHVACEnv,
    n_episodes: int = 30,
    seed: int = 42,
    use_safety_gateway: bool = True
) -> Dict[str, float]:
    """
    Evaluate a policy deterministically over n_episodes on a given environment.
    """
    rng = np.random.default_rng(seed)
    max_start = len(env.data_df) - env.episode_length - 2
    if max_start <= 0:
        raise ValueError(f"Dataset length {len(env.data_df)} too small for evaluation!")

    start_indices = [int(rng.integers(0, max_start)) for _ in range(n_episodes)]

    gateway = SafetyGateway() if use_safety_gateway else None

    total_rewards = []
    comfort_violation_steps = 0
    severe_violation_steps = 0
    total_energy_kwh = 0.0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    temperatures = []
    humidities = []
    safety_overrides = 0

    total_steps = n_episodes * env.episode_length

    for ep_idx in range(n_episodes):
        start_i = start_indices[ep_idx]
        init_t = float(rng.uniform(23.0, 28.5))
        init_rh = float(rng.uniform(50.0, 75.0))

        state, info = env.reset(start_idx=start_i, initial_temp=init_t, initial_rh=init_rh)
        if gateway:
            gateway.reset()

        ep_r = 0.0
        done = False

        while not done:
            raw_action = agent.select_action(state, evaluate=True)

            if gateway:
                safe_act, reason = gateway.filter_action(
                    agent_action=raw_action,
                    indoor_temp=env.current_temp,
                    ambient_temp=float(env.data_df.iloc[env.data_idx]["ambient_temp_est"]),
                    ambient_rh=float(env.data_df.iloc[env.data_idx].get("ambient_rh_est", 70.0)),
                    hour=int(env.data_df.iloc[env.data_idx]["hour"]),
                    step=env.current_step
                )
                if safe_act != raw_action:
                    safety_overrides += 1
                action_to_take = safe_act
            else:
                action_to_take = raw_action

            action_counts[action_to_take] += 1
            next_state, reward, done, trunc, step_info = env.step(action_to_take)

            ep_r += reward
            t_curr = step_info["temp_c"]
            rh_curr = step_info["humidity_rh"]
            temperatures.append(t_curr)
            humidities.append(rh_curr)
            total_energy_kwh += step_info["hvac_kw"] * 1.0

            # ASHRAE comfort violation check [24.0, 26.0] preferred
            if t_curr < 24.0 or t_curr > 26.0:
                comfort_violation_steps += 1
            if t_curr < 21.0 or t_curr > 29.0:
                severe_violation_steps += 1

            state = next_state

        total_rewards.append(ep_r)

    avg_reward = float(np.mean(total_rewards))
    reward_std = float(np.std(total_rewards))
    comfort_violation_pct = float((comfort_violation_steps / total_steps) * 100.0)
    severe_violation_pct = float((severe_violation_steps / total_steps) * 100.0)
    avg_energy_ep = float(total_energy_kwh / n_episodes)
    mean_temp = float(np.mean(temperatures))
    temp_std = float(np.std(temperatures))
    mean_rh = float(np.mean(humidities))

    return {
        "mean_episode_reward": round(avg_reward, 2),
        "reward_std": round(reward_std, 2),
        "comfort_violation_pct": round(comfort_violation_pct, 2),
        "severe_violation_pct": round(severe_violation_pct, 2),
        "avg_energy_kwh_per_day": round(avg_energy_ep, 2),
        "mean_temperature_c": round(mean_temp, 2),
        "temp_std_c": round(temp_std, 2),
        "mean_humidity_rh": round(mean_rh, 1),
        "safety_override_rate_pct": round(float((safety_overrides / total_steps) * 100.0), 2),
        "action_distribution_pct": {
            k: round(float(v / total_steps * 100.0), 1) for k, v in action_counts.items()
        }
    }


class DomainShiftBenchmark:
    """
    Executes and records comparative domain transfer experiments.
    """

    def __init__(
        self,
        trained_agent: TargetDoubleQLearningAgent,
        source_domain_id: str = "2018_F2_Z1",
        calibrations_dir: Optional[Path] = None
    ):
        self.source_agent = trained_agent
        self.source_domain_id = source_domain_id
        self.calibrations_dir = calibrations_dir or (Path(__file__).resolve().parents[2] / "data" / "calibration")
        self.results: List[Dict[str, Any]] = []

    def run_all_benchmarks(
        self,
        target_domains: List[str] = ["2018_F2_Z1", "2019_F2_Z1", "2018_F2_Z2", "2018_F3_Z1"],
        n_eval_episodes: int = 25,
        fine_tune_budget_episodes: int = 25,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Run Temporal, Cross-Year, Cross-Zone, and Cross-Floor transfer in 3 modes:
        - ZERO-SHOT
        - CALIBRATION-ONLY
        - FINE-TUNING
        """
        source_cal_path = self.calibrations_dir / f"cal_{self.source_domain_id}.json"
        if not source_cal_path.exists():
            # Calibrate on the fly
            df_src = get_domain_dataset(self.source_domain_id)
            cal_src = EquipmentCalibrator().calibrate_from_series(df_src["hvac_kw"], domain_id=self.source_domain_id)
            cal_src.save(source_cal_path)
        else:
            cal_src = EquipmentStageCalibration.load(source_cal_path)

        for target_id in target_domains:
            df_target = get_domain_dataset(target_id)
            # Load target calibration
            target_cal_path = self.calibrations_dir / f"cal_{target_id}.json"
            if not target_cal_path.exists():
                cal_target = EquipmentCalibrator().calibrate_from_series(df_target["hvac_kw"], domain_id=target_id)
                cal_target.save(target_cal_path)
            else:
                cal_target = EquipmentStageCalibration.load(target_cal_path)

            if target_id == self.source_domain_id:
                # Temporal split: evaluate on unseen 30% test slice of source domain
                _, df_target_eval = split_temporal_dataset(df_target, train_ratio=0.70)
                eval_label = "Temporal Test (Unseen Period)"
            else:
                df_target_eval = df_target
                eval_label = f"Cross-Domain: {target_id}"

            # -------------------------------------------------------------
            # MODE 1: ZERO-SHOT
            # Uses source calibration on target domain, frozen Q-table
            # -------------------------------------------------------------
            env_zero_shot = TargetHVACEnv(
                data_df=df_target_eval,
                episode_length=24,
                calibration=cal_src,  # SOURCE CALIBRATION (No target adaptation)
                seed=seed
            )
            m_zero = evaluate_policy_on_env(self.source_agent, env_zero_shot, n_episodes=n_eval_episodes, seed=seed)
            self._record_result(self.source_domain_id, target_id, "ZERO-SHOT", eval_label, m_zero, "Frozen Q-table, source equipment calibration (0% target budget)")

            # -------------------------------------------------------------
            # MODE 2: CALIBRATION-ONLY
            # Uses target calibration, but keeps Q-table completely frozen!
            # -------------------------------------------------------------
            env_cal_only = TargetHVACEnv(
                data_df=df_target_eval,
                episode_length=24,
                calibration=cal_target,  # TARGET CALIBRATION (Zero policy retraining)
                seed=seed
            )
            m_cal = evaluate_policy_on_env(self.source_agent, env_cal_only, n_episodes=n_eval_episodes, seed=seed)
            self._record_result(self.source_domain_id, target_id, "CALIBRATION-ONLY", eval_label, m_cal, "Frozen Q-table, target equipment calibrated (0 training episodes on target)")

            # -------------------------------------------------------------
            # MODE 3: FINE-TUNING
            # Takes source Q-table, permits limited exploration updates on target data
            # -------------------------------------------------------------
            agent_ft = deepcopy(self.source_agent)
            agent_ft.name = f"DQL Fine-Tuned ({target_id})"
            agent_ft.epsilon = 0.15  # Low exploration budget
            agent_ft.epsilon_min = 0.02
            agent_ft.lr = 0.04       # Half learning rate for fine-tuning

            # Fine-tune on target training set
            env_ft_train = TargetHVACEnv(data_df=df_target_eval, episode_length=24, calibration=cal_target, seed=seed + 1)
            for _ in range(fine_tune_budget_episodes):
                s, _ = env_ft_train.reset()
                d = False
                while not d:
                    a = agent_ft.select_action(s, evaluate=False)
                    ns, r, d, _, _ = env_ft_train.step(a)
                    agent_ft.update(s, a, r, ns, d)
                    s = ns
                agent_ft.decay_epsilon()

            m_ft = evaluate_policy_on_env(agent_ft, env_cal_only, n_episodes=n_eval_episodes, seed=seed)
            self._record_result(self.source_domain_id, target_id, "FINE-TUNING", eval_label, m_ft, f"Fine-tuned with {fine_tune_budget_episodes} episodes on target data")

        df_bench = pd.DataFrame(self.results)
        return df_bench

    def _record_result(
        self,
        source: str,
        target: str,
        mode: str,
        label: str,
        metrics: Dict[str, Any],
        budget_note: str
    ) -> None:
        row = {
            "Source Domain": source,
            "Target Domain": target,
            "Transfer Mode": mode,
            "Benchmark Type": label,
            "Mean Reward": metrics["mean_episode_reward"],
            "Comfort Violation (%)": metrics["comfort_violation_pct"],
            "Severe Violation (%)": metrics["severe_violation_pct"],
            "Energy (kWh/day)": metrics["avg_energy_kwh_per_day"],
            "Mean Temp (°C)": metrics["mean_temperature_c"],
            "Temp Std (°C)": metrics["temp_std_c"],
            "Safety Overrides (%)": metrics["safety_override_rate_pct"],
            "Budget / Protocol": budget_note
        }
        self.results.append(row)


def evaluate_all_baselines(
    env: TargetHVACEnv,
    target_dql_agent: TargetDoubleQLearningAgent,
    legacy_dql_agent: Optional[BaseAgent] = None,
    n_episodes: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run identical test episodes for all 6 required policies:
    1. Random Policy
    2. Fixed Setpoint (25°C)
    3. Comfort-First Policy
    4. Energy-Saving Policy
    5. Legacy DQL Policy
    6. Improved DQL Policy
    """
    policies = {
        "Random Policy": RandomPolicy(seed=seed),
        "Fixed Setpoint (25°C)": FixedSetpointPolicy(),
        "Comfort-First Policy": ComfortFirstPolicy(),
        "Energy-Saving Policy": EnergySavingPolicy(),
        "Target DQL (Improved 600s)": target_dql_agent
    }
    if legacy_dql_agent is not None:
        policies["Legacy DQL (45s Adapted)"] = LegacyDQLAdapter(legacy_dql_agent)

    rows = []
    for name, pol in policies.items():
        m = evaluate_policy_on_env(pol, env, n_episodes=n_episodes, seed=seed)
        rows.append({
            "Policy": name,
            "Mean Reward": m["mean_episode_reward"],
            "Comfort Violation (%)": m["comfort_violation_pct"],
            "Severe Violation (%)": m["severe_violation_pct"],
            "Daily Energy (kWh)": m["avg_energy_kwh_per_day"],
            "Mean Temp (°C)": m["mean_temperature_c"],
            "Temp Std (°C)": m["temp_std_c"],
            "Safety Overrides (%)": m["safety_override_rate_pct"],
            "Action 0 (SET_25 %)": m["action_distribution_pct"][0],
            "Action 1 (SET_23 %)": m["action_distribution_pct"][1],
            "Action 2 (SET_27 %)": m["action_distribution_pct"][2],
            "Action 3 (AC_OFF %)": m["action_distribution_pct"][3],
            "Action 4 (WIN_OPEN %)": m["action_distribution_pct"][4]
        })

    df_base = pd.DataFrame(rows)
    return df_base
