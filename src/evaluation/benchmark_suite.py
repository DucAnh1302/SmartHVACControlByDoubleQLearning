"""
Comprehensive Domain Shift and Cross-Domain Generalization Benchmark.
Evaluates:
1. Temporal Generalization (Train early 2018 -> Test late 2018)
2. Cross-Year Generalization (2018 Floor 2 Zone 1 -> 2019 Floor 2 Zone 1)
3. Cross-Zone Generalization (Floor 2 Zone 1 -> Floor 2 Zone 2)
4. Cross-Floor Generalization (Floor 2 Zone 1 -> Floor 3 Zone 1)

Guarantees Strict Leakage-Free Partitions:
- Calibration Set (20%): Used ONLY for equipment stage clustering & Grey-Box parameter fitting.
- Adaptation Set (40%): Used ONLY for agent fine-tuning budget.
- Evaluation Set (40%): Held-out test set NEVER accessed during training, calibration, or adaptation.

Transfer Modes:
- ZERO-SHOT: Frozen Q-table, source equipment & physics calibration (0% target data budget)
- CALIBRATION-ONLY: Frozen Q-table, target equipment & physics calibrated on calibration slice (0 retraining episodes)
- FINE-TUNING: Target calibrated + policy fine-tuned on adaptation slice across budgets (e.g. 10, 25 episodes)
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
    EnergySavingPolicy, RuleBasedThermostatPolicy, ACOffPolicy, LegacyDQLAdapter
)
from src.environment.target_env import TargetHVACEnv
from src.environment.greybox_model import GreyBoxThermalModel, PhysicalParameters
from src.data.domain_config import get_domain_spec, DOMAINS
from src.data.domain_loader import (
    get_domain_dataset, split_temporal_dataset, split_target_domain_dataset
)
from src.data.calibration import EquipmentCalibrator, EquipmentStageCalibration
from src.safety.safety_gateway import SafetyGateway


def evaluate_policy_on_env(
    agent: BaseAgent,
    env: TargetHVACEnv,
    n_episodes: int = 30,
    seed: int = 42,
    use_safety_gateway: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a policy deterministically over n_episodes on a given environment.
    Computes comprehensive research metrics across Comfort, Energy, Control, and Safety.
    """
    rng = np.random.default_rng(seed)
    max_start = len(env.data_df) - env.episode_length - 2
    if max_start <= 0:
        raise ValueError(f"Dataset length {len(env.data_df)} too small for {env.episode_length}-hour evaluation!")

    start_indices = [int(rng.integers(0, max_start)) for _ in range(n_episodes)]
    gateway = SafetyGateway() if use_safety_gateway else None

    total_rewards = []
    episode_energies = []
    comfort_violation_steps = 0
    mild_violation_steps = 0
    severe_violation_steps = 0
    degree_hours_comfort = 0.0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    temperatures = []
    humidities = []
    safety_overrides = 0
    switching_events = 0
    ac_toggle_events = 0
    window_toggle_events = 0

    total_steps = n_episodes * env.episode_length

    for ep_idx in range(n_episodes):
        start_i = start_indices[ep_idx]
        init_t = float(rng.uniform(23.0, 28.5))
        init_rh = float(rng.uniform(50.0, 75.0))

        state, info = env.reset(start_idx=start_i, initial_temp=init_t, initial_rh=init_rh)
        if gateway:
            gateway.reset()

        ep_r = 0.0
        ep_energy = 0.0
        prev_act = 0
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

            # Control dynamics metrics
            if action_to_take != prev_act:
                switching_events += 1
                if (action_to_take in [3, 4]) != (prev_act in [3, 4]):
                    ac_toggle_events += 1
                if (action_to_take == 4) != (prev_act == 4):
                    window_toggle_events += 1
            prev_act = action_to_take

            action_counts[action_to_take] += 1
            next_state, reward, done, trunc, step_info = env.step(action_to_take)

            ep_r += reward
            step_energy = step_info["hvac_kw"] * 1.0
            ep_energy += step_energy

            t_curr = step_info["temp_c"]
            rh_curr = step_info["humidity_rh"]
            temperatures.append(t_curr)
            humidities.append(rh_curr)

            # Degree hours & violations
            if t_curr < 24.0:
                degree_hours_comfort += (24.0 - t_curr)
            elif t_curr > 26.0:
                degree_hours_comfort += (t_curr - 26.0)

            if t_curr < 24.0 or t_curr > 26.0:
                comfort_violation_steps += 1

            if (21.0 <= t_curr < 24.0) or (26.0 < t_curr <= 29.0):
                mild_violation_steps += 1
            elif t_curr < 21.0 or t_curr > 29.0:
                severe_violation_steps += 1

            state = next_state

        total_rewards.append(ep_r)
        episode_energies.append(ep_energy)

    avg_reward = float(np.mean(total_rewards))
    reward_std = float(np.std(total_rewards))
    avg_energy_ep = float(np.mean(episode_energies))
    energy_std = float(np.std(episode_energies))
    mean_temp = float(np.mean(temperatures))
    temp_std = float(np.std(temperatures))
    mean_rh = float(np.mean(humidities))

    comfort_pct = float(100.0 - (comfort_violation_steps / total_steps) * 100.0)
    mild_pct = float((mild_violation_steps / total_steps) * 100.0)
    severe_pct = float((severe_violation_steps / total_steps) * 100.0)
    mad_comfort = float(np.mean([abs(t - 25.0) for t in temperatures]))

    return {
        "mean_episode_reward": round(avg_reward, 2),
        "reward_std": round(reward_std, 2),
        "median_reward": round(float(np.median(total_rewards)), 2),
        "min_reward": round(float(np.min(total_rewards)), 2),
        "max_reward": round(float(np.max(total_rewards)), 2),
        "time_in_comfort_pct": round(comfort_pct, 2),
        "comfort_violation_pct": round(float((comfort_violation_steps / total_steps) * 100.0), 2),
        "mild_violation_pct": round(mild_pct, 2),
        "severe_violation_pct": round(severe_pct, 2),
        "mad_comfort_c": round(mad_comfort, 3),
        "degree_hours_outside_comfort": round(degree_hours_comfort / n_episodes, 2),
        "avg_energy_kwh_per_day": round(avg_energy_ep, 2),
        "energy_std_kwh": round(energy_std, 2),
        "energy_kwh_per_hour": round(avg_energy_ep / env.episode_length, 2),
        "mean_temperature_c": round(mean_temp, 2),
        "temp_std_c": round(temp_std, 2),
        "mean_humidity_rh": round(mean_rh, 1),
        "switching_events_per_day": round(switching_events / n_episodes, 1),
        "ac_toggles_per_day": round(ac_toggle_events / n_episodes, 1),
        "window_toggles_per_day": round(window_toggle_events / n_episodes, 1),
        "safety_override_rate_pct": round(float((safety_overrides / total_steps) * 100.0), 2),
        "action_distribution_pct": {
            k: round(float(v / total_steps * 100.0), 1) for k, v in action_counts.items()
        }
    }


class DomainShiftBenchmark:
    """
    Executes and records leakage-free cross-domain transfer experiments.
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
        fine_tune_budgets: List[int] = [10, 25],
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Run Temporal, Cross-Year, Cross-Zone, and Cross-Floor transfer across 3 strict modes:
        - ZERO-SHOT (0% target budget)
        - CALIBRATION-ONLY (0 retraining episodes)
        - FINE-TUNING (tested across multiple adaptation budgets)
        """
        # 1. Prepare Source Domain & Calibrations
        df_src = get_domain_dataset(self.source_domain_id)
        df_src_train, df_src_test = split_temporal_dataset(df_src, train_ratio=0.70)

        source_cal_path = self.calibrations_dir / f"cal_{self.source_domain_id}.json"
        if not source_cal_path.exists():
            cal_src = EquipmentCalibrator().calibrate_from_series(df_src_train["hvac_kw"], domain_id=self.source_domain_id)
            cal_src.save(source_cal_path)
        else:
            cal_src = EquipmentStageCalibration.load(source_cal_path)

        # Fit Source Grey-Box Physics Model strictly on source training set
        model_src = GreyBoxThermalModel().fit(df_src_train)

        for target_id in target_domains:
            df_target = get_domain_dataset(target_id)

            if target_id == self.source_domain_id:
                # Same-domain temporal split:
                df_target_calib = df_src_train.iloc[:int(len(df_src_train) * 0.3)].copy()
                df_target_adapt = df_src_train.iloc[int(len(df_src_train) * 0.3):].copy()
                df_target_test = df_src_test
                cal_target = cal_src
                model_target = model_src
                eval_label = "Temporal Test (Unseen Period)"
            else:
                # 3-Way Leakage-Free Target Domain Partition:
                df_target_calib, df_target_adapt, df_target_test = split_target_domain_dataset(
                    df_target, calib_ratio=0.20, adapt_ratio=0.40, test_ratio=0.40
                )
                eval_label = f"Cross-Domain: {target_id}"

                # Target equipment calibration fitted ONLY on calibration slice
                cal_target = EquipmentCalibrator().calibrate_from_series(df_target_calib["hvac_kw"], domain_id=target_id)
                # Target physics model fitted ONLY on calibration slice
                model_target = GreyBoxThermalModel().fit(df_target_calib)

            # Sanity assertion against data leakage
            assert df_target_calib["datetime"].max() < df_target_test["datetime"].min(), "Data leakage between calib and test!"
            assert df_target_adapt["datetime"].max() < df_target_test["datetime"].min(), "Data leakage between adapt and test!"

            # -------------------------------------------------------------
            # MODE 1: ZERO-SHOT
            # Source calibration & source physics applied to target test slice
            # -------------------------------------------------------------
            env_zero_shot = TargetHVACEnv(
                data_df=df_target_test,
                episode_length=24,
                calibration=cal_src,
                thermal_model=model_src,
                seed=seed
            )
            m_zero = evaluate_policy_on_env(self.source_agent, env_zero_shot, n_episodes=n_eval_episodes, seed=seed)
            self._record_result(self.source_domain_id, target_id, "ZERO-SHOT", eval_label, m_zero, "Frozen Q-table, source equipment calibration (0% target budget)")

            # -------------------------------------------------------------
            # MODE 2: CALIBRATION-ONLY
            # Target equipment & physics calibrated on calibration slice, Q-table frozen!
            # -------------------------------------------------------------
            env_cal_only = TargetHVACEnv(
                data_df=df_target_test,
                episode_length=24,
                calibration=cal_target,
                thermal_model=model_target,
                seed=seed
            )
            m_cal = evaluate_policy_on_env(self.source_agent, env_cal_only, n_episodes=n_eval_episodes, seed=seed)
            self._record_result(self.source_domain_id, target_id, "CALIBRATION-ONLY", eval_label, m_cal, "Frozen Q-table, target equipment calibrated on calib slice (0 retraining episodes)")

            # -------------------------------------------------------------
            # MODE 3: FINE-TUNING ACROSS BUDGETS
            # Policy updated ONLY on adaptation slice, evaluated ONLY on held-out test slice
            # -------------------------------------------------------------
            for budget in fine_tune_budgets:
                agent_ft = deepcopy(self.source_agent)
                agent_ft.name = f"DQL Fine-Tuned ({target_id}, {budget}ep)"
                agent_ft.epsilon = 0.15
                agent_ft.epsilon_min = 0.02
                agent_ft.lr = 0.04

                # Train on adaptation slice ONLY
                env_ft_train = TargetHVACEnv(
                    data_df=df_target_adapt, episode_length=24,
                    calibration=cal_target, thermal_model=model_target, seed=seed + 1
                )
                for _ in range(budget):
                    s, _ = env_ft_train.reset()
                    d = False
                    while not d:
                        a = agent_ft.select_action(s, evaluate=False)
                        ns, r, d, _, _ = env_ft_train.step(a)
                        agent_ft.update(s, a, r, ns, d)
                        s = ns
                    agent_ft.decay_epsilon()

                # Evaluate on strictly held-out test slice
                m_ft = evaluate_policy_on_env(agent_ft, env_cal_only, n_episodes=n_eval_episodes, seed=seed)
                self._record_result(
                    self.source_domain_id, target_id, f"FINE-TUNING-{budget}EP", eval_label,
                    m_ft, f"Fine-tuned with {budget} episodes on adaptation slice ONLY; evaluated on test slice"
                )

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
            "Reward Std": metrics["reward_std"],
            "Comfort Violation (%)": metrics["comfort_violation_pct"],
            "Severe Violation (%)": metrics["severe_violation_pct"],
            "Degree Hours": metrics["degree_hours_outside_comfort"],
            "Energy (kWh/day)": metrics["avg_energy_kwh_per_day"],
            "Mean Temp (°C)": metrics["mean_temperature_c"],
            "Temp Std (°C)": metrics["temp_std_c"],
            "Switching/Day": metrics["switching_events_per_day"],
            "Safety Overrides (%)": metrics["safety_override_rate_pct"],
            "Budget / Protocol": budget_note
        }
        self.results.append(row)


def evaluate_all_baselines_multiseed(
    env: TargetHVACEnv,
    target_dql_agent: TargetDoubleQLearningAgent,
    single_q_agent: Optional[BaseAgent] = None,
    legacy_dql_agent: Optional[BaseAgent] = None,
    seeds: List[int] = [42, 101, 202, 303, 404],
    n_episodes_per_seed: int = 15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run identical test episodes for all candidate policies across multiple random seeds.
    Computes rigorous statistical summaries (Mean, Std, Median, Min, Max, 95% CI).
    """
    base_policies_factories = {
        "Target DQL (Improved 600s)": lambda s: target_dql_agent,
        "Fixed Setpoint (25°C)": lambda s: FixedSetpointPolicy(),
        "Rule-Based Thermostat": lambda s: RuleBasedThermostatPolicy(),
        "Comfort-First Policy": lambda s: ComfortFirstPolicy(),
        "Energy-Saving Policy": lambda s: EnergySavingPolicy(),
        "AC-Always-Off Baseline": lambda s: ACOffPolicy(),
        "Random Policy": lambda s: RandomPolicy(seed=s)
    }
    if single_q_agent is not None:
        base_policies_factories["Single Q-Learning (600s)"] = lambda s: single_q_agent
    if legacy_dql_agent is not None:
        base_policies_factories["Legacy DQL (45s Adapted)"] = lambda s: LegacyDQLAdapter(legacy_dql_agent)

    seed_records = []

    for seed in seeds:
        for name, factory in base_policies_factories.items():
            policy = factory(seed)
            m = evaluate_policy_on_env(policy, env, n_episodes=n_episodes_per_seed, seed=seed)
            seed_records.append({
                "Seed": seed,
                "Policy": name,
                "Mean Reward": m["mean_episode_reward"],
                "Comfort Violation (%)": m["comfort_violation_pct"],
                "Severe Violation (%)": m["severe_violation_pct"],
                "Degree Hours": m["degree_hours_outside_comfort"],
                "Daily Energy (kWh)": m["avg_energy_kwh_per_day"],
                "Mean Temp (°C)": m["mean_temperature_c"],
                "Temp Std (°C)": m["temp_std_c"],
                "Switching/Day": m["switching_events_per_day"],
                "Safety Overrides (%)": m["safety_override_rate_pct"]
            })

    df_seeds = pd.DataFrame(seed_records)

    # Calculate statistics across seeds
    summary_rows = []
    fixed_energy_mean = df_seeds[df_seeds["Policy"] == "Fixed Setpoint (25°C)"]["Daily Energy (kWh)"].mean()

    for name in base_policies_factories.keys():
        sub = df_seeds[df_seeds["Policy"] == name]
        r_mean = float(sub["Mean Reward"].mean())
        r_std = float(sub["Mean Reward"].std())
        r_med = float(sub["Mean Reward"].median())
        r_min = float(sub["Mean Reward"].min())
        r_max = float(sub["Mean Reward"].max())

        e_mean = float(sub["Daily Energy (kWh)"].mean())
        e_std = float(sub["Daily Energy (kWh)"].std())
        saving_pct = float(((fixed_energy_mean - e_mean) / max(1e-3, fixed_energy_mean)) * 100.0)

        comf_mean = float(sub["Comfort Violation (%)"].mean())
        sev_mean = float(sub["Severe Violation (%)"].mean())
        deg_mean = float(sub["Degree Hours"].mean())
        temp_mean = float(sub["Mean Temp (°C)"].mean())
        so_mean = float(sub["Safety Overrides (%)"].mean())
        sw_mean = float(sub["Switching/Day"].mean())

        ci_95 = 1.96 * (r_std / np.sqrt(len(seeds)))

        summary_rows.append({
            "Policy": name,
            "Seeds Evaluated": len(seeds),
            "Mean Reward": round(r_mean, 2),
            "Reward Std": round(r_std, 2),
            "Reward 95% CI": f"[{round(r_mean - ci_95, 2)}, {round(r_mean + ci_95, 2)}]",
            "Median Reward": round(r_med, 2),
            "Min Reward": round(r_min, 2),
            "Max Reward": round(r_max, 2),
            "Comfort Violation (%)": round(comf_mean, 2),
            "Severe Violation (%)": round(sev_mean, 2),
            "Degree Hours Outside": round(deg_mean, 2),
            "Daily Energy (kWh)": round(e_mean, 2),
            "Energy Std": round(e_std, 2),
            "Simulated Saving vs Fixed-25 (%)": round(saving_pct, 2),
            "Mean Temp (°C)": round(temp_mean, 2),
            "Switching/Day": round(sw_mean, 1),
            "Safety Overrides (%)": round(so_mean, 2)
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_summary, df_seeds


def evaluate_all_baselines(
    env: TargetHVACEnv,
    target_dql_agent: TargetDoubleQLearningAgent,
    single_q_agent: Optional[BaseAgent] = None,
    legacy_dql_agent: Optional[BaseAgent] = None,
    n_episodes: int = 30,
    seed: int = 42
) -> pd.DataFrame:
    """Convenience wrapper for backward compatibility."""
    df_sum, _ = evaluate_all_baselines_multiseed(
        env=env,
        target_dql_agent=target_dql_agent,
        single_q_agent=single_q_agent,
        legacy_dql_agent=legacy_dql_agent,
        seeds=[seed],
        n_episodes_per_seed=n_episodes
    )
    return df_sum
