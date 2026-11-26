"""
Comprehensive Evaluation Suite for Smart HVAC Controllers.
Measures Reward, Thermal Comfort (ASHRAE 55), Energy Consumption, and Savings.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.environment.hvac_env import SmartHVACEnv, ACTION_POWER_MAP
from src.agent.base_agent import BaseAgent


def evaluate_agent(
    agent: BaseAgent,
    env: SmartHVACEnv,
    n_episodes: int = 50,
    start_indices: Optional[List[int]] = None
) -> Tuple[Dict[str, float], List[pd.DataFrame]]:
    """
    Evaluate an agent over multiple test episodes.
    
    Returns:
    --------
    (summary_metrics, list_of_trajectory_dfs)
    """
    episode_rewards = []
    comfort_violations = []
    total_energies = []
    temps = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    trajectories = []

    for ep in range(n_episodes):
        start_idx = start_indices[ep] if start_indices and ep < len(start_indices) else None
        state, info = env.reset(start_idx=start_idx, initial_temp=26.5)

        ep_reward = 0.0
        ep_energy = 0.0
        ep_violations = 0
        ep_history = []

        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            action_counts[action] += 1

            next_state, reward, done, trunc, step_info = env.step(action)

            ep_reward += reward
            ep_energy += step_info["hvac_kw"] * 1.0  # 1 hour = 1 kWh per kW

            temp = step_info["temp_c"]
            temps.append(temp)
            if temp < 22.0 or temp > 24.5:
                ep_violations += 1

            ep_history.append({
                "step": step_info["step"],
                "temp_c": temp,
                "action": action,
                "hvac_kw": step_info["hvac_kw"],
                "ambient_temp": step_info["ambient_temp"],
                "reward": reward,
                "is_workhour": step_info["is_workhour"]
            })

            state = next_state

        episode_rewards.append(ep_reward)
        total_energies.append(ep_energy)
        comfort_violations.append((ep_violations / env.episode_length) * 100.0)
        trajectories.append(pd.DataFrame(ep_history))

    total_steps = n_episodes * env.episode_length
    summary = {
        "Total Reward": float(np.mean(episode_rewards)),
        "Reward Std": float(np.std(episode_rewards)),
        "Comfort Violation (%)": float(np.mean(comfort_violations)),
        "Average Temp (°C)": float(np.mean(temps)),
        "Temp Std (°C)": float(np.std(temps)),
        "Total Energy (kWh)": float(np.mean(total_energies)),
        "Action 0 (%)": float((action_counts[0] / total_steps) * 100.0),
        "Action 1 (%)": float((action_counts[1] / total_steps) * 100.0),
        "Action 2 (%)": float((action_counts[2] / total_steps) * 100.0),
        "Action 3 (%)": float((action_counts[3] / total_steps) * 100.0)
    }

    return summary, trajectories


def compare_controllers(
    controllers: Dict[str, BaseAgent],
    env: SmartHVACEnv,
    n_episodes: int = 50,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Run identical test episodes across all controllers and calculate relative energy savings.
    """
    np.random.seed(seed)
    # Generate fixed random start indices for deterministic comparison
    max_start = len(env.data_df) - env.episode_length - 2
    fixed_starts = [int(np.random.randint(0, max_start)) for _ in range(n_episodes)]

    results = []
    trajectories_sample = {}

    for name, agent in controllers.items():
        summary, trajs = evaluate_agent(agent, env, n_episodes=n_episodes, start_indices=fixed_starts)
        summary["Model"] = name
        results.append(summary)
        # Store first episode trajectory for visual inspection
        trajectories_sample[name] = trajs[0]

    df_results = pd.DataFrame(results)

    # Compute Energy Savings (%) relative to Rule-Based Baseline if present
    if "Rule-Based" in df_results["Model"].values:
        baseline_energy = df_results.loc[df_results["Model"] == "Rule-Based", "Total Energy (kWh)"].values[0]
        df_results["Energy Savings (%)"] = (
            (baseline_energy - df_results["Total Energy (kWh)"]) / baseline_energy * 100.0
        ).round(2)
    else:
        df_results["Energy Savings (%)"] = 0.0

    # Reorder columns
    cols = ["Model", "Total Reward", "Comfort Violation (%)", "Total Energy (kWh)", "Energy Savings (%)", "Average Temp (°C)", "Temp Std (°C)"]
    df_results = df_results[cols + [c for c in df_results.columns if c not in cols]]

    return df_results, trajectories_sample
