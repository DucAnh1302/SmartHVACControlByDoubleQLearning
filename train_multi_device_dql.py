"""
Training and Benchmark Pipeline for Multi-Device Double Q-Learning Agent.
Trains Double Q-Learning and Single Q-Learning agents on the MultiDeviceHVACEnv,
evaluates baselines across multiple random seeds, and records energy, comfort,
and indoor air quality metrics.
"""

from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.environment.multi_device_env import MultiDeviceHVACEnv, ROOM_PRESETS, MULTI_DEVICE_ACTION_TEMPLATES
from src.reward.multi_device_reward import MultiDeviceRewardFunction
from src.safety.multi_device_safety_gateway import MultiDeviceSafetyGateway
from src.agent.multi_device_double_q_agent import MultiDeviceDoubleQAgent, SingleQMultiDeviceAgent
from src.agent.multi_device_baselines import (
    FixedSetpointMultiDevice, ConventionalThermostatMultiDevice,
    RuleBasedSmartHVACMultiDevice, RandomMultiDevicePolicy
)
from src.data.domain_loader import get_domain_dataset


def train_multi_device_agent(
    agent: MultiDeviceDoubleQAgent,
    env: MultiDeviceHVACEnv,
    reward_fn: MultiDeviceRewardFunction,
    n_episodes: int = 500
) -> Dict[str, List[float]]:
    """Train Double Q-Learning agent on multi-device digital twin."""
    print(f"\n=======================================================")
    print(f"Training {agent.name} on Multi-Device HVAC Digital Twin")
    print(f"Episodes: {n_episodes} | States: {agent.n_states} | Actions: {agent.n_actions}")
    print(f"=======================================================")

    history = {
        "episode_rewards": [],
        "epsilons": [],
        "avg_max_q": [],
        "coverage_pct": [],
        "energy_kwh": []
    }

    for ep in range(1, n_episodes + 1):
        state, info = env.reset()
        ep_r = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, _, done, trunc, step_info = env.step(action)

            # Compute multi-objective reward
            reward, breakdown = reward_fn.calculate_reward(
                temp_c=step_info["temp_c"],
                humidity_rh=step_info["humidity_rh"],
                co2_ppm=step_info["co2_ppm"],
                pm25_ug_m3=step_info["pm25_ug_m3"],
                total_power_kw=step_info["powers_kw"]["total_kw"],
                hour=step_info["hour"],
                dayofweek=step_info["dayofweek"],
                is_workhour=step_info.get("is_workhour"),
                action_changed=step_info["action_changed"]
            )

            agent.update(state, action, reward, next_state, done)
            ep_r += reward
            state = next_state

        agent.decay_epsilon()

        history["episode_rewards"].append(float(ep_r))
        history["epsilons"].append(float(agent.epsilon))
        history["avg_max_q"].append(float(agent.get_avg_max_q()))
        history["coverage_pct"].append(float(agent.get_coverage_percentage()))
        history["energy_kwh"].append(float(env.cumul_energy["total_kwh"]))

        if ep % 50 == 0 or ep == n_episodes:
            rec_r = float(np.mean(history["episode_rewards"][-20:]))
            cov = history["coverage_pct"][-1]
            mq = history["avg_max_q"][-1]
            en = float(np.mean(history["energy_kwh"][-20:]))
            print(f"Ep {ep:4d}/{n_episodes} | Avg R: {rec_r:7.2f} | Max Q: {mq:6.2f} | Cov: {cov:5.1f}% | Energy: {en:5.1f} kWh | Eps: {agent.epsilon:.3f}")

    return history


def train_single_q_baseline(
    agent: SingleQMultiDeviceAgent,
    env: MultiDeviceHVACEnv,
    reward_fn: MultiDeviceRewardFunction,
    n_episodes: int = 500
) -> Dict[str, List[float]]:
    """Train Single Q-learning baseline on identical environment."""
    print(f"\n--- Training {agent.name} for {n_episodes} episodes ---")
    history = {"episode_rewards": [], "epsilons": []}

    for ep in range(1, n_episodes + 1):
        state, info = env.reset()
        ep_r = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, _, done, trunc, step_info = env.step(action)
            reward, _ = reward_fn.calculate_reward(
                temp_c=step_info["temp_c"],
                humidity_rh=step_info["humidity_rh"],
                co2_ppm=step_info["co2_ppm"],
                pm25_ug_m3=step_info["pm25_ug_m3"],
                total_power_kw=step_info["powers_kw"]["total_kw"],
                hour=step_info["hour"],
                dayofweek=step_info["dayofweek"],
                is_workhour=step_info.get("is_workhour"),
                action_changed=step_info["action_changed"]
            )
            agent.update(state, action, reward, next_state, done)
            ep_r += reward
            state = next_state

        agent.decay_epsilon()
        history["episode_rewards"].append(float(ep_r))

    return history


def evaluate_policy(
    policy,
    env: MultiDeviceHVACEnv,
    reward_fn: MultiDeviceRewardFunction,
    safety_gateway: Optional[MultiDeviceSafetyGateway] = None,
    n_episodes: int = 10,
    seed: int = 100
) -> Dict[str, float]:
    """Rigorous evaluation of a policy across multiple evaluation episodes."""
    tot_rewards = []
    tot_energies = []
    temp_violations = []
    rh_violations = []
    co2_violations = []
    pm25_violations = []
    overrides_count = 0
    total_steps = 0

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        start_idx = int(rng.integers(0, max(1, len(env.data_df) - env.episode_length - 2)))
        state, info = env.reset(start_idx=start_idx, seed=int(rng.integers(0, 100000)))
        if safety_gateway:
            safety_gateway.reset()

        ep_r = 0.0
        done = False

        while not done:
            raw_action = policy.select_action(state, evaluate=True)

            if safety_gateway:
                safe_act, reason = safety_gateway.filter_action(
                    agent_action=raw_action,
                    indoor_temp=env.current_temp,
                    indoor_rh=env.current_rh,
                    indoor_co2=env.current_co2,
                    indoor_pm25=env.current_pm25,
                    ambient_temp=float(env.data_df.iloc[env.data_idx].get("ambient_temp_est", 32.0)),
                    ambient_rh=float(env.data_df.iloc[env.data_idx].get("ambient_rh_est", 65.0)),
                    ambient_pm25=float(env.data_df.iloc[env.data_idx].get("ambient_pm25_est", 25.0)),
                    hour=int(env.data_df.iloc[env.data_idx].get("hour", 12)),
                    step=env.current_step
                )
                if safe_act != raw_action:
                    overrides_count += 1
                act_to_exec = safe_act
            else:
                act_to_exec = raw_action

            next_state, _, done, trunc, step_info = env.step(act_to_exec)
            r, _ = reward_fn.calculate_reward(
                temp_c=step_info["temp_c"],
                humidity_rh=step_info["humidity_rh"],
                co2_ppm=step_info["co2_ppm"],
                pm25_ug_m3=step_info["pm25_ug_m3"],
                total_power_kw=step_info["powers_kw"]["total_kw"],
                hour=step_info["hour"],
                dayofweek=step_info["dayofweek"],
                is_workhour=step_info.get("is_workhour"),
                action_changed=step_info["action_changed"]
            )
            ep_r += r

            # Check comfort & safety violations
            t = step_info["temp_c"]
            rh = step_info["humidity_rh"]
            co2 = step_info["co2_ppm"]
            pm = step_info["pm25_ug_m3"]

            temp_violations.append(1.0 if (t < 24.0 or t > 26.0) else 0.0)
            rh_violations.append(1.0 if (rh < 40.0 or rh > 65.0) else 0.0)
            co2_violations.append(1.0 if co2 > 1000.0 else 0.0)
            pm25_violations.append(1.0 if pm > 35.0 else 0.0)
            total_steps += 1
            state = next_state

        tot_rewards.append(ep_r)
        tot_energies.append(env.cumul_energy["total_kwh"])

    return {
        "mean_reward": float(np.mean(tot_rewards)),
        "std_reward": float(np.std(tot_rewards)),
        "mean_energy_kwh": float(np.mean(tot_energies)),
        "std_energy_kwh": float(np.std(tot_energies)),
        "temp_violation_pct": float(np.mean(temp_violations) * 100.0),
        "rh_violation_pct": float(np.mean(rh_violations) * 100.0),
        "co2_violation_pct": float(np.mean(co2_violations) * 100.0),
        "pm25_violation_pct": float(np.mean(pm25_violations) * 100.0),
        "safety_overrides": overrides_count,
        "total_eval_steps": total_steps
    }


def run_multi_device_training_and_benchmark():
    """Main pipeline execution."""
    data_df = get_domain_dataset("2018_F2_Z1")
    reward_fn = MultiDeviceRewardFunction()
    safety_gateway = MultiDeviceSafetyGateway()

    env = MultiDeviceHVACEnv(data_df=data_df, room=ROOM_PRESETS["medium"], episode_length=24, seed=42)

    # 1. Train Double Q-Learning Agent
    dql_agent = MultiDeviceDoubleQAgent(seed=42)
    dql_history = train_multi_device_agent(dql_agent, env, reward_fn, n_episodes=400)

    # 2. Train Single Q-Learning Agent Baseline
    sql_agent = SingleQMultiDeviceAgent(seed=42)
    sql_history = train_single_q_baseline(sql_agent, env, reward_fn, n_episodes=400)

    # Save models
    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dql_save_path = model_dir / "multi_device_double_q_model.pkl"
    dql_agent.save_checkpoint(dql_save_path)

    sql_save_path = model_dir / "multi_device_single_q_model.pkl"
    with open(sql_save_path, "wb") as f:
        import pickle
        pickle.dump(sql_agent, f)
    print(f"Single Q model saved to {sql_save_path}")

    # 3. Comparative Benchmark across Policies
    policies = {
        "Double Q-Learning (Multi-Device)": dql_agent,
        "Single Q-Learning (Multi-Device)": sql_agent,
        "Fixed Setpoint (25°C Cool)": FixedSetpointMultiDevice(),
        "Conventional Thermostat": ConventionalThermostatMultiDevice(),
        "Rule-Based Smart HVAC": RuleBasedSmartHVACMultiDevice(),
        "Random Policy": RandomMultiDevicePolicy(seed=42)
    }

    print("\n--- Running Multi-Policy Benchmark Evaluation (15 episodes each) ---")
    results = {}
    for name, pol in policies.items():
        res = evaluate_policy(pol, env, reward_fn, safety_gateway, n_episodes=15, seed=2026)
        results[name] = res
        print(f"[{name:32s}] R: {res['mean_reward']:7.1f} | Energy: {res['mean_energy_kwh']:5.1f} kWh | "
              f"T-Viol: {res['temp_violation_pct']:4.1f}% | CO2-Viol: {res['co2_violation_pct']:4.1f}% | Overrides: {res['safety_overrides']}")

    # Save benchmark metrics to JSON
    result_dir = project_root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    bench_json_path = result_dir / "multi_device_benchmark.json"
    with open(bench_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nBenchmark results saved to: {bench_json_path}")

    # Plot Training Convergence
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(dql_history["episode_rewards"], label="Double Q-Learning", color="#0284c7")
    axes[0].plot(sql_history["episode_rewards"], label="Single Q-Learning", color="#f97316", alpha=0.7)
    axes[0].set_title("Training Reward Curve")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(dql_history["avg_max_q"], label="Avg Max Q", color="#10b981")
    axes[1].set_title("Double Q Value Scale")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg Max Q")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dql_history["coverage_pct"], label="Coverage %", color="#8b5cf6")
    axes[2].set_title("State-Action Coverage (720x12)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Coverage %")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = result_dir / "figure" / "multi_device_training_curves.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Training figure saved to: {fig_path}")


if __name__ == "__main__":
    run_multi_device_training_and_benchmark()
