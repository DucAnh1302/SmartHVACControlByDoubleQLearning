"""
Master Training and Benchmarking Pipeline for Target Double Q-Learning.
Trains the 600-state Target Agent on Primary Domain (2018 Floor 2 Zone 1),
evaluates Baselines with 5-Seed Statistics, runs DomainShiftBenchmark across 4 domains & 3 transfer modes,
and performs Robustness & Plant Uncertainty testing.
"""

from pathlib import Path
import sys
import json
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.domain_loader import get_domain_dataset, split_temporal_dataset
from src.data.calibration import EquipmentCalibrator, EquipmentStageCalibration
from src.environment.target_env import TargetHVACEnv
from src.environment.greybox_model import GreyBoxThermalModel
from src.agent.target_double_q_agent import TargetDoubleQLearningAgent
from src.agent.q_learning_agent import QLearningAgent
from src.agent.double_q_learning_agent import DoubleQLearningAgent
from src.evaluation.benchmark_suite import (
    DomainShiftBenchmark, evaluate_all_baselines_multiseed, evaluate_policy_on_env
)
from src.evaluation.robustness_suite import RobustnessExperimentSuite


def train_target_agent(
    agent: TargetDoubleQLearningAgent,
    env: TargetHVACEnv,
    n_episodes: int = 800
) -> Dict:
    """Train target agent and log detailed learning diagnostics."""
    print(f"\n--- Training {agent.name} for {n_episodes} episodes ---")
    history = {
        "episode_rewards": [],
        "epsilons": [],
        "avg_max_q": [],
        "coverage_pct": [],
        "mean_td_error": []
    }

    for ep in range(1, n_episodes + 1):
        state, info = env.reset()
        ep_r = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, trunc, step_info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_r += reward
            state = next_state

        agent.decay_epsilon()

        history["episode_rewards"].append(float(ep_r))
        history["epsilons"].append(float(agent.epsilon))
        history["avg_max_q"].append(float(agent.get_avg_max_q()))
        history["coverage_pct"].append(float(agent.get_coverage_percentage()))
        recent_td = float(np.mean(agent.td_errors[-24:])) if agent.td_errors else 0.0
        history["mean_td_error"].append(recent_td)

        if ep % 100 == 0 or ep == n_episodes:
            rec_r = np.mean(history["episode_rewards"][-30:])
            cov = history["coverage_pct"][-1]
            mq = history["avg_max_q"][-1]
            print(f"DQL Ep {ep:4d}/{n_episodes} | Avg R: {rec_r:6.2f} | Max Q: {mq:6.2f} | Coverage: {cov:5.1f}% | Eps: {agent.epsilon:.3f}")

    return history


def train_single_q_agent(
    agent: QLearningAgent,
    env: TargetHVACEnv,
    n_episodes: int = 800
) -> Dict:
    """Train single Q-learning baseline agent on the identical environment."""
    print(f"\n--- Training {agent.name} for {n_episodes} episodes ---")
    history = {
        "episode_rewards": [],
        "epsilons": [],
        "avg_max_q": [],
        "mean_td_error": []
    }

    for ep in range(1, n_episodes + 1):
        state, info = env.reset()
        ep_r = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, trunc, step_info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_r += reward
            state = next_state

        agent.decay_epsilon()

        history["episode_rewards"].append(float(ep_r))
        history["epsilons"].append(float(agent.epsilon))
        history["avg_max_q"].append(float(agent.get_avg_max_q()))
        recent_td = float(np.mean(agent.td_errors[-24:])) if agent.td_errors else 0.0
        history["mean_td_error"].append(recent_td)

        if ep % 100 == 0 or ep == n_episodes:
            rec_r = np.mean(history["episode_rewards"][-30:])
            mq = history["avg_max_q"][-1]
            print(f"Single-Q Ep {ep:4d}/{n_episodes} | Avg R: {rec_r:6.2f} | Max Q: {mq:6.2f} | Eps: {agent.epsilon:.3f}")

    return history


def plot_target_training(history: Dict, save_path: Path) -> None:
    """Plot target agent training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    episodes = range(1, len(history["episode_rewards"]) + 1)
    smooth_r = pd.Series(history["episode_rewards"]).rolling(25, min_periods=1).mean()

    # 1. Reward
    axes[0].plot(episodes, history["episode_rewards"], color="#0275d8", alpha=0.15)
    axes[0].plot(episodes, smooth_r, color="#0275d8", lw=2.2)
    axes[0].set_title("Episode Reward Convergence (600 States)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Daily Reward")
    axes[0].grid(True, alpha=0.3)

    # 2. Average Max Q
    axes[1].plot(episodes, history["avg_max_q"], color="#5cb85c", lw=2.2)
    axes[1].set_title("Average Max Action-Value Q(s, a)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Mean Q-value")
    axes[1].grid(True, alpha=0.3)

    # 3. State-Action Coverage
    axes[2].plot(episodes, history["coverage_pct"], color="#f0ad4e", lw=2.2)
    axes[2].set_title("State-Action Space Coverage (%)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Visited Pairs (%)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved training figure to: {save_path}")


def plot_dql_vs_ql(history_dql: Dict, history_sq: Dict, save_path: Path) -> None:
    """Plot DQL vs Single Q-Learning Q-value overestimation and reward comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    episodes = range(1, len(history_dql["episode_rewards"]) + 1)

    # 1. Action-Value Q(s, a) Overestimation
    axes[0].plot(episodes, history_sq["avg_max_q"], color="#ef4444", lw=2.0, label="Single Q-Learning (Overestimation Bias)")
    axes[0].plot(episodes, history_dql["avg_max_q"], color="#0284c7", lw=2.2, label="Double Q-Learning (Decoupled Estimators)")
    axes[0].set_title("Action-Value Function Q(s, a) Estimation", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Max Q-Value")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # 2. Reward Comparison (Rolling Mean)
    r_dql_smooth = pd.Series(history_dql["episode_rewards"]).rolling(25, min_periods=1).mean()
    r_sq_smooth = pd.Series(history_sq["episode_rewards"]).rolling(25, min_periods=1).mean()
    axes[1].plot(episodes, r_sq_smooth, color="#ef4444", lw=2.0, label="Single Q-Learning")
    axes[1].plot(episodes, r_dql_smooth, color="#0284c7", lw=2.2, label="Double Q-Learning")
    axes[1].set_title("Training Reward Progression (25-Ep Rolling Mean)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Total Daily Reward")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved DQL vs QL comparison to: {save_path}")


def main():
    print("==========================================================================")
    print("  SmartHVACControlByDoubleQLearning: Target Agent Training & Benchmarking  ")
    print("==========================================================================")

    # 1. Load Primary Domain (2018 Floor 2 Zone 1)
    df18 = get_domain_dataset("2018_F2_Z1")
    df_train, df_test = split_temporal_dataset(df18, train_ratio=0.70)
    print(f"Primary Domain: {len(df_train)} train hours, {len(df_test)} test hours.")

    # 2. Equipment Calibration on Training Set
    cal_dir = project_root / "data" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_path = cal_dir / "cal_2018_F2_Z1.json"
    if cal_path.exists():
        cal = EquipmentStageCalibration.load(cal_path)
    else:
        cal = EquipmentCalibrator().calibrate_from_series(df_train["hvac_kw"], domain_id="2018_F2_Z1")
        cal.save(cal_path)
    print(f"Equipment Calibration (2018): {cal.stage_power_map}")

    # 3. Explicitly Fit Grey-Box RC Thermal Model on Training Set
    print("\n--- Fitting Grey-Box RC Thermal Model on Training Set ---")
    thermal_model = GreyBoxThermalModel()
    fit_metrics = thermal_model.fit(df_train)
    greybox_path = cal_dir / "fitted_greybox_2018_F2_Z1.json"
    thermal_model.save(str(greybox_path))
    print(f"Fitted Thermal Model saved to: {greybox_path}")
    print(f"Identified Parameters: k_env={thermal_model.params.k_env}, k_int={thermal_model.params.k_int}, k_ac={thermal_model.params.k_ac}")

    # Sensitivity Analysis
    log_dir = project_root / "result" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    df_sensitivity = thermal_model.perform_sensitivity_analysis()
    df_sensitivity.to_csv(log_dir / "greybox_sensitivity_analysis.csv", index=False)
    with open(log_dir / "greybox_sensitivity_analysis.json", "w") as f:
        json.dump(df_sensitivity.to_dict(orient="records"), f, indent=2)
    print(f"Sensitivity Analysis saved to: {log_dir / 'greybox_sensitivity_analysis.csv'}")

    # 4. Initialize Environments with Fitted Physics Model
    env_train = TargetHVACEnv(data_df=df_train, episode_length=24, calibration=cal, thermal_model=thermal_model, seed=42)
    env_test = TargetHVACEnv(data_df=df_test, episode_length=24, calibration=cal, thermal_model=thermal_model, seed=123)

    # 5. Train Target Double Q-Learning Agent (600 states)
    agent = TargetDoubleQLearningAgent(n_states=600, n_actions=5, seed=42)
    history_dql = train_target_agent(agent, env_train, n_episodes=800)

    # Save Double Q-Learning Model
    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    target_model_path = model_dir / "target_double_q_model.pkl"
    agent.save(str(target_model_path))

    # 6. Train Single Q-Learning Agent as Direct Baseline (600 states)
    env_train_sq = TargetHVACEnv(data_df=df_train, episode_length=24, calibration=cal, thermal_model=thermal_model, seed=42)
    single_q_agent = QLearningAgent(
        n_states=600, n_actions=5,
        learning_rate=0.08, gamma=0.95,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.02,
        name="Single Q-Learning (600s)"
    )
    history_sq = train_single_q_agent(single_q_agent, env_train_sq, n_episodes=800)
    single_q_agent.save(str(model_dir / "single_q_model.pkl"))

    # Plot Convergence and Overestimation
    fig_dir = project_root / "result" / "figure"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_target_training(history_dql, fig_dir / "target_training_convergence.png")
    plot_dql_vs_ql(history_dql, history_sq, fig_dir / "dql_vs_ql_overestimation.png")

    # 7. Baseline Evaluation on Unseen 2018 Test Period (5 Seeds)
    print("\n================== BASELINE POLICIES BENCHMARK (5 SEEDS) ==================")
    legacy_agent_path = model_dir / "double_q_learning_model.pkl"
    legacy_agent = None
    if legacy_agent_path.exists():
        try:
            legacy_agent = DoubleQLearningAgent.load(str(legacy_agent_path))
        except Exception as e:
            print("Could not load legacy model:", e)

    df_baselines_summary, df_baselines_seeds = evaluate_all_baselines_multiseed(
        env=env_test,
        target_dql_agent=agent,
        single_q_agent=single_q_agent,
        legacy_dql_agent=legacy_agent,
        seeds=[42, 101, 202, 303, 404],
        n_episodes_per_seed=15
    )
    print("\n--- Multi-Seed Baseline Summary (5 Seeds) ---")
    print(df_baselines_summary.to_string(index=False))

    # 8. Domain Shift Benchmark (4 domains, 3 transfer modes, multi-budget fine-tuning)
    print("\n================== DOMAIN SHIFT BENCHMARK (3 TRANSFER MODES) ==================")
    benchmark = DomainShiftBenchmark(agent, source_domain_id="2018_F2_Z1", calibrations_dir=cal_dir)
    df_domain_shift = benchmark.run_all_benchmarks(
        target_domains=["2018_F2_Z1", "2019_F2_Z1", "2018_F2_Z2", "2018_F3_Z1"],
        n_eval_episodes=20,
        fine_tune_budgets=[10, 25],
        seed=123
    )
    print(df_domain_shift.to_string(index=False))

    # 9. Robustness & Plant Uncertainty Test (7 Plant Perturbations)
    print("\n================== ROBUSTNESS & PLANT UNCERTAINTY TEST ==================")
    robustness_suite = RobustnessExperimentSuite(agent, base_env_data=df_test, base_calibration=cal)
    df_robustness = robustness_suite.run_robustness_test(seeds=[42, 123, 999], n_episodes_per_seed=15)
    print(df_robustness.to_string(index=False))

    # 10. Save All Benchmark Outputs to result/log
    df_baselines_summary.to_csv(log_dir / "baselines_benchmark.csv", index=False)
    df_baselines_seeds.to_csv(log_dir / "baselines_multiseed_raw.csv", index=False)
    df_domain_shift.to_csv(log_dir / "domain_shift_benchmark.csv", index=False)
    df_robustness.to_csv(log_dir / "robustness_benchmark.csv", index=False)

    with open(log_dir / "target_training_history.json", "w") as f:
        json.dump(history_dql, f, indent=2)
    with open(log_dir / "single_q_training_history.json", "w") as f:
        json.dump(history_sq, f, indent=2)

    print("\n[Complete] All training, baseline comparisons, domain shift benchmarks, and robustness tests completed!")


if __name__ == "__main__":
    main()
