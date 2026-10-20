"""
Comprehensive Evaluation Script for Smart HVAC Control.
Evaluates Rule-Based, Q-Learning Baseline, and Double Q-Learning on unseen test data.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Setup project root in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.environment.hvac_env import SmartHVACEnv
from src.agent.q_learning_agent import QLearningAgent
from src.agent.double_q_learning_agent import DoubleQLearningAgent
from src.agent.rule_based_agent import RuleBasedAgent
from src.evaluation.evaluator import compare_controllers
from src.util.visualization import plot_evaluation_comparison, plot_control_trajectory


def main():
    print("==========================================================")
    print("  SmartHVACControlByDoubleQLearning: Model Evaluation     ")
    print("==========================================================")

    # 1. Load Processed Dataset and slice Test Split
    data_path = project_root / "data" / "processed" / "cubems_zone1_hourly.csv"
    df = pd.read_csv(data_path)
    train_size = int(len(df) * 0.8)
    df_test = df.iloc[train_size:].reset_index(drop=True)
    print(f"Loaded Test Dataset: {len(df_test)} hours ({df_test['datetime'].min()} to {df_test['datetime'].max()})")

    # 2. Setup Evaluation Environment
    env_test = SmartHVACEnv(df_test, episode_length=24, seed=123)

    # 3. Load Trained Agents
    model_dir = project_root / "model"
    ql_path = model_dir / "q_learning_model.pkl"
    dql_path = model_dir / "double_q_learning_model.pkl"

    if not ql_path.exists() or not dql_path.exists():
        raise FileNotFoundError("Trained models not found! Run train.py first.")

    agent_rule = RuleBasedAgent(env_test.n_states, env_test.action_space_size)
    agent_ql = QLearningAgent.load(str(ql_path))
    agent_dql = DoubleQLearningAgent.load(str(dql_path))

    controllers = {
        "Rule-Based": agent_rule,
        "Q-Learning": agent_ql,
        "Double Q-Learning": agent_dql
    }

    # 4. Run Comparative Evaluation (50 test episodes)
    n_test_episodes = 50
    print(f"\nRunning evaluation over {n_test_episodes} test episodes per controller...")
    df_results, trajectories = compare_controllers(controllers, env_test, n_episodes=n_test_episodes, seed=123)

    # 5. Print Results Table
    print("\n================== EVALUATION RESULTS SUMMARY ==================")
    print(df_results.to_string(index=False))

    # 6. Save Evaluation Metrics to result/log
    log_dir = project_root / "result" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(log_dir / "evaluation_summary.csv", index=False)
    with open(log_dir / "evaluation_summary.json", "w") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=2)
    print(f"\n[Logging] Saved evaluation summary to: {log_dir / 'evaluation_summary.csv'}")

    # 7. Generate and Save Comparison Visualizations
    plot_evaluation_comparison(df_results)
    plot_control_trajectory(trajectories)

    print("\n=== Evaluation Completed Successfully! ===")


if __name__ == "__main__":
    main()
