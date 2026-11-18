"""
Training Script for Smart HVAC Reinforcement Learning Controllers.
Trains Q-Learning (Baseline) and Double Q-Learning (Proposed) and saves models.
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
from src.util.visualization import plot_training_convergence


def train_agent(agent, env, n_episodes=600):
    """Train an RL agent on the environment and return performance history."""
    print(f"\n--- Training {agent.name} for {n_episodes} episodes ---")
    history = {
        "rewards": [],
        "epsilons": [],
        "avg_max_q": []
    }

    for ep in range(1, n_episodes + 1):
        state, info = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, trunc, step_info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state

        agent.decay_epsilon()

        # Track metrics
        history["rewards"].append(float(ep_reward))
        history["epsilons"].append(float(agent.epsilon))
        history["avg_max_q"].append(float(agent.get_avg_max_q()))

        if ep % 50 == 0 or ep == n_episodes:
            recent_r = np.mean(history["rewards"][-25:])
            print(f"Episode {ep:4d}/{n_episodes} | Avg Reward (last 25): {recent_r:6.2f} | Max Q: {history['avg_max_q'][-1]:6.2f} | Epsilon: {agent.epsilon:.3f}")

    return history


def main():
    print("==========================================================")
    print("  SmartHVACControlByDoubleQLearning: Model Training       ")
    print("==========================================================")

    # 1. Load Processed Dataset
    data_path = project_root / "data" / "processed" / "cubems_zone1_hourly.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {data_path}. Run run_preprocess.py first!")

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {len(df)} records ({df['datetime'].min()} to {df['datetime'].max()})")

    # Train / Test split: 80% train, 20% test
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    print(f"Train set: {len(df_train)} hours | Test set: {len(df_test)} hours")

    # 2. Setup Environment
    env_train = SmartHVACEnv(df_train, episode_length=24, seed=42)
    n_states = env_train.n_states
    n_actions = env_train.action_space_size
    print(f"Environment initialized: {n_states} discrete states, {n_actions} discrete actions.")

    # 3. Initialize Agents with identical hyperparameters
    n_episodes = 600
    lr = 0.1
    gamma = 0.95
    eps_decay = 0.995

    agent_ql = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=lr,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=eps_decay,
        name="Q-Learning"
    )

    agent_dql = DoubleQLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=lr,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=eps_decay,
        name="Double Q-Learning"
    )

    # 4. Train Both Agents
    np.random.seed(42)
    history_ql = train_agent(agent_ql, env_train, n_episodes=n_episodes)

    np.random.seed(42)
    history_dql = train_agent(agent_dql, env_train, n_episodes=n_episodes)

    # 5. Save Models
    model_dir = project_root / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    agent_ql.save(str(model_dir / "q_learning_model.pkl"))
    agent_dql.save(str(model_dir / "double_q_learning_model.pkl"))

    # 6. Save Convergence Plot
    plot_training_convergence(history_ql, history_dql, window=25)

    # 7. Save Training Logs
    log_dir = project_root / "result" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    training_log = {
        "n_episodes": n_episodes,
        "train_hours": len(df_train),
        "test_hours": len(df_test),
        "hyperparameters": {
            "learning_rate": lr,
            "gamma": gamma,
            "epsilon_decay": eps_decay,
            "n_states": n_states,
            "n_actions": n_actions
        },
        "q_learning": history_ql,
        "double_q_learning": history_dql
    }
    with open(log_dir / "training_history.json", "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n[Logging] Saved training history to: {log_dir / 'training_history.json'}")

    print("\n=== Training Completed Successfully! ===")


if __name__ == "__main__":
    main()
