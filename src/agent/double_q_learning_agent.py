"""
Double Q-Learning Agent (Proposed Method) for Smart HVAC Control.
Addresses the maximization bias (overestimation) of standard Q-learning.
"""

from typing import Dict, List, Optional
import numpy as np

from src.agent.base_agent import BaseAgent


class DoubleQLearningAgent(BaseAgent):
    """
    Double Q-Learning (Hado van Hasselt, 2010):
    Maintains two independent estimators Q_A and Q_B.
    Decouples action selection from action evaluation.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        name: str = "Double Q-Learning"
    ):
        super().__init__(n_states, n_actions, name)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Two independent Q-Tables
        self.q_table_a = np.zeros((n_states, n_actions), dtype=np.float64)
        self.q_table_b = np.zeros((n_states, n_actions), dtype=np.float64)

        # Tracking metrics
        self.training_steps = 0
        self.td_errors: List[float] = []

    @property
    def combined_q_table(self) -> np.ndarray:
        """Combined action-value function used for decision-making."""
        return (self.q_table_a + self.q_table_b) / 2.0

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy based on combined Q-values."""
        if not evaluate and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        # Greedy choice based on Q_A + Q_B
        combined_q = self.combined_q_table[state]
        max_q = np.max(combined_q)
        best_actions = np.where(combined_q == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """
        Decoupled update: randomly update either Q_A using Q_B evaluation,
        or Q_B using Q_A evaluation.
        """
        if np.random.rand() < 0.5:
            # Update Q_A: select action using Q_A, evaluate using Q_B
            if not done:
                q_a_next = self.q_table_a[next_state]
                max_next_a = np.max(q_a_next)
                best_actions = np.where(q_a_next == max_next_a)[0]
                best_action = np.random.choice(best_actions)
                eval_next_q = self.q_table_b[next_state, best_action]
            else:
                eval_next_q = 0.0

            td_target = reward + self.gamma * eval_next_q
            td_error = td_target - self.q_table_a[state, action]
            self.q_table_a[state, action] += self.lr * td_error

        else:
            # Update Q_B: select action using Q_B, evaluate using Q_A
            if not done:
                q_b_next = self.q_table_b[next_state]
                max_next_b = np.max(q_b_next)
                best_actions = np.where(q_b_next == max_next_b)[0]
                best_action = np.random.choice(best_actions)
                eval_next_q = self.q_table_a[next_state, best_action]
            else:
                eval_next_q = 0.0

            td_target = reward + self.gamma * eval_next_q
            td_error = td_target - self.q_table_b[state, action]
            self.q_table_b[state, action] += self.lr * td_error

        self.td_errors.append(abs(float(td_error)))
        self.training_steps += 1

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_avg_max_q(self) -> float:
        """Return average maximum Q-value across all states."""
        return float(np.mean(np.max(self.combined_q_table, axis=1)))
