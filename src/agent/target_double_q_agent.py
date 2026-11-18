"""
Approved Target Double Q-Learning Agent for Smart HVAC Control.
Specifications:
- 600 States × 5 Actions = 3,000 Q-entries per table
- Two independent estimators Q_A and Q_B (Hado van Hasselt, 2010)
- Visit count matrix N(s, a) tracking state-action space exploration
- Seeded epsilon-greedy exploration with deterministic tie-breaking
- Optimistic initialization option
- Complete training diagnostics and state coverage tracking
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import pickle
from pathlib import Path
import numpy as np

from src.agent.base_agent import BaseAgent


class TargetDoubleQLearningAgent(BaseAgent):
    """
    Target Double Q-Learning Agent (600 states, 5 actions).
    Decouples action selection from action evaluation to eliminate maximization bias.
    """

    def __init__(
        self,
        n_states: int = 600,
        n_actions: int = 5,
        learning_rate: float = 0.08,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        optimistic_init: float = 0.0,
        seed: int = 42,
        name: str = "Double Q-Learning (Target 600s)"
    ):
        super().__init__(n_states=n_states, n_actions=n_actions, name=name)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Two independent Q-tables (600 x 5)
        self.q_table_a = np.full((n_states, n_actions), fill_value=optimistic_init, dtype=np.float64)
        self.q_table_b = np.full((n_states, n_actions), fill_value=optimistic_init, dtype=np.float64)

        # Visit count matrix N(s, a)
        self.visit_counts = np.zeros((n_states, n_actions), dtype=np.int32)

        # Diagnostics
        self.training_steps = 0
        self.td_errors: List[float] = []

    @property
    def combined_q_table(self) -> np.ndarray:
        """Combined action-value function for decision making."""
        return (self.q_table_a + self.q_table_b) / 2.0

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """
        Epsilon-greedy action selection with deterministic tie-breaking.
        """
        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Greedy choice based on combined Q-table
        q_vals = self.combined_q_table[state]
        max_q = np.max(q_vals)
        # Deterministic tie-breaking: select candidates with max value
        best_indices = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
        if len(best_indices) == 1:
            return int(best_indices[0])
        # Seeded deterministic tie-break
        return int(self.rng.choice(best_indices))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """
        Double Q-Learning update:
        Randomly update Q_A using Q_B evaluation, or Q_B using Q_A evaluation.
        """
        self.visit_counts[state, action] += 1
        self.training_steps += 1

        if self.rng.random() < 0.5:
            # Update Q_A
            if not done:
                # Action selection from Q_A
                q_a_next = self.q_table_a[next_state]
                max_a = np.max(q_a_next)
                best_actions = np.where(np.isclose(q_a_next, max_a, atol=1e-8))[0]
                best_action = int(self.rng.choice(best_actions))
                # Value evaluation from Q_B
                eval_next_q = self.q_table_b[next_state, best_action]
            else:
                eval_next_q = 0.0

            td_target = reward + self.gamma * eval_next_q
            td_error = td_target - self.q_table_a[state, action]
            self.q_table_a[state, action] += self.lr * td_error

        else:
            # Update Q_B
            if not done:
                # Action selection from Q_B
                q_b_next = self.q_table_b[next_state]
                max_b = np.max(q_b_next)
                best_actions = np.where(np.isclose(q_b_next, max_b, atol=1e-8))[0]
                best_action = int(self.rng.choice(best_actions))
                # Value evaluation from Q_A
                eval_next_q = self.q_table_a[next_state, best_action]
            else:
                eval_next_q = 0.0

            td_target = reward + self.gamma * eval_next_q
            td_error = td_target - self.q_table_b[state, action]
            self.q_table_b[state, action] += self.lr * td_error

        self.td_errors.append(abs(float(td_error)))

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_coverage_percentage(self) -> float:
        """Percentage of the 3,000 state-action pairs visited at least once."""
        visited = np.sum(self.visit_counts > 0)
        total = self.n_states * self.n_actions
        return float((visited / total) * 100.0)

    def get_avg_max_q(self) -> float:
        """Average maximum Q-value across all 600 states."""
        return float(np.mean(np.max(self.combined_q_table, axis=1)))

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic metrics for reporting."""
        return {
            "name": self.name,
            "training_steps": self.training_steps,
            "coverage_pct": round(self.get_coverage_percentage(), 2),
            "avg_max_q": round(self.get_avg_max_q(), 3),
            "current_epsilon": round(float(self.epsilon), 4),
            "mean_td_error": round(float(np.mean(self.td_errors[-200:])), 4) if self.td_errors else 0.0
        }
