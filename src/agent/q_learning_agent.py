"""
Standard Q-Learning Agent (Baseline) for Smart HVAC Control.
"""

from typing import Dict, List, Optional
import numpy as np

from src.agent.base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Standard Q-Learning (Watkins, 1989):
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    Features independent seeded numpy.random.Generator for complete reproducibility.
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
        seed: Optional[int] = 42,
        name: str = "Q-Learning"
    ):
        super().__init__(n_states, n_actions, name)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Single Q-Table initialized to zeros
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        
        # Tracking metrics
        self.training_steps = 0
        self.td_errors: List[float] = []

    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy policy with independent seeded RNG."""
        if not evaluate and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Greedy action with deterministic seeded tie-breaking
        q_vals = self.q_table[state]
        max_q = np.max(q_vals)
        best_actions = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """Update Q-value with standard Bellman equation."""
        best_next_q = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]

        self.q_table[state, action] += self.lr * td_error
        self.td_errors.append(abs(float(td_error)))
        self.training_steps += 1

    def decay_epsilon(self) -> None:
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_avg_max_q(self) -> float:
        """Return average maximum Q-value across all states."""
        return float(np.mean(np.max(self.q_table, axis=1)))
