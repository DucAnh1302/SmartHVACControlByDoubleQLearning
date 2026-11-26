"""
Multi-Device Target Double Q-Learning Agent.
Manages dual independent action-value tables (Q_A, Q_B) for 720 discrete states
and 12 coordinated multi-device HVAC action templates.
Includes visit tracking N(s, a), seeded tie-breaking, schema versioning,
and decoupled action selection to eliminate maximization bias.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone
import pickle
from pathlib import Path
import numpy as np

from src.agent.base_agent import BaseAgent


SCHEMA_VERSION_MULTI_DEVICE = "2.0-multi-device-720s12a"


class MultiDeviceDoubleQAgent(BaseAgent):
    """
    Target Double Q-Learning Agent for Multi-Device HVAC Digital Twin.
    Space: 720 States x 12 Action Templates = 8,640 Q-values per table.
    """

    def __init__(
        self,
        n_states: int = 720,
        n_actions: int = 12,
        learning_rate: float = 0.08,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        optimistic_init: float = 0.0,
        seed: int = 42,
        name: str = "Multi-Device Double Q-Learning"
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

        # Dual Q-Tables (720 x 12)
        self.q_table_a = np.full((n_states, n_actions), fill_value=optimistic_init, dtype=np.float64)
        self.q_table_b = np.full((n_states, n_actions), fill_value=optimistic_init, dtype=np.float64)

        # Visit count matrix N(s, a)
        self.visit_counts = np.zeros((n_states, n_actions), dtype=np.int32)

        # Training diagnostics
        self.training_steps = 0
        self.update_count = 0
        self.td_errors: List[float] = []

        # Versioning metadata
        self.metadata = {
            "schema_version": SCHEMA_VERSION_MULTI_DEVICE,
            "algorithm": "DoubleQLearning",
            "n_states": n_states,
            "n_actions": n_actions,
            "seed": seed,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self.is_learning_agent = True
        self.last_action_is_exploring = False

    @property
    def combined_q_table(self) -> np.ndarray:
        """Combined action-value estimate (Q_A + Q_B) / 2."""
        return (self.q_table_a + self.q_table_b) / 2.0

    def select_action(self, state: int, evaluate: bool = False, return_info: bool = False) -> Union[int, Tuple[int, Dict[str, Any]]]:
        """
        Epsilon-greedy action selection with deterministic tie-breaking.
        If evaluate=True, strictly greedy exploitation (epsilon ignored).
        Tracks exploration vs exploitation without redundant RNG sampling.
        """
        state_clamped = min(max(state, 0), self.n_states - 1)

        if not evaluate and self.rng.random() < self.epsilon:
            self.last_action_is_exploring = True
            action = int(self.rng.integers(0, self.n_actions))
        else:
            self.last_action_is_exploring = False
            # Greedy choice based on combined Q-table (row slice avoids full table alloc)
            q_vals = 0.5 * (self.q_table_a[state_clamped] + self.q_table_b[state_clamped])
            max_q = np.max(q_vals)
            best_indices = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
            if len(best_indices) == 1:
                action = int(best_indices[0])
            else:
                # Deterministic seeded tie-break
                action = int(self.rng.choice(best_indices))

        if return_info:
            return action, {"is_exploring": self.last_action_is_exploring}
        return action

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """
        Double Q-Learning update rule:
        Decouples action selection from value evaluation.
        """
        s = min(max(state, 0), self.n_states - 1)
        a = min(max(action, 0), self.n_actions - 1)
        s_next = min(max(next_state, 0), self.n_states - 1)

        self.visit_counts[s, a] += 1
        self.training_steps += 1
        self.update_count += 1

        if self.rng.random() < 0.5:
            # Update Q_A: select greedy action from Q_A, evaluate using Q_B
            if not done:
                q_a_next = self.q_table_a[s_next]
                max_a = np.max(q_a_next)
                best_actions = np.where(np.isclose(q_a_next, max_a, atol=1e-8))[0]
                best_action = int(self.rng.choice(best_actions))
                eval_next_q = self.q_table_b[s_next, best_action]
            else:
                eval_next_q = 0.0

            td_target = reward + self.gamma * eval_next_q
            td_error = td_target - self.q_table_a[s, a]
            self.q_table_a[s, a] += self.lr * td_error
        else:
            # Update Q_B: select greedy action from Q_B, evaluate using Q_A
            if not done:
                q_b_next = self.q_table_b[s_next]
                max_b = np.max(q_b_next)
                best_actions = np.where(np.isclose(q_b_next, max_b, atol=1e-8))[0]
                best_action = int(self.rng.choice(best_actions))
                eval_next_q = self.q_table_a[s_next, best_action]
            else:
                eval_next_q = 0.0

            td_target = reward + self.gamma * eval_next_q
            td_error = td_target - self.q_table_b[s, a]
            self.q_table_b[s, a] += self.lr * td_error

        self.td_errors.append(abs(float(td_error)))

    def decay_epsilon(self) -> None:
        """Decay exploration parameter epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_coverage_percentage(self) -> float:
        """Percentage of state-action entries visited at least once."""
        visited = np.sum(self.visit_counts > 0)
        total = self.n_states * self.n_actions
        return float((visited / total) * 100.0)

    def get_avg_max_q(self) -> float:
        """Average maximum Q-value across all 720 states."""
        return float(np.mean(np.max(self.combined_q_table, axis=1)))

    def save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Save model with metadata and schema compatibility checks."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self.metadata,
            "q_table_a": self.q_table_a,
            "q_table_b": self.q_table_b,
            "visit_counts": self.visit_counts,
            "training_steps": self.training_steps,
            "update_count": self.update_count,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
            "seed": self.seed
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[{self.name}] Checkpoint saved with schema {self.metadata['schema_version']} to: {path}")

    @classmethod
    def load_checkpoint(cls, filepath: Union[str, Path]) -> "MultiDeviceDoubleQAgent":
        """Load model with schema validation."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        with open(path, "rb") as f:
            obj = pickle.load(f)

        # Handle legacy vs new checkpoint formats
        if isinstance(obj, dict) and "metadata" in obj:
            schema = obj["metadata"].get("schema_version", "")
            if schema != SCHEMA_VERSION_MULTI_DEVICE:
                raise ValueError(
                    f"Incompatible checkpoint schema '{schema}'! Expected '{SCHEMA_VERSION_MULTI_DEVICE}'."
                )
            agent = cls(
                n_states=obj["metadata"]["n_states"],
                n_actions=obj["metadata"]["n_actions"],
                learning_rate=obj["lr"],
                gamma=obj["gamma"],
                epsilon=obj["epsilon"],
                seed=obj["seed"]
            )
            agent.q_table_a = obj["q_table_a"]
            agent.q_table_b = obj["q_table_b"]
            agent.visit_counts = obj["visit_counts"]
            agent.training_steps = obj["training_steps"]
            agent.update_count = obj.get("update_count", agent.training_steps)
            agent.metadata = obj["metadata"]
            print(f"[{agent.name}] Successfully loaded checkpoint with {agent.training_steps} steps.")
            return agent
        elif isinstance(obj, cls):
            return obj
        else:
            raise TypeError(f"Unrecognized checkpoint format in: {filepath}")


class SingleQMultiDeviceAgent(BaseAgent):
    """Single Q-Learning baseline agent for the 720-state multi-device environment."""

    def __init__(
        self,
        n_states: int = 720,
        n_actions: int = 12,
        learning_rate: float = 0.08,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        seed: int = 42,
        name: str = "Multi-Device Single Q-Learning (Baseline)"
    ):
        super().__init__(n_states=n_states, n_actions=n_actions, name=name)
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.visit_counts = np.zeros((n_states, n_actions), dtype=np.int32)
        self.training_steps = 0
        self.update_count = 0
        self.is_learning_agent = True
        self.last_action_is_exploring = False

    def select_action(self, state: int, evaluate: bool = False, return_info: bool = False) -> Union[int, Tuple[int, Dict[str, Any]]]:
        s = min(max(state, 0), self.n_states - 1)
        if not evaluate and self.rng.random() < self.epsilon:
            self.last_action_is_exploring = True
            action = int(self.rng.integers(0, self.n_actions))
        else:
            self.last_action_is_exploring = False
            q_vals = self.q_table[s]
            max_q = np.max(q_vals)
            best_indices = np.where(np.isclose(q_vals, max_q, atol=1e-8))[0]
            if len(best_indices) == 1:
                action = int(best_indices[0])
            else:
                action = int(self.rng.choice(best_indices))

        if return_info:
            return action, {"is_exploring": self.last_action_is_exploring}
        return action

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        s = min(max(state, 0), self.n_states - 1)
        a = min(max(action, 0), self.n_actions - 1)
        s_next = min(max(next_state, 0), self.n_states - 1)

        self.visit_counts[s, a] += 1
        self.training_steps += 1
        self.update_count += 1

        next_max = 0.0 if done else np.max(self.q_table[s_next])
        td_target = reward + self.gamma * next_max
        td_error = td_target - self.q_table[s, a]
        self.q_table[s, a] += self.lr * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
