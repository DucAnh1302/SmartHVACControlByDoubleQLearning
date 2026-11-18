"""
Base Agent Interface for Smart HVAC Reinforcement Learning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pickle
from pathlib import Path


class BaseAgent(ABC):
    """Abstract Base Class for RL and Rule-based HVAC controllers."""

    def __init__(self, n_states: int, n_actions: int, name: str = "BaseAgent"):
        self.n_states = n_states
        self.n_actions = n_actions
        self.name = name

    @abstractmethod
    def select_action(self, state: int, evaluate: bool = False) -> int:
        """Select action given state. If evaluate=True, act greedily."""
        pass

    @abstractmethod
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> None:
        """Update value function / policy from experience transition."""
        pass

    def save(self, filepath: str) -> None:
        """Serialize agent state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[{self.name}] Model saved to: {path}")

    @classmethod
    def load(cls, filepath: str) -> "BaseAgent":
        """Load agent instance from disk."""
        with open(filepath, "rb") as f:
            agent = pickle.load(f)
        print(f"[{agent.name}] Model loaded successfully from: {filepath}")
        return agent
