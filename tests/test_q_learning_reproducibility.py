"""
Unit tests for Q-Learning independent RNG and seed reproducibility.
Verifies that QLearningAgent produces deterministic, reproducible trajectories
under identical seeds and distinct trajectories under different seeds,
without relying on global random state.
"""

import pytest
import numpy as np
from src.agent.q_learning_agent import QLearningAgent


def test_q_learning_identical_seed_reproducibility():
    """Verify that identical seeds produce identical action sequences and Q-tables."""
    agent1 = QLearningAgent(n_states=20, n_actions=5, seed=42)
    agent2 = QLearningAgent(n_states=20, n_actions=5, seed=42)

    # 1. Action selection sequence under exploration (epsilon=1.0)
    actions1 = [agent1.select_action(state=s % 20, evaluate=False) for s in range(100)]
    actions2 = [agent2.select_action(state=s % 20, evaluate=False) for s in range(100)]

    assert actions1 == actions2, "Identical seeds must produce identical exploration action sequences!"

    # 2. Update sequence
    for step in range(50):
        s = step % 20
        a = actions1[step]
        r = float(10.0 - (step % 5))
        s_next = (s + 1) % 20
        done = (step == 49)

        agent1.update(s, a, r, s_next, done)
        agent2.update(s, a, r, s_next, done)

    assert np.allclose(agent1.q_table, agent2.q_table), "Identical updates with identical seeds must yield identical Q-tables!"


def test_q_learning_different_seeds_diversity():
    """Verify that different seeds produce different exploration trajectories."""
    agent_a = QLearningAgent(n_states=20, n_actions=5, seed=123)
    agent_b = QLearningAgent(n_states=20, n_actions=5, seed=987)

    actions_a = [agent_a.select_action(state=0, evaluate=False) for _ in range(50)]
    actions_b = [agent_b.select_action(state=0, evaluate=False) for _ in range(50)]

    assert actions_a != actions_b, "Different seeds must produce different exploration sequences!"


def test_q_learning_does_not_pollute_global_np_random():
    """Verify that running QLearningAgent does not advance or depend on global np.random state."""
    np.random.seed(9999)
    global_val_before = np.random.rand()

    # Reset global seed to 9999
    np.random.seed(9999)

    # Run agent operations with independent seed
    agent = QLearningAgent(n_states=10, n_actions=3, seed=777)
    _ = [agent.select_action(state=0, evaluate=False) for _ in range(30)]

    # Global np.random should still produce the exact same next float
    global_val_after = np.random.rand()
    assert global_val_before == global_val_after, "QLearningAgent must not consume global np.random state!"
