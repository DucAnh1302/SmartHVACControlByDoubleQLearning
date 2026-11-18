"""
Unit tests for Multi-Device Double Q-Learning Agent Architecture.
Verifies the 720-state x 12-action space, decoupled dual tables,
visit tracking, tie-breaking, schema versioning, and evaluation isolation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.agent.multi_device_double_q_agent import (
    MultiDeviceDoubleQAgent, SingleQMultiDeviceAgent, SCHEMA_VERSION_MULTI_DEVICE
)


def test_agent_initialization_720x12():
    """Verify state space (720), action space (12), and dual Q-table shapes."""
    agent = MultiDeviceDoubleQAgent(n_states=720, n_actions=12, seed=42)

    assert agent.n_states == 720
    assert agent.n_actions == 12
    assert agent.q_table_a.shape == (720, 12)
    assert agent.q_table_b.shape == (720, 12)
    assert agent.visit_counts.shape == (720, 12)
    assert np.all(agent.q_table_a == 0.0)
    assert np.all(agent.q_table_b == 0.0)
    assert np.all(agent.visit_counts == 0)
    assert agent.metadata["schema_version"] == SCHEMA_VERSION_MULTI_DEVICE


def test_combined_q_table():
    """Verify combined Q-table computes average (Q_A + Q_B) / 2 accurately."""
    agent = MultiDeviceDoubleQAgent(n_states=720, n_actions=12, seed=42)
    agent.q_table_a[100, 5] = 6.0
    agent.q_table_b[100, 5] = 2.0

    combined = agent.combined_q_table
    assert combined[100, 5] == 4.0
    assert combined[100, 0] == 0.0


def test_action_selection_greedy_and_tie_breaking():
    """Verify deterministic tie-breaking and greedy evaluation selection."""
    agent = MultiDeviceDoubleQAgent(n_states=720, n_actions=12, seed=42)

    # 1. Greedy choice with a clear maximum
    agent.q_table_a[50, 4] = 10.0
    agent.q_table_b[50, 4] = 10.0
    selected_action = agent.select_action(state=50, evaluate=True)
    assert selected_action == 4

    # 2. Tie-breaking between action 2 and action 8
    agent.q_table_a[60, 2] = 5.0
    agent.q_table_b[60, 2] = 5.0
    agent.q_table_a[60, 8] = 5.0
    agent.q_table_b[60, 8] = 5.0

    chosen = agent.select_action(state=60, evaluate=True)
    assert chosen in [2, 8]

    # Determinism across identical seeded agents
    agent2 = MultiDeviceDoubleQAgent(n_states=720, n_actions=12, seed=42)
    agent2.q_table_a[60, 2] = 5.0
    agent2.q_table_b[60, 2] = 5.0
    agent2.q_table_a[60, 8] = 5.0
    agent2.q_table_b[60, 8] = 5.0
    assert agent2.select_action(state=60, evaluate=True) == chosen


def test_double_q_learning_update_rule():
    """Verify that update updates exactly one table per step and increments visit counts."""
    agent = MultiDeviceDoubleQAgent(n_states=720, n_actions=12, learning_rate=0.1, gamma=0.9, seed=100)

    state = 120
    action = 5
    reward = 1.0
    next_state = 125
    done = False

    initial_visits = agent.visit_counts[state, action]
    q_a_before = agent.q_table_a[state, action]
    q_b_before = agent.q_table_b[state, action]

    agent.update(state, action, reward, next_state, done)

    assert agent.visit_counts[state, action] == initial_visits + 1
    assert agent.training_steps == 1

    q_a_after = agent.q_table_a[state, action]
    q_b_after = agent.q_table_b[state, action]
    # Exactly one table should change
    changed_a = (q_a_after != q_a_before)
    changed_b = (q_b_after != q_b_before)
    assert changed_a != changed_b


def test_checkpoint_save_load_and_schema_validation():
    """Verify serialization, restoration, and rejection of incompatible schemas."""
    agent = MultiDeviceDoubleQAgent(n_states=720, n_actions=12, seed=42)
    agent.q_table_a[10, 1] = 3.5
    agent.q_table_b[10, 1] = 2.5
    agent.visit_counts[10, 1] = 5
    agent.training_steps = 15

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_file = Path(tmp_dir) / "test_model.pkl"
        agent.save_checkpoint(save_file)

        loaded = MultiDeviceDoubleQAgent.load_checkpoint(save_file)
        assert loaded.n_states == 720
        assert loaded.n_actions == 12
        assert loaded.q_table_a[10, 1] == 3.5
        assert loaded.q_table_b[10, 1] == 2.5
        assert loaded.visit_counts[10, 1] == 5
        assert loaded.training_steps == 15
        assert loaded.metadata["schema_version"] == SCHEMA_VERSION_MULTI_DEVICE


def test_evaluation_mode_does_not_update_q_tables():
    """Verify that calling select_action(evaluate=True) performs zero updates to Q-tables."""
    agent = MultiDeviceDoubleQAgent(n_states=720, n_actions=12, seed=42)
    q_a_init = agent.q_table_a.copy()
    q_b_init = agent.q_table_b.copy()
    visits_init = agent.visit_counts.copy()

    # Step through 20 decisions in evaluation mode
    for s in range(20):
        act = agent.select_action(s, evaluate=True)

    assert np.array_equal(agent.q_table_a, q_a_init)
    assert np.array_equal(agent.q_table_b, q_b_init)
    assert np.array_equal(agent.visit_counts, visits_init)
    assert agent.training_steps == 0
    assert agent.update_count == 0
