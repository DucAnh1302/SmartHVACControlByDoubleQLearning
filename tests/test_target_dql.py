"""
Unit tests for Target Double Q-Learning Agent Architecture.
Verifies the 600-state x 5-action space, decoupled Q-tables, visit counting, and tie-breaking.
"""

import pytest
import numpy as np
from src.agent.target_double_q_agent import TargetDoubleQLearningAgent


def test_agent_initialization():
    """Verify target state space (600), action space (5), and dual Q-table shapes."""
    agent = TargetDoubleQLearningAgent(
        n_states=600,
        n_actions=5,
        optimistic_init=0.0,
        seed=42
    )

    assert agent.n_states == 600
    assert agent.n_actions == 5
    assert agent.q_table_a.shape == (600, 5)
    assert agent.q_table_b.shape == (600, 5)
    assert agent.visit_counts.shape == (600, 5)
    assert np.all(agent.q_table_a == 0.0)
    assert np.all(agent.q_table_b == 0.0)
    assert np.all(agent.visit_counts == 0)


def test_combined_q_table():
    """Verify combined Q-table computes average (Q_A + Q_B) / 2 accurately."""
    agent = TargetDoubleQLearningAgent(n_states=600, n_actions=5, seed=42)
    agent.q_table_a[10, 2] = 4.0
    agent.q_table_b[10, 2] = 2.0

    combined = agent.combined_q_table
    assert combined[10, 2] == 3.0
    assert combined[10, 0] == 0.0


def test_action_selection_greedy_and_tie_breaking():
    """Verify deterministic tie-breaking and greedy selection."""
    agent = TargetDoubleQLearningAgent(n_states=600, n_actions=5, seed=42)

    # 1. Greedy choice with a clear maximum
    agent.q_table_a[50, 3] = 10.0
    agent.q_table_b[50, 3] = 10.0
    selected_action = agent.select_action(state=50, evaluate=True)
    assert selected_action == 3

    # 2. Tie-breaking between action 1 and action 4
    agent.q_table_a[60, 1] = 5.0
    agent.q_table_b[60, 1] = 5.0
    agent.q_table_a[60, 4] = 5.0
    agent.q_table_b[60, 4] = 5.0

    # With seed=42, evaluate=True must pick one of [1, 4] deterministically
    chosen = agent.select_action(state=60, evaluate=True)
    assert chosen in [1, 4]

    # Repeated calls with evaluate=True on fresh seeded agents should yield the exact same choice
    agent2 = TargetDoubleQLearningAgent(n_states=600, n_actions=5, seed=42)
    agent2.q_table_a[60, 1] = 5.0
    agent2.q_table_b[60, 1] = 5.0
    agent2.q_table_a[60, 4] = 5.0
    agent2.q_table_b[60, 4] = 5.0
    assert agent2.select_action(state=60, evaluate=True) == chosen


def test_double_q_learning_update_rule():
    """Verify that update updates only one table per step and increments visit counts."""
    agent = TargetDoubleQLearningAgent(
        n_states=600,
        n_actions=5,
        learning_rate=0.1,
        gamma=0.9,
        seed=100
    )

    state = 120
    action = 2
    reward = 1.0
    next_state = 125
    done = False

    initial_visits = agent.visit_counts[state, action]
    q_a_before = agent.q_table_a[state, action]
    q_b_before = agent.q_table_b[state, action]

    agent.update(state, action, reward, next_state, done)

    # Visit count must increment
    assert agent.visit_counts[state, action] == initial_visits + 1
    assert agent.training_steps == 1

    # Exactly one table should change (or both stay >= 0)
    q_a_after = agent.q_table_a[state, action]
    q_b_after = agent.q_table_b[state, action]
    changed_a = (q_a_after != q_a_before)
    changed_b = (q_b_after != q_b_before)

    assert changed_a or changed_b
    assert not (changed_a and changed_b)  # In any single step, only Q_A or Q_B is updated


def test_epsilon_decay():
    """Verify exploration rate decay and clipping at epsilon_min."""
    agent = TargetDoubleQLearningAgent(
        n_states=600,
        n_actions=5,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.90
    )

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.90)

    # Decay many times until min is reached
    for _ in range(50):
        agent.decay_epsilon()
    assert agent.epsilon == 0.05


def test_diagnostics_and_coverage():
    """Verify coverage percentage calculation and diagnostic dictionary."""
    agent = TargetDoubleQLearningAgent(n_states=600, n_actions=5)

    # Initial coverage: 0%
    assert agent.get_coverage_percentage() == 0.0

    # Mark 30 distinct state-action pairs as visited
    for i in range(30):
        agent.visit_counts[i, 0] = 1

    # 30 / (600 * 5) = 30 / 3000 = 1.0%
    assert agent.get_coverage_percentage() == 1.0

    diag = agent.get_diagnostics()
    assert "coverage_pct" in diag
    assert diag["coverage_pct"] == 1.0
    assert "avg_max_q" in diag
    assert "current_epsilon" in diag


def test_target_env_model_config_reflection():
    """
    Verify that TargetHVACEnv accurately exposes its underlying physical model
    parameters, fitting state, and provenance via get_model_config().
    """
    import pandas as pd
    from src.environment.target_env import TargetHVACEnv
    from src.environment.greybox_model import GreyBoxThermalModel
    from src.data.provenance import ProvenanceType

    df_dummy = pd.DataFrame({
        "temp_c": [25.0] * 24,
        "humidity_rh": [60.0] * 24,
        "internal_heat_kw": [2.0] * 24,
        "ambient_temp_est": [30.0] * 24,
        "ambient_rh_est": [70.0] * 24,
        "hour": list(range(24)),
        "dayofweek": [0] * 24,
        "hvac_kw": [15.0] * 24
    })

    # Case 1: Un-fitted default
    env_unfitted = TargetHVACEnv(data_df=df_dummy)
    cfg_unfitted = env_unfitted.get_model_config()
    assert cfg_unfitted["thermal_model_is_fitted"] is False
    assert cfg_unfitted["thermal_model_provenance"] == ProvenanceType.SIMULATED_ASSUMED.value

    # Case 2: Explicitly fitted model
    thermal_model = GreyBoxThermalModel()
    thermal_model.is_fitted = True
    thermal_model.provenance = ProvenanceType.FITTED_WITH_CAVEAT
    thermal_model.parameters_calibrated = True

    env_fitted = TargetHVACEnv(data_df=df_dummy, thermal_model=thermal_model)
    cfg_fitted = env_fitted.get_model_config()
    assert cfg_fitted["thermal_model_is_fitted"] is True
    assert cfg_fitted["thermal_model_provenance"] == ProvenanceType.FITTED_WITH_CAVEAT.value

