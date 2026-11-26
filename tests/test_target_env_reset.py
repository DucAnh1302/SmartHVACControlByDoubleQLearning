"""
Unit tests for target environment reset and episode isolation.
Verifies that reset() completely restores every mutable episode state
and eliminates spurious switching penalties on step 0.
"""

import pytest
import pandas as pd
import numpy as np

from src.environment.target_env import TargetHVACEnv, TARGET_ACTIONS


@pytest.fixture
def dummy_data():
    """Create dummy dataframe mimicking domain telemetry."""
    n = 100
    dates = pd.date_range("2018-07-01", periods=n, freq="h")
    return pd.DataFrame({
        "datetime": dates,
        "temp_c": np.linspace(24.0, 27.0, n),
        "humidity_rh": np.linspace(50.0, 65.0, n),
        "internal_heat_kw": np.ones(n) * 2.5,
        "ambient_temp_est": np.linspace(28.0, 34.0, n),
        "ambient_rh_est": np.ones(n) * 70.0,
        "hour": [d.hour for d in dates],
        "dayofweek": [d.dayofweek for d in dates],
        "is_workhour": [1 if 8 <= d.hour <= 18 else 0 for d in dates]
    })


def test_reset_episode_isolation(dummy_data):
    """Test that reset() fully restores all episode state variables."""
    env = TargetHVACEnv(data_df=dummy_data, episode_length=10, seed=42)

    # 1. Step with action 4 (WINDOW_OPEN)
    state, info = env.reset(start_idx=0, initial_temp=26.0)
    assert env.window_state == 0
    assert env.last_action is None
    assert env.episode_energy_kwh == 0.0
    assert env.episode_switching_count == 0

    next_state, r1, done, trunc, step_info = env.step(4)  # WINDOW_OPEN
    assert env.window_state == 1
    assert step_info["window_open"] == 1
    assert env.last_action == 4

    # Run remaining steps
    for _ in range(9):
        env.step(1)

    assert env.current_step == 10
    assert env.episode_energy_kwh > 0.0
    assert env.episode_switching_count > 0

    # 2. Reset and verify complete isolation
    state2, info2 = env.reset(start_idx=5, initial_temp=25.0)
    assert env.current_step == 0
    assert env.window_state == 0
    assert env.last_action is None
    assert env.prev_hvac_kw == 0.0
    assert env.prev_setpoint is None
    assert env.episode_energy_kwh == 0.0
    assert env.episode_switching_count == 0
    assert info2["step"] == 0
    assert info2["window_open"] == 0
    assert info2["episode_energy_kwh"] == 0.0
    assert info2["episode_switching_count"] == 0


def test_no_step_0_switching_penalty(dummy_data):
    """Test that step 0 never incurs a switching penalty for ANY chosen action."""
    for action in TARGET_ACTIONS.keys():
        env = TargetHVACEnv(data_df=dummy_data, episode_length=5, seed=100 + action)
        state, info = env.reset(start_idx=10, initial_temp=25.5)

        next_state, r, done, trunc, step_info = env.step(action)

        # Switching cost on step 0 must be 0.0
        assert step_info.get("switching_cost", 0.0) == 0.0, f"Action {action} had non-zero switching cost on step 0!"
        assert env.episode_switching_count == 0, f"Action {action} caused switching count increment on step 0!"


def test_seed_reproducibility_on_reset(dummy_data):
    """Test that resetting with the same seed yields identical initial states."""
    env = TargetHVACEnv(data_df=dummy_data, episode_length=10)

    s1, info1 = env.reset(seed=999)
    s2, info2 = env.reset(seed=999)

    assert s1 == s2
    assert info1["temp_c"] == info2["temp_c"]
    assert info1["datetime"] == info2["datetime"]
