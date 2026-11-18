"""
Unit tests for AdvancedHVACRewardFunction signed decomposition and preference weighting.
"""

import pytest
import numpy as np

from src.reward.advanced_reward import AdvancedHVACRewardFunction


def test_signed_decomposition_identity():
    """Verify that total_reward is strictly equal to the sum of its signed components."""
    reward_fn = AdvancedHVACRewardFunction()

    test_cases = [
        (25.0, 50.0, 10.0, 1, 12, False),  # Preferred zone, workhour
        (22.0, 70.0, 20.0, 1, 14, True),   # Moderately cold, high RH, action changed
        (19.0, 60.0, 0.0, 0, 22, False),   # Severe cold violation (<21°C)
        (30.5, 45.0, 30.0, 1, 15, True),   # Severe hot violation (>29°C)
    ]

    for temp_c, rh, hvac_kw, is_work, hour, changed in test_cases:
        total_r, b = reward_fn.calculate_reward(temp_c, rh, hvac_kw, is_work, hour, changed)
        sum_components = (
            b["reward_comfort"] +
            b["reward_safety"] +
            b["reward_energy"] +
            b["reward_switching"] +
            b["reward_humidity"]
        )
        assert np.isclose(total_r, sum_components, atol=1e-6), (
            f"Signed decomposition mismatch for T={temp_c}: total={total_r}, sum={sum_components}"
        )


def test_safety_penalty_isolated():
    """Verify that safe conditions yield reward_safety == 0, and unsafe conditions yield reward_safety < 0."""
    reward_fn = AdvancedHVACRewardFunction()

    # Within [21, 29]
    _, b_safe = reward_fn.calculate_reward(25.0, 50.0, 10.0, 1, 12)
    assert b_safe["reward_safety"] == 0.0

    # Below 21°C
    _, b_freeze = reward_fn.calculate_reward(20.0, 50.0, 0.0, 1, 12)
    assert b_freeze["reward_safety"] < -5.0

    # Above 29°C
    _, b_overheat = reward_fn.calculate_reward(30.0, 50.0, 0.0, 1, 12)
    assert b_overheat["reward_safety"] < -5.0


def test_preference_profiles():
    """Verify that preference configurations adjust weights as specified."""
    rf = AdvancedHVACRewardFunction(preference="balanced")
    assert rf.w_comfort == 1.00
    assert rf.w_energy == 0.08

    rf.configure_weights("comfort_priority")
    assert rf.w_comfort == 2.00
    assert rf.w_energy == 0.03

    rf.configure_weights("energy_priority")
    assert rf.w_comfort == 0.50
    assert rf.w_energy == 0.18

    with pytest.raises(ValueError):
        rf.configure_weights("invalid_profile")
