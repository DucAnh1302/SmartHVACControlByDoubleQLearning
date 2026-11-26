"""
Unit tests for MultiDeviceRewardFunction.
Verifies signed additive decomposition, weight configurations, and safety penalties.
"""

import pytest
import numpy as np

from src.reward.multi_device_reward import MultiDeviceRewardFunction


def test_signed_additive_decomposition():
    """Verify that total_reward equals the exact sum of all 7 signed components."""
    rf = MultiDeviceRewardFunction()

    test_states = [
        (25.0, 50.0, 600.0, 15.0, 1.2, 12, 0, 1, False, False),  # Optimal comfort & IAQ
        (28.0, 70.0, 1300.0, 45.0, 3.5, 14, 0, 1, True, False),   # High temp, RH, CO2, PM2.5
        (17.5, 35.0, 500.0, 10.0, 0.0, 3, 0, 0, False, False),    # Extreme cold freezing hazard
        (30.5, 60.0, 1900.0, 110.0, 4.0, 15, 0, 1, True, True),   # Severe multi-hazard
    ]

    for temp, rh, co2, pm, power, hour, dow, is_work, changed, viol in test_states:
        total_r, b = rf.calculate_reward(temp, rh, co2, pm, power, hour, dow, is_work, changed, viol)
        comp_sum = (
            b["reward_temp"] +
            b["reward_humidity"] +
            b["reward_co2"] +
            b["reward_air_quality"] +
            b["reward_energy"] +
            b["reward_switching"] +
            b["reward_safety"]
        )
        assert np.isclose(total_r, comp_sum, atol=1e-4), (
            f"Signed decomposition mismatch for T={temp}: total={total_r}, sum={comp_sum}"
        )


def test_preference_profiles():
    """Verify that preference configurations update weights appropriately."""
    rf = MultiDeviceRewardFunction(preference="balanced")
    assert rf.w_temp == 1.00
    assert rf.w_energy == 0.12

    rf.configure_weights("comfort_priority")
    assert rf.w_temp == 2.00
    assert rf.w_energy == 0.04

    rf.configure_weights("energy_priority")
    assert rf.w_temp == 0.60
    assert rf.w_energy == 0.25

    rf.configure_weights("air_quality_priority")
    assert rf.w_co2 == 0.70
    assert rf.w_pm25 == 0.60

    with pytest.raises(ValueError):
        rf.configure_weights("invalid_profile")


def test_workhour_temporal_reward_weighting():
    """Verify that is_workhour correctly scales occupancy-dependent rewards (workhour vs night/weekend)."""
    rf = MultiDeviceRewardFunction()

    # Optimal temperature (25°C) gives +0.50 * occ_mult
    # When is_workhour=1, occ_mult = 2.0 -> r_temp = +1.00
    _, b_work = rf.calculate_reward(temp_c=25.0, humidity_rh=50.0, co2_ppm=600.0, pm25_ug_m3=15.0,
                                   total_power_kw=1.0, hour=14, dayofweek=2, is_workhour=1)
    assert np.isclose(b_work["reward_temp"], 1.00)

    # When is_workhour=0, occ_mult = 0.5 -> r_temp = +0.25
    _, b_off = rf.calculate_reward(temp_c=25.0, humidity_rh=50.0, co2_ppm=600.0, pm25_ug_m3=15.0,
                                  total_power_kw=1.0, hour=2, dayofweek=2, is_workhour=0)
    assert np.isclose(b_off["reward_temp"], 0.25)

    # When is_workhour is None, it automatically infers from dayofweek & hour
    # Tuesday 14:00 (workhour) -> 1.00
    _, b_inferred_work = rf.calculate_reward(temp_c=25.0, humidity_rh=50.0, co2_ppm=600.0, pm25_ug_m3=15.0,
                                            total_power_kw=1.0, hour=14, dayofweek=1, is_workhour=None)
    assert np.isclose(b_inferred_work["reward_temp"], 1.00)

    # Saturday 14:00 (weekend) -> 0.25
    _, b_inferred_weekend = rf.calculate_reward(temp_c=25.0, humidity_rh=50.0, co2_ppm=600.0, pm25_ug_m3=15.0,
                                               total_power_kw=1.0, hour=14, dayofweek=5, is_workhour=None)
    assert np.isclose(b_inferred_weekend["reward_temp"], 0.25)
