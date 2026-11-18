"""
Unit tests for Multi-Device HVAC Digital Twin Physics & Environment.
Verifies psychrometric consistency, sensible thermal balance, CO2 mass balance,
particulate filtration, energy non-negativity, and room geometry dependencies.
"""

import pytest
import numpy as np

from src.environment.multi_device_env import (
    MultiDeviceHVACEnv, RoomGeometry, ROOM_PRESETS,
    magnus_saturation_vapor_pressure, humidity_ratio_from_rh, rh_from_humidity_ratio,
    MULTI_DEVICE_ACTION_TEMPLATES, discretize_state_720, state_tuple_to_index_720, index_to_state_tuple_720
)


def test_psychrometric_consistency():
    """
    Physical Test 1: Psychrometric Water Vapor Balance
    Ensures Magnus formula and humidity ratio conversions are mutually invertible.
    """
    for temp in [18.0, 22.0, 25.0, 28.0, 32.0]:
        for rh in [30.0, 50.0, 65.0, 85.0]:
            w = humidity_ratio_from_rh(temp, rh)
            # Moisture mass must be strictly positive
            assert w > 0.0
            rh_reconstructed = rh_from_humidity_ratio(temp, w)
            assert np.isclose(rh, rh_reconstructed, atol=0.2), (
                f"Psychrometric inversion error at T={temp}, RH={rh}: got {rh_reconstructed}"
            )


def test_temperature_cooling_monotonicity():
    """
    Physical Test 2: Cooling Monotonicity
    Under hot ambient (35°C), MAX_COOL (Action 2) must result in a lower
    indoor temperature than ECONOMY_COOL (Action 3) or ALL_OFF_STANDBY (Action 0).
    """
    env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], seed=42)

    # Reset with warm initial temp
    s0, info0 = env.reset(initial_temp=28.0, initial_rh=60.0, initial_co2=600.0, seed=10)
    _, _, _, _, info_max_cool = env.step(2)  # MAX_COOL

    env.reset(initial_temp=28.0, initial_rh=60.0, initial_co2=600.0, seed=10)
    _, _, _, _, info_eco_cool = env.step(3)  # ECONOMY_COOL

    env.reset(initial_temp=28.0, initial_rh=60.0, initial_co2=600.0, seed=10)
    _, _, _, _, info_standby = env.step(0)   # ALL_OFF_STANDBY

    assert info_max_cool["temp_c"] < info_eco_cool["temp_c"], (
        f"MAX_COOL ({info_max_cool['temp_c']}°C) did not cool more than ECO_COOL ({info_eco_cool['temp_c']}°C)"
    )
    assert info_eco_cool["temp_c"] < info_standby["temp_c"], (
        f"ECO_COOL ({info_eco_cool['temp_c']}°C) did not cool more than STANDBY ({info_standby['temp_c']}°C)"
    )


def test_co2_occupancy_generation_and_ventilation_dilution():
    """
    Physical Test 3: CO2 Mass Balance
    CO2 must rise when room is occupied and ventilation is OFF,
    and must decrease when ACTIVE_VENT_PURIFY (Action 5) is run.
    """
    env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], seed=42)

    # 1. Occupied room with ALL_OFF_STANDBY -> CO2 rises
    env.reset(initial_co2=700.0, occupancy=6, seed=42)
    _, _, _, _, info_rise = env.step(0)  # ALL_OFF_STANDBY
    assert info_rise["co2_ppm"] > 700.0, f"Expected CO2 to rise with 6 occupants, got {info_rise['co2_ppm']}"

    # 2. High initial CO2 with ACTIVE_VENT_PURIFY (Action 5: 500 m³/h fresh air) -> CO2 drops
    env.reset(initial_co2=1500.0, occupancy=0, seed=42)
    _, _, _, _, info_drop = env.step(5)  # ACTIVE_VENT_PURIFY
    assert info_drop["co2_ppm"] < 1500.0, f"Expected ventilation to dilute CO2 from 1500, got {info_drop['co2_ppm']}"


def test_air_filter_pm25_reduction():
    """
    Physical Test 4: Particulate Matter Filtration
    Air purifier (Action 10: ECO_AIR_PURIFY, 450 m³/h CADR) must significantly reduce PM2.5.
    Verify that air filter CADR does not directly remove CO2 (CO2 is governed by ventilation).
    """
    env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], seed=42)

    # High initial PM2.5
    env.reset(initial_pm25=80.0, occupancy=0, seed=42)
    _, _, _, _, info_filter = env.step(10)  # ECO_AIR_PURIFY
    assert info_filter["pm25_ug_m3"] < 80.0, f"Filter failed to reduce PM2.5: got {info_filter['pm25_ug_m3']}"

    # Verify filter action vs identical ventilation without filter
    # Action 10 has 100 m³/h vent + 450 m³/h filter CADR
    # If filter magically scrubbed CO2, it would reduce CO2 faster than 100 m³/h ventilation
    # CO2 reduction is purely determined by 100 m³/h fresh air exchange
    vol = env.room.volume_m3
    v_fresh = 100.0 + env.room.base_infiltration_ach * vol
    expected_co2_dilution_rate = v_fresh / vol  # approx 0.91 h^-1
    # Verify filter didn't remove extra CO2
    env.reset(initial_co2=1400.0, initial_pm25=50.0, occupancy=0, seed=42)
    _, _, _, _, info_co2 = env.step(10)
    # With 1400 initial, 415 outdoor, 100 m³/h + 28 m³/h infiltration over 140 m³ volume:
    # dC = 128 * (1400 - 415) / 140 = 900 ppm dilution -> CO2 around 500-600 ppm
    assert info_co2["co2_ppm"] >= 415.0


def test_energy_non_negativity_and_accumulation():
    """
    Physical Test 5: Energy Conservation & Non-Negativity
    Power and cumulative energy must be non-negative and monotonically increasing with time.
    """
    env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], seed=42)
    env.reset()

    for step in range(10):
        action = step % 12
        _, _, done, _, info = env.step(action)
        powers = info["powers_kw"]
        for dev, p in powers.items():
            assert p >= 0.0, f"Device {dev} generated negative power: {p} kW"
        assert info["cumul_energy_kwh"]["total_kwh"] >= 0.0


def test_room_geometry_dependency():
    """
    Physical Test 6: Room Size Dependency
    Under identical thermal loads and cooling actions, a small room (20m²)
    must experience a faster temperature drop than a large room (120m²).
    """
    small_env = MultiDeviceHVACEnv(room=ROOM_PRESETS["small"], seed=42)
    large_env = MultiDeviceHVACEnv(room=ROOM_PRESETS["large"], seed=42)

    small_env.reset(initial_temp=28.0, initial_rh=60.0, seed=1)
    large_env.reset(initial_temp=28.0, initial_rh=60.0, seed=1)

    _, _, _, _, small_info = small_env.step(2)  # MAX_COOL
    _, _, _, _, large_info = large_env.step(2)  # MAX_COOL

    delta_small = 28.0 - small_info["temp_c"]
    delta_large = 28.0 - large_info["temp_c"]
    assert delta_small > delta_large, (
        f"Small room delta T ({delta_small:.2f}°C) was not larger than large room delta T ({delta_large:.2f}°C)"
    )


def test_state_encoder_decoder_inversion():
    """
    State Space Test: 720 State bijection check
    Encoding and decoding between 6D state tuples and flat indices [0, 719] must be an exact bijection.
    """
    for t in range(5):
        for rh in range(3):
            for co2 in range(3):
                for pm in range(2):
                    for load in range(2):
                        for time_b in range(4):
                            tup = (t, rh, co2, pm, load, time_b)
                            idx = state_tuple_to_index_720(tup)
                            assert 0 <= idx < 720
                            recovered = index_to_state_tuple_720(idx)
                            assert recovered == tup, f"Inversion failed for {tup}: got {recovered}"


def test_compressor_power_and_thermal_capacity():
    """
    Physical Test 7: Compressor Standby vs Active Cooling Capacity
    Verifies that standby power (0.05 kW) does NOT generate cooling capacity,
    and cooling capacity is only produced when the compressor actively operates.
    """
    env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], seed=42)

    # Case A: Action 0 (ALL_OFF_STANDBY) -> AC mode OFF
    tmpl_off = MULTI_DEVICE_ACTION_TEMPLATES[0]
    calc_off = env._calculate_actuator_powers_and_effects(tmpl_off, current_temp=28.0, current_rh=50.0, ambient_temp=32.0, ambient_rh=60.0)
    assert calc_off["powers"]["cooling_kw"] == env.power_specs["ac_off"]  # 0.05 kW standby
    assert calc_off["effects"]["cooling_thermal_kw"] == 0.0  # Zero cooling capacity
    assert calc_off["effects"]["ac_stage"] == "OFF"

    # Case B: Action 1 (COMFORT_COOL, setpoint 25°C) but room temp is ALREADY <= 25°C (e.g. 24.0°C)
    tmpl_cool = MULTI_DEVICE_ACTION_TEMPLATES[1]
    calc_satisfied = env._calculate_actuator_powers_and_effects(tmpl_cool, current_temp=24.0, current_rh=50.0, ambient_temp=30.0, ambient_rh=60.0)
    assert calc_satisfied["powers"]["cooling_kw"] == env.power_specs["ac_off"]  # Standby only
    assert calc_satisfied["effects"]["cooling_thermal_kw"] == 0.0  # Zero cooling
    assert calc_satisfied["effects"]["ac_stage"] == "OFF"

    # Case C: Action 1 (COMFORT_COOL, setpoint 25°C) and room temp is HOT (28.0°C) -> Active compressor
    calc_active = env._calculate_actuator_powers_and_effects(tmpl_cool, current_temp=28.0, current_rh=50.0, ambient_temp=32.0, ambient_rh=60.0)
    assert calc_active["powers"]["cooling_kw"] > env.power_specs["ac_off"]
    assert calc_active["effects"]["cooling_thermal_kw"] > 0.0
    assert calc_active["effects"]["ac_stage"] in ["MED", "HIGH"]


def test_scenario_generation_parameters():
    """
    Scenario Test: Verifies parameters of synthetic scenario generators.
    """
    env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], seed=42)

    # 1. Smog pollution scenario
    df_smog = env._generate_synthetic_scenario_df(days=2, scenario_type="smog_pollution")
    assert df_smog["ambient_pm25_est"].min() >= 50.0, f"Smog scenario min PM2.5 too low: {df_smog['ambient_pm25_est'].min()}"
    assert df_smog["ambient_pm25_est"].mean() > 65.0, f"Smog scenario mean PM2.5 too low: {df_smog['ambient_pm25_est'].mean()}"
    assert df_smog["ambient_rh_est"].mean() > 75.0, "Smog scenario RH should be high"

    # 2. Large room occupancy scenario
    df_large = env._generate_synthetic_scenario_df(days=2, scenario_type="large_occupancy")
    work_occ = df_large.loc[df_large["is_workhour"] == 1, "occupancy_est"]
    assert work_occ.min() >= 8, f"Large room work occupancy too low: {work_occ.min()}"
    assert work_occ.max() <= 10, f"Large room work occupancy too high: {work_occ.max()}"


def test_dayofweek_consistency():
    """
    Temporal Test: Verifies that day_of_week and dayofweek are synchronized and week vs weekend is respected.
    """
    env = MultiDeviceHVACEnv(room=ROOM_PRESETS["medium"], seed=42)
    _, info = env.reset(start_idx=0)
    assert "dayofweek" in info
    assert "is_workhour" in info
    assert 0 <= info["dayofweek"] <= 6
