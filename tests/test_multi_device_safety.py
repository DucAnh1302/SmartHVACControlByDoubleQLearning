"""
Unit tests for MultiDeviceSafetyGateway.
Verifies emergency overrides, anti-short-cycling, environmental window vetoes, and sensor fail-safes.
"""

import pytest
from src.safety.multi_device_safety_gateway import MultiDeviceSafetyGateway, MultiDeviceSafetyConfig


def test_emergency_overheat_override():
    """Verify that room temp >= 29.5°C triggers immediate MAX_COOL (Action 2)."""
    gateway = MultiDeviceSafetyGateway()

    # Agent proposes standby (Action 0) when room is 30.5°C
    safe_act, reason = gateway.filter_action(
        agent_action=0,
        indoor_temp=30.5,
        indoor_rh=60.0,
        indoor_co2=700.0,
        indoor_pm25=15.0,
        ambient_temp=33.0,
        ambient_rh=65.0,
        ambient_pm25=20.0,
        hour=14,
        step=1
    )

    assert safe_act == 2  # Forced MAX_COOL
    assert reason is not None
    assert "EMERGENCY_OVERHEAT" in reason
    assert len(gateway.interventions) == 1


def test_emergency_freeze_override():
    """Verify that room temp <= 18.0°C triggers COMFORT_HEAT (Action 6)."""
    gateway = MultiDeviceSafetyGateway()

    # Agent proposes MAX_COOL (Action 2) when room is 17.5°C
    safe_act, reason = gateway.filter_action(
        agent_action=2,
        indoor_temp=17.5,
        indoor_rh=50.0,
        indoor_co2=600.0,
        indoor_pm25=10.0,
        ambient_temp=18.0,
        ambient_rh=50.0,
        ambient_pm25=15.0,
        hour=4,
        step=1
    )

    assert safe_act == 6  # Forced COMFORT_HEAT
    assert reason is not None
    assert "EMERGENCY_FREEZE" in reason


def test_emergency_co2_flush():
    """Verify that CO2 >= 1600 ppm triggers ACTIVE_VENT_PURIFY (Action 5)."""
    gateway = MultiDeviceSafetyGateway()

    # Agent proposes ALL_OFF_STANDBY (Action 0) when room has 1750 ppm CO2
    safe_act, reason = gateway.filter_action(
        agent_action=0,
        indoor_temp=25.0,
        indoor_rh=55.0,
        indoor_co2=1750.0,
        indoor_pm25=15.0,
        ambient_temp=30.0,
        ambient_rh=60.0,
        ambient_pm25=20.0,
        hour=11,
        step=1
    )

    assert safe_act == 5  # Forced ACTIVE_VENT_PURIFY
    assert "EMERGENCY_CO2_FLUSH" in reason


def test_window_environmental_vetoes():
    """Verify that window opening is vetoed during night, heatwave, rain, or high PM2.5."""
    gateway = MultiDeviceSafetyGateway()

    # 1. Night veto (23:00)
    act1, reason1 = gateway.filter_action(
        agent_action=4, indoor_temp=25.0, indoor_rh=55.0, indoor_co2=700.0, indoor_pm25=15.0,
        ambient_temp=28.0, ambient_rh=65.0, ambient_pm25=20.0, hour=23, step=1
    )
    assert act1 != 4
    assert "Night security" in reason1

    # 2. Outdoor heatwave veto (35°C)
    act2, reason2 = gateway.filter_action(
        agent_action=4, indoor_temp=25.0, indoor_rh=55.0, indoor_co2=700.0, indoor_pm25=15.0,
        ambient_temp=35.0, ambient_rh=65.0, ambient_pm25=20.0, hour=12, step=2
    )
    assert act2 != 4
    assert "Outdoor temperature" in reason2

    # 3. Outdoor storm rain veto (90% RH)
    act3, reason3 = gateway.filter_action(
        agent_action=4, indoor_temp=25.0, indoor_rh=55.0, indoor_co2=700.0, indoor_pm25=15.0,
        ambient_temp=27.0, ambient_rh=90.0, ambient_pm25=20.0, hour=10, step=3
    )
    assert act3 != 4
    assert "rain hazard" in reason3.lower()

    # 4. Outdoor smog veto (80 µg/m³ PM2.5)
    act4, reason4 = gateway.filter_action(
        agent_action=4, indoor_temp=25.0, indoor_rh=55.0, indoor_co2=700.0, indoor_pm25=15.0,
        ambient_temp=28.0, ambient_rh=65.0, ambient_pm25=80.0, hour=14, step=4
    )
    assert act4 != 4
    assert "smog hazard" in reason4.lower()


def test_anti_short_cycle_compressor_lock():
    """Verify that compressor cannot toggle on/off within 2 consecutive steps."""
    gateway = MultiDeviceSafetyGateway(MultiDeviceSafetyConfig(min_compressor_dwell_steps=2))

    # Step 1: Start in COMFORT_COOL (Action 1, AC active)
    act1, _ = gateway.filter_action(1, 25.0, 55.0, 700.0, 15.0, 30.0, 60.0, 20.0, 10, 1)
    assert act1 == 1

    # Step 2: Next step, agent wants ALL_OFF_STANDBY (Action 0, AC off) -> blocked!
    act2, reason2 = gateway.filter_action(0, 25.0, 55.0, 700.0, 15.0, 30.0, 60.0, 20.0, 11, 2)
    assert act2 == 1  # Retained
    assert "SHORT_CYCLE_PROTECTION" in reason2

    # Step 3: Now 2 steps have elapsed in state 1; transitioning to Action 0 is permitted
    act3, reason3 = gateway.filter_action(0, 25.0, 55.0, 700.0, 15.0, 30.0, 60.0, 20.0, 12, 3)
    assert act3 == 0
    assert reason3 is None


def test_sensor_anomaly_failsafe():
    """Verify NaN sensor values trigger safe fallback."""
    gateway = MultiDeviceSafetyGateway()
    act, reason = gateway.filter_action(
        2, float("nan"), 55.0, 700.0, 15.0, 30.0, 60.0, 20.0, 12, 1
    )
    assert act == 1
    assert "SENSOR_ANOMALY_FAILSAFE" in reason
