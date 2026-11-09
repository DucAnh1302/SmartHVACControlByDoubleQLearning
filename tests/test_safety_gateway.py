"""
Unit tests for Safety Gateway and BMS Hardware-in-the-Loop Interlock.
Verifies hard safety overrides, short-cycle compressor protection, and weather vetoes.
"""

import pytest
from src.safety.safety_gateway import SafetyGateway, DeploymentReadinessLevel, ShadowModeRunner
from src.safety.hil_interface import MockBMSControllerHIL, BMSTelemetryPacket, BMSCommandPacket


def test_overheat_emergency_override():
    """Verify that room temp >= 29.5°C triggers immediate maximum cooling (SET_23)."""
    gateway = SafetyGateway(emergency_overheat_temp=29.5)

    # Agent wants to turn AC off (Action 3) when room is 30.5°C
    safe_act, reason = gateway.filter_action(
        agent_action=3,
        indoor_temp=30.5,
        ambient_temp=33.0,
        ambient_rh=65.0,
        hour=14,
        step=1
    )

    assert safe_act == 1  # Overridden to SET_23
    assert reason is not None
    assert "EMERGENCY_OVERHEAT" in reason
    assert len(gateway.interventions) == 1
    assert gateway.interventions[0].rule_violated == "EMERGENCY_OVERHEAT_OVERRIDE"


def test_freeze_emergency_override():
    """Verify that room temp <= 18.0°C triggers immediate AC shutdown (AC_OFF)."""
    gateway = SafetyGateway(emergency_freeze_temp=18.0)

    # Agent wants to cool to 23°C (Action 1) when room is 17.5°C
    safe_act, reason = gateway.filter_action(
        agent_action=1,
        indoor_temp=17.5,
        ambient_temp=22.0,
        ambient_rh=60.0,
        hour=5,
        step=1
    )

    assert safe_act == 3  # Overridden to AC_OFF
    assert reason is not None
    assert "EMERGENCY_FREEZE" in reason
    assert len(gateway.interventions) == 1


def test_window_night_security_veto():
    """Verify that window opening is strictly prohibited during night hours (22:00 - 06:00)."""
    gateway = SafetyGateway()

    # Agent tries to open window (Action 4) at 23:00
    safe_act, reason = gateway.filter_action(
        agent_action=4,
        indoor_temp=25.0,
        ambient_temp=26.0,
        ambient_rh=65.0,
        hour=23,
        step=1
    )

    assert safe_act == 3  # Closed & AC_OFF
    assert "Night security" in reason


def test_window_weather_hazard_veto():
    """Verify that window opening is vetoed when ambient is scorching (>33°C) or storming (>85% RH)."""
    gateway = SafetyGateway(window_max_ambient_temp=33.0, window_max_ambient_rh=85.0)

    # Case A: Scorching heat (35°C) at 13:00
    safe_act_heat, reason_heat = gateway.filter_action(
        agent_action=4,
        indoor_temp=25.5,
        ambient_temp=35.0,
        ambient_rh=60.0,
        hour=13,
        step=1
    )
    assert safe_act_heat == 0  # Fallback to SET_25 closed
    assert "hazardous" in reason_heat.lower()

    # Case B: Tropical heavy rain (92% RH) at 10:00
    safe_act_rain, reason_rain = gateway.filter_action(
        agent_action=4,
        indoor_temp=25.5,
        ambient_temp=27.0,
        ambient_rh=92.0,
        hour=10,
        step=2
    )
    assert safe_act_rain == 0
    assert "hazardous" in reason_rain.lower()


def test_compressor_short_cycle_protection():
    """Verify that AC is not toggled between OFF and Cooling within 2 consecutive hours."""
    gateway = SafetyGateway(min_cycles_between_toggles=2)

    # Step 1: Start with SET_25 (Action 0)
    act1, _ = gateway.filter_action(agent_action=0, indoor_temp=25.0, ambient_temp=30.0, ambient_rh=60.0, hour=10, step=1)
    assert act1 == 0

    # Step 2: Next hour, agent immediately wants to shut OFF (Action 3) -> should be blocked!
    act2, reason2 = gateway.filter_action(agent_action=3, indoor_temp=25.0, ambient_temp=30.0, ambient_rh=60.0, hour=11, step=2)
    assert act2 == 0  # Retains action 0 to protect compressor
    assert "SHORT_CYCLE_PROTECTION" in reason2

    # Step 3: Now 2 hours in state 0 have elapsed; shutting OFF should now be permitted
    act3, reason3 = gateway.filter_action(agent_action=3, indoor_temp=25.0, ambient_temp=30.0, ambient_rh=60.0, hour=12, step=3)
    assert act3 == 3
    assert reason3 is None


def test_mock_bms_hil_controller():
    """Verify Mock BMS Hardware-In-The-Loop controller dispatch and validation."""
    hil = MockBMSControllerHIL(device_id="BMS_AHU_FL02_Z01")
    assert hil.device_id == "BMS_AHU_FL02_Z01"
    assert hil.is_connected is False

    # Connect
    assert hil.connect() is True
    assert hil.is_connected is True

    # Test commanding safe packet
    cmd = BMSCommandPacket(
        device_id="BMS_AHU_FL02_Z01",
        target_setpoint_c=25.0,
        compressor_enable=True,
        window_actuator_enable=False,
        safety_interlock_verified=True
    )
    resp = hil.send_control_packet(cmd)
    assert resp["status"] == "ACK"
    assert resp["setpoint_applied"] == 25.0
    assert resp["compressor_enabled"] is True

    # Test commanding unsafe packet without safety interlock verification
    unsafe_cmd = BMSCommandPacket(
        device_id="BMS_AHU_FL02_Z01",
        target_setpoint_c=16.0,
        compressor_enable=True,
        window_actuator_enable=False,
        safety_interlock_verified=False
    )
    with pytest.raises(PermissionError):
        hil.send_control_packet(unsafe_cmd)
