"""
Unit tests verifying Action 5 (ACTIVE_VENT_PURIFY) conditioned cooling-ventilation
behavior under hot tropical weather (Protocol v2).
"""

import numpy as np
import pandas as pd

from src.environment.multi_device_env import (
    MULTI_DEVICE_ACTION_TEMPLATES,
    MultiDeviceHVACEnv,
)
from src.safety.multi_device_safety_gateway import MultiDeviceSafetyGateway


def _tropical_hot_high_co2_df(n_hours: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "hour": [h % 24 for h in range(n_hours)],
        "dayofweek": [0] * n_hours,
        "is_workhour": [1] * n_hours,
        "ambient_temp_est": [34.0] * n_hours,
        "ambient_rh_est": [68.0] * n_hours,
        "ambient_co2_est": [415.0] * n_hours,
        "ambient_pm25_est": [28.0] * n_hours,
        "internal_heat_kw": [2.2] * n_hours,
        "occupancy_est": [4] * n_hours,
        "temp_c": [26.2] * n_hours,
        "humidity_rh": [60.0] * n_hours,
    })


def test_action5_template_has_conditioned_cooling():
    tpl5 = MULTI_DEVICE_ACTION_TEMPLATES[5]
    assert tpl5.name == "ACTIVE_VENT_PURIFY"
    assert tpl5.ac_mode == "SET_25"
    assert tpl5.ac_setpoint_c == 25.0
    assert tpl5.vent_flow_m3_h == 500.0
    assert tpl5.filter_cadr_m3_h == 450.0


def test_action5_purges_co2_without_tropical_overheating():
    df = _tropical_hot_high_co2_df()
    env = MultiDeviceHVACEnv(data_df=df, episode_length=6, seed=7)
    env.reset(
        start_idx=0,
        initial_temp=26.2,
        initial_rh=60.0,
        initial_co2=1500.0,
        initial_pm25=30.0,
        occupancy=4,
        seed=7,
    )
    gw = MultiDeviceSafetyGateway()
    gw.reset()

    for step in range(4):
        safe_act, _ = gw.filter_action(
            agent_action=5,
            indoor_temp=env.current_temp,
            indoor_rh=env.current_rh,
            indoor_co2=env.current_co2,
            indoor_pm25=env.current_pm25,
            ambient_temp=34.0,
            ambient_rh=68.0,
            ambient_pm25=28.0,
            hour=13,
            step=step,
        )
        _, _, _, _, info = env.step(safe_act)

    assert info["co2_ppm"] < 950.0, f"Expected CO2 < 950 ppm after flush, got {info['co2_ppm']:.1f}"
    assert info["temp_c"] < 27.0, f"Action 5 must prevent tropical overheating, got {info['temp_c']:.2f}°C"
