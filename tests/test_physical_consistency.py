"""
Unit tests for Physical Consistency and Thermodynamic Validity.
Verifies that the Grey-Box RC model adheres strictly to the laws of thermodynamics.
"""

import pytest
import numpy as np
from src.environment.greybox_model import GreyBoxThermalModel, PhysicalParameters


def test_cooling_monotonicity():
    """
    Thermodynamic Test 1: Cooling Monotonicity
    Applying more HVAC cooling power must result in a lower (or equal) next temperature.
    """
    model = GreyBoxThermalModel(noise_std_temp=0.0, noise_std_rh=0.0)

    t_now = 26.0
    rh_now = 60.0
    p_int = 1.0
    t_amb = 32.0
    rh_amb = 70.0

    # Test powers: 0 kW, 12 kW, 21 kW, 32 kW
    powers = [0.0, 12.0, 21.0, 32.0]
    next_temps = []

    for p in powers:
        t_next, rh_next = model.predict_step(
            current_temp=t_now,
            current_rh=rh_now,
            hvac_kw=p,
            internal_heat_kw=p_int,
            ambient_temp=t_amb,
            ambient_rh=rh_amb,
            hour=14,
            window_open=0
        )
        next_temps.append(t_next)

    # Verify strictly monotonic decreasing with respect to HVAC power
    for i in range(len(next_temps) - 1):
        assert next_temps[i] > next_temps[i + 1], (
            f"Cooling non-monotonic: {powers[i]}kW gave {next_temps[i]}°C, "
            f"but {powers[i+1]}kW gave {next_temps[i+1]}°C"
        )


def test_internal_heat_gain():
    """
    Thermodynamic Test 2: Internal Heat Gain
    Increased plug/lighting electrical power must increase next temperature.
    """
    model = GreyBoxThermalModel(noise_std_temp=0.0, noise_std_rh=0.0)

    t_now = 25.0
    rh_now = 60.0
    t_amb = 28.0
    rh_amb = 70.0
    hvac_p = 0.0

    t_next_low_load, _ = model.predict_step(
        current_temp=t_now, current_rh=rh_now, hvac_kw=hvac_p,
        internal_heat_kw=0.5, ambient_temp=t_amb, ambient_rh=rh_amb,
        hour=10, window_open=0
    )

    t_next_high_load, _ = model.predict_step(
        current_temp=t_now, current_rh=rh_now, hvac_kw=hvac_p,
        internal_heat_kw=4.0, ambient_temp=t_amb, ambient_rh=rh_amb,
        hour=10, window_open=0
    )

    assert t_next_high_load > t_next_low_load, (
        f"Heat gain failed: high load {t_next_high_load}°C <= low load {t_next_low_load}°C"
    )


def test_ambient_envelope_heat_transfer():
    """
    Thermodynamic Test 3: Envelope Conductance
    Passive room (HVAC=0, internal=0) must move towards ambient temperature.
    """
    model = GreyBoxThermalModel(noise_std_temp=0.0, noise_std_rh=0.0)

    # Warmer outdoor -> indoor heats up
    t_warm, _ = model.predict_step(
        current_temp=24.0, current_rh=60.0, hvac_kw=0.0,
        internal_heat_kw=0.0, ambient_temp=34.0, ambient_rh=70.0,
        hour=14, window_open=0
    )
    assert t_warm > 24.0, f"Expected warming towards 34°C, got {t_warm}°C"

    # Cooler outdoor -> indoor cools down
    t_cool, _ = model.predict_step(
        current_temp=26.0, current_rh=60.0, hvac_kw=0.0,
        internal_heat_kw=0.0, ambient_temp=20.0, ambient_rh=70.0,
        hour=2, window_open=0
    )
    assert t_cool < 26.0, f"Expected cooling towards 20°C, got {t_cool}°C"


def test_psychrometric_rh_coupling():
    """
    Thermodynamic Test 4: Psychrometric Coupling
    An increase in temperature at constant moisture mass must reduce relative humidity (h_T < 0).
    """
    model = GreyBoxThermalModel(noise_std_temp=0.0, noise_std_rh=0.0)

    # Case A: HVAC is off, high ambient heat causes temperature rise
    t_next_hot, rh_next_hot = model.predict_step(
        current_temp=23.0, current_rh=70.0, hvac_kw=0.0,
        internal_heat_kw=3.0, ambient_temp=35.0, ambient_rh=70.0,
        hour=14, window_open=0
    )

    # Temperature rose
    assert t_next_hot > 23.0
    # The temperature shift term h_T * dT must be negative
    temp_diff = t_next_hot - 23.0
    expected_rh_component = model.params.h_T * temp_diff
    assert expected_rh_component < 0.0


def test_temperature_safe_bounding():
    """
    Physical limit test: Extreme inputs must be safely clipped within [16.0, 38.0]°C.
    """
    model = GreyBoxThermalModel(noise_std_temp=0.0, noise_std_rh=0.0)

    # Test extreme heating
    t_high, _ = model.predict_step(
        current_temp=37.5, current_rh=50.0, hvac_kw=0.0,
        internal_heat_kw=50.0, ambient_temp=50.0, ambient_rh=50.0,
        hour=12, window_open=0
    )
    assert t_high <= 38.0

    # Test extreme cooling
    t_low, _ = model.predict_step(
        current_temp=17.0, current_rh=50.0, hvac_kw=100.0,
        internal_heat_kw=0.0, ambient_temp=10.0, ambient_rh=50.0,
        hour=4, window_open=0
    )
    assert t_low >= 16.0
