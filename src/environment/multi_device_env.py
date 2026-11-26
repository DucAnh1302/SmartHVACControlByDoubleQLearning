"""
Multi-Device HVAC Digital Twin Simulation Environment.
Simulates coupled thermal, psychrometric moisture, CO2, particulate matter (PM2.5),
and multi-device electrical energy dynamics across 7 distinct actuator groups:
1. Cooling (Air Conditioner with compressor staging)
2. Heating (Electric/Hydronic Heater)
3. Mechanical Ventilation (Supply & Exhaust fans)
4. Window / Damper (Natural ventilation)
5. Humidification (Ultrasonic humidifier)
6. Dehumidification (Dedicated refrigerant dehumidifier)
7. Air Filtration (HEPA Air Purifier)

Provides:
- 720 Discrete States (5 Temp x 3 RH x 3 CO2 x 2 PM2.5 x 2 Heat Load x 4 Time/Tariff)
- 12 Coordinated Action Templates
- Explicit room geometry (Small 20m², Medium 50m², Large 120m²)
- Psychrometric water vapor mass balance (Magnus formula)
- Transparent Data Provenance tracking
"""

from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd

from src.data.provenance import ProvenanceType


# ---------------------------------------------------------------------------
# Psychrometric Utilities (Magnus Formula & Water Vapor Mass Balance)
# ---------------------------------------------------------------------------
def _clip(val: float, lo: float, hi: float) -> float:
    return lo if val < lo else (hi if val > hi else float(val))


def magnus_saturation_vapor_pressure(temp_c: float) -> float:
    """
    Calculate saturation water vapor pressure P_sat (kPa) at temp_c (°C)
    using the Magnus-Tetens approximation (valid -40°C to 50°C).
    """
    t = _clip(float(temp_c), -20.0, 60.0)
    # P_sat in kPa
    return float(0.61078 * math.exp((17.27 * t) / (t + 237.3)))


def humidity_ratio_from_rh(temp_c: float, rh_pct: float, p_atm_kpa: float = 101.325) -> float:
    """
    Calculate humidity ratio W (kg water vapor / kg dry air)
    from dry-bulb temperature (°C) and relative humidity (%).
    """
    p_sat = magnus_saturation_vapor_pressure(temp_c)
    p_v = (_clip(float(rh_pct), 1.0, 100.0) / 100.0) * p_sat
    # Avoid singularity if p_v approaches p_atm
    p_v = min(p_v, p_atm_kpa * 0.95)
    w = 0.62198 * (p_v / (p_atm_kpa - p_v))
    return float(max(1e-5, w))


def rh_from_humidity_ratio(temp_c: float, w: float, p_atm_kpa: float = 101.325) -> float:
    """
    Calculate relative humidity (%) from temperature (°C) and humidity ratio W (kg/kg).
    """
    w_safe = max(1e-5, float(w))
    # Partial vapor pressure P_v
    p_v = (w_safe * p_atm_kpa) / (0.62198 + w_safe)
    p_sat = magnus_saturation_vapor_pressure(temp_c)
    rh = (p_v / p_sat) * 100.0
    return _clip(rh, 5.0, 99.0)


# ---------------------------------------------------------------------------
# Room Geometry & Thermal Properties
# ---------------------------------------------------------------------------
@dataclass
class RoomGeometry:
    """Physical building/room dimensions and thermal envelope specifications."""
    name: str = "Medium Office"
    floor_area_m2: float = 50.0            # Floor area in m²
    ceiling_height_m: float = 2.8          # Ceiling height in m
    u_envelope: float = 0.55               # Envelope U-value in W/(m²·K)
    window_area_m2: float = 6.0            # Total window openable area in m²
    base_infiltration_ach: float = 0.20    # Baseline natural infiltration rate (air changes per hour)
    specific_heat_capacity_kj_m2_k: float = 220.0  # Building thermal capacitance per floor area (kJ/(m²·K))

    @property
    def volume_m3(self) -> float:
        return self.floor_area_m2 * self.ceiling_height_m

    @property
    def envelope_area_m2(self) -> float:
        # Approximate square room envelope: 4 walls + roof
        perimeter = 4.0 * math.sqrt(self.floor_area_m2)
        wall_area = perimeter * self.ceiling_height_m
        return wall_area + self.floor_area_m2

    @property
    def air_mass_kg(self) -> float:
        rho_air = 1.204  # kg/m³ at 20°C
        return self.volume_m3 * rho_air

    @property
    def total_heat_capacity_kj_k(self) -> float:
        # Realistic building thermal capacitance: walls, concrete slabs, furniture (~220 kJ/(m²·K))
        return self.floor_area_m2 * self.specific_heat_capacity_kj_m2_k

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "floor_area_m2": self.floor_area_m2,
            "ceiling_height_m": self.ceiling_height_m,
            "volume_m3": round(self.volume_m3, 2),
            "envelope_area_m2": round(self.envelope_area_m2, 2),
            "u_envelope": self.u_envelope,
            "air_mass_kg": round(self.air_mass_kg, 2),
            "heat_capacity_kj_k": round(self.total_heat_capacity_kj_k, 2),
            "window_area_m2": self.window_area_m2,
            "base_infiltration_ach": self.base_infiltration_ach
        }


# Standard Presets for Room Size Dependency Evaluation
ROOM_PRESETS: Dict[str, RoomGeometry] = {
    "small": RoomGeometry(
        name="Small Office / Study",
        floor_area_m2=20.0,
        ceiling_height_m=2.6,
        u_envelope=0.65,
        window_area_m2=3.0,
        base_infiltration_ach=0.25,
        specific_heat_capacity_kj_m2_k=200.0
    ),
    "medium": RoomGeometry(
        name="Medium Open Office",
        floor_area_m2=50.0,
        ceiling_height_m=2.8,
        u_envelope=0.55,
        window_area_m2=6.0,
        base_infiltration_ach=0.20,
        specific_heat_capacity_kj_m2_k=220.0
    ),
    "large": RoomGeometry(
        name="Large Conference / Open Floor",
        floor_area_m2=120.0,
        ceiling_height_m=3.2,
        u_envelope=0.45,
        window_area_m2=15.0,
        base_infiltration_ach=0.18,
        specific_heat_capacity_kj_m2_k=250.0
    )
}


# ---------------------------------------------------------------------------
# Coordinated Action Templates (12 Action Space)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionTemplate:
    action_id: int
    name: str
    description: str
    # 1. AC Cooling
    ac_mode: str               # "OFF", "SET_25", "SET_23", "SET_27", "SET_26.5"
    ac_setpoint_c: Optional[float]
    # 2. Heating
    heater_mode: str           # "OFF", "LOW", "HIGH"
    heater_setpoint_c: Optional[float]
    # 3. Ventilation fans
    vent_fan_speed: str        # "OFF", "LOW", "MED", "HIGH"
    vent_flow_m3_h: float
    # 4. Window
    window_open: int           # 0: Closed, 1: Open
    # 5. Humidification
    humidifier_mode: str       # "OFF", "LOW", "MED", "HIGH"
    humidifier_rate_kg_h: float
    # 6. Dehumidification
    dehumidifier_mode: str     # "OFF", "LOW", "HIGH"
    dehumidifier_rate_kg_h: float
    # 7. Air Filter
    filter_speed: str          # "OFF", "LOW", "MED", "HIGH"
    filter_cadr_m3_h: float


MULTI_DEVICE_ACTION_TEMPLATES: Dict[int, ActionTemplate] = {
    0: ActionTemplate(
        action_id=0,
        name="ALL_OFF_STANDBY",
        description="All HVAC equipment OFF / standby. Minimum energy idle state.",
        ac_mode="OFF", ac_setpoint_c=None,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="OFF", vent_flow_m3_h=0.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="OFF", filter_cadr_m3_h=0.0
    ),
    1: ActionTemplate(
        action_id=1,
        name="COMFORT_COOL",
        description="Standard cooling target 25°C + mild fresh air ventilation + low filtration.",
        ac_mode="SET_25", ac_setpoint_c=25.0,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="LOW", vent_flow_m3_h=150.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="LOW", filter_cadr_m3_h=120.0
    ),
    2: ActionTemplate(
        action_id=2,
        name="MAX_COOL",
        description="Intensive cooling target 23°C + boosted fresh air + medium air filtration.",
        ac_mode="SET_23", ac_setpoint_c=23.0,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="MED", vent_flow_m3_h=300.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="MED", filter_cadr_m3_h=250.0
    ),
    3: ActionTemplate(
        action_id=3,
        name="ECONOMY_COOL",
        description="Energy-efficient cooling target 27°C + baseline ventilation + low filtration.",
        ac_mode="SET_27", ac_setpoint_c=27.0,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="LOW", vent_flow_m3_h=150.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="LOW", filter_cadr_m3_h=100.0
    ),
    4: ActionTemplate(
        action_id=4,
        name="NATURAL_VENTILATION",
        description="Free cooling / fresh air via open window; active mechanical cooling/heating OFF.",
        ac_mode="OFF", ac_setpoint_c=None,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="OFF", vent_flow_m3_h=0.0,
        window_open=1,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="LOW", filter_cadr_m3_h=80.0
    ),
    5: ActionTemplate(
        action_id=5,
        name="ACTIVE_VENT_PURIFY",
        description="High fresh air exchange (500 m³/h) + max HEPA filtration + conditioned cooling setpoint 25°C for tropical IAQ purge.",
        ac_mode="SET_25", ac_setpoint_c=25.0,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="HIGH", vent_flow_m3_h=500.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="HIGH", filter_cadr_m3_h=450.0
    ),
    6: ActionTemplate(
        action_id=6,
        name="COMFORT_HEAT",
        description="Comfort heating target 22°C + mild ventilation + gentle humidification.",
        ac_mode="OFF", ac_setpoint_c=None,
        heater_mode="LOW", heater_setpoint_c=22.0,
        vent_fan_speed="LOW", vent_flow_m3_h=150.0,
        window_open=0,
        humidifier_mode="LOW", humidifier_rate_kg_h=0.25,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="LOW", filter_cadr_m3_h=100.0
    ),
    7: ActionTemplate(
        action_id=7,
        name="MAX_HEAT",
        description="High heating target 24°C + medium humidification for cold spells.",
        ac_mode="OFF", ac_setpoint_c=None,
        heater_mode="HIGH", heater_setpoint_c=24.0,
        vent_fan_speed="LOW", vent_flow_m3_h=150.0,
        window_open=0,
        humidifier_mode="MED", humidifier_rate_kg_h=0.45,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="LOW", filter_cadr_m3_h=100.0
    ),
    8: ActionTemplate(
        action_id=8,
        name="DEHUMIDIFY_ACTIVE",
        description="Targeted humidity reduction: dedicated dehumidifier + moderate cooling (26.0°C).",
        ac_mode="SET_26", ac_setpoint_c=26.0,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="LOW", vent_flow_m3_h=150.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="HIGH", dehumidifier_rate_kg_h=1.20,
        filter_speed="LOW", filter_cadr_m3_h=100.0
    ),
    9: ActionTemplate(
        action_id=9,
        name="HUMIDIFY_COMFORT",
        description="Targeted dry room humidification: high humidifier + comfortable cooling 25°C.",
        ac_mode="SET_25", ac_setpoint_c=25.0,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="LOW", vent_flow_m3_h=150.0,
        window_open=0,
        humidifier_mode="HIGH", humidifier_rate_kg_h=0.60,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="LOW", filter_cadr_m3_h=100.0
    ),
    10: ActionTemplate(
        action_id=10,
        name="ECO_AIR_PURIFY",
        description="Air purification priority: max HEPA filtration without active thermal conditioning.",
        ac_mode="OFF", ac_setpoint_c=None,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="LOW", vent_flow_m3_h=100.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="HIGH", filter_cadr_m3_h=450.0
    ),
    11: ActionTemplate(
        action_id=11,
        name="NIGHT_SETBACK",
        description="Overnight setback cooling (26.5°C) + low quiet ventilation + window closed.",
        ac_mode="SET_26.5", ac_setpoint_c=26.5,
        heater_mode="OFF", heater_setpoint_c=None,
        vent_fan_speed="LOW", vent_flow_m3_h=100.0,
        window_open=0,
        humidifier_mode="OFF", humidifier_rate_kg_h=0.0,
        dehumidifier_mode="OFF", dehumidifier_rate_kg_h=0.0,
        filter_speed="LOW", filter_cadr_m3_h=80.0
    )
}


# ---------------------------------------------------------------------------
# State Space Discretization (720 Discrete States)
# ---------------------------------------------------------------------------
def discretize_state_720(
    temp_c: float,
    humidity_rh: float,
    co2_ppm: float,
    pm25_ug_m3: float,
    internal_heat_kw: float,
    hour: int,
    dayofweek: int
) -> Tuple[int, int, int, int, int, int]:
    """
    Discretize continuous environmental state into a 6D index:
    1. Temp Bin (5 bins): <21, [21, 24), [24, 26), [26, 28.5), >=28.5
    2. Humidity Bin (3 bins): <40, [40, 65), >=65
    3. CO2 Bin (3 bins): <800, [800, 1200), >=1200
    4. PM2.5 Bin (2 bins): <=35 (Good), >35 (Polluted)
    5. Heat Load Bin (2 bins): <0.5 kW (Low / Unoccupied), >=0.5 kW (High / Occupied)
    6. Time/Tariff Bin (4 bins): Night/Weekend, Morning, Midday Peak, Evening
    Returns: (t_bin, rh_bin, co2_bin, pm_bin, load_bin, time_bin)
    """
    # 1. Temperature (5 bins: cutoffs [21.0, 24.0, 26.0, 28.5])
    if temp_c < 21.0:
        t_bin = 0
    elif temp_c < 24.0:
        t_bin = 1
    elif temp_c < 26.0:
        t_bin = 2
    elif temp_c < 28.5:
        t_bin = 3
    else:
        t_bin = 4

    # 2. Humidity (3 bins: cutoffs [40.0, 65.0])
    if humidity_rh < 40.0:
        rh_bin = 0
    elif humidity_rh < 65.0:
        rh_bin = 1
    else:
        rh_bin = 2

    # 3. CO2 (3 bins: cutoffs [800.0, 1200.0])
    if co2_ppm < 800.0:
        co2_bin = 0
    elif co2_ppm < 1200.0:
        co2_bin = 1
    else:
        co2_bin = 2

    # 4. PM2.5 (2 bins)
    pm_bin = 0 if pm25_ug_m3 <= 35.0 else 1

    # 5. Internal Heat Load (2 bins: tải nhiệt nội sinh quy mô, ngưỡng 0.5 kW, tương đương chỉ báo có người, trùng 96.8%)
    load_bin = 0 if internal_heat_kw < 0.5 else 1

    # 6. Time / Tariff (4 bins)
    is_weekend = (dayofweek >= 5)
    if is_weekend or (0 <= hour < 6 or hour >= 22):
        time_bin = 0  # Off-Peak / Night
    elif 6 <= hour < 9:
        time_bin = 1  # Morning Ramp
    elif 9 <= hour < 17:
        time_bin = 2  # Midday Peak Tariff
    else:
        time_bin = 3  # Evening Shoulder

    return (t_bin, rh_bin, co2_bin, pm_bin, load_bin, time_bin)


def state_tuple_to_index_720(state_tuple: Tuple[int, int, int, int, int, int]) -> int:
    """Encode 6D discrete state tuple into a flat integer in [0, 719]."""
    t_bin, rh_bin, co2_bin, pm_bin, load_bin, time_bin = state_tuple
    t_bin = min(max(t_bin, 0), 4)
    rh_bin = min(max(rh_bin, 0), 2)
    co2_bin = min(max(co2_bin, 0), 2)
    pm_bin = min(max(pm_bin, 0), 1)
    load_bin = min(max(load_bin, 0), 1)
    time_bin = min(max(time_bin, 0), 3)

    idx = ((((t_bin * 3 + rh_bin) * 3 + co2_bin) * 2 + pm_bin) * 2 + load_bin) * 4 + time_bin
    return int(idx)


def index_to_state_tuple_720(state_idx: int) -> Tuple[int, int, int, int, int, int]:
    """Decode a flat integer in [0, 719] into 6D discrete state tuple."""
    idx = 0 if state_idx < 0 else (719 if state_idx > 719 else int(state_idx))
    time_bin = idx % 4
    rem = idx // 4
    load_bin = rem % 2
    rem = rem // 2
    pm_bin = rem % 2
    rem = rem // 2
    co2_bin = rem % 3
    rem = rem // 3
    rh_bin = rem % 3
    t_bin = rem // 3
    return (t_bin, rh_bin, co2_bin, pm_bin, load_bin, time_bin)


# ---------------------------------------------------------------------------
# Multi-Device HVAC Digital Twin Environment
# ---------------------------------------------------------------------------
class MultiDeviceHVACEnv:
    """
    Complete HVAC Digital Twin research simulation environment.
    Models multi-device coupled thermodynamics, psychrometric moisture balance,
    CO2 ventilation dynamics, PM2.5 filtration, and multi-actuator electrical power.
    """

    def __init__(
        self,
        data_df: Optional[pd.DataFrame] = None,
        room: Optional[RoomGeometry] = None,
        episode_length: int = 24,
        step_duration_hours: float = 1.0,
        scenario_type: str = "standard",
        seed: int = 42
    ):
        self.episode_length = episode_length
        self.step_duration_hours = step_duration_hours
        self.scenario_type = scenario_type
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.room = room or ROOM_PRESETS["medium"]

        # If telemetry data is provided, use it for ambient and base loads; otherwise generate synthetic
        if data_df is not None and len(data_df) > 0:
            self.data_df = data_df.reset_index(drop=True)
            self.is_synthetic_scenario = False
        else:
            self.data_df = self._generate_synthetic_scenario_df(days=7, scenario_type=scenario_type)
            self.is_synthetic_scenario = True

        # Space sizes
        self.n_states = 720
        self.n_actions = len(MULTI_DEVICE_ACTION_TEMPLATES)  # 12

        # Physical Device Nominal Power Ratings (kW electrical)
        # Scaled realistically to room volume (50m² standard)
        # AC sizing updated in Phase 1b: Capacity / Q_design = 1.30 (within [1.2, 1.5])
        scale = max(0.5, self.room.floor_area_m2 / 50.0)
        self.power_specs = {
            "ac_off": 0.05,
            "ac_low": 0.52 * scale,
            "ac_med": 1.01 * scale,
            "ac_high": 1.59 * scale,
            "ac_cop": 3.2,                       # Coefficient of Performance (Thermal kW / Electrical kW)
            "heater_off": 0.0,
            "heater_low": 1.50 * scale,
            "heater_high": 3.00 * scale,
            "vent_fan_low": 0.04 * scale,
            "vent_fan_med": 0.09 * scale,
            "vent_fan_high": 0.18 * scale,
            "exhaust_fan_power": 0.05 * scale,
            "humidifier_low": 0.08 * scale,
            "humidifier_high": 0.20 * scale,
            "dehumidifier_power": 0.45 * scale,
            "filter_low": 0.02 * scale,
            "filter_med": 0.05 * scale,
            "filter_high": 0.12 * scale,
        }

        # Runtime state variables
        self.current_step = 0
        self.data_idx = 0
        self.current_temp = 25.5
        self.current_rh = 55.0
        self.current_co2 = 650.0
        self.current_pm25 = 18.0
        self.occupancy_count = 3
        self.window_state = 0
        self.last_action: Optional[int] = None
        self.current_state_idx = 0

        # Physical internal heat load scaling factor for target room geometry
        # Baseline CU-BEMS zone load has p95 ~ 12.795 kW across whole floor section.
        # Scaled to nominal 35 W/m² (0.035 kW/m²) for this room geometry (DERIVED).
        self.internal_heat_scale = (0.035 * self.room.floor_area_m2) / 12.7947

        # Energy tracking (cumulative kWh per device)
        self.cumul_energy = {
            "cooling_kwh": 0.0,
            "heating_kwh": 0.0,
            "ventilation_kwh": 0.0,
            "humidifier_kwh": 0.0,
            "dehumidifier_kwh": 0.0,
            "filtration_kwh": 0.0,
            "total_kwh": 0.0
        }
        self.instant_power = {k.replace("_kwh", "_kw"): 0.0 for k in self.cumul_energy.keys()}
        self.episode_switching_count = 0

        # Automated Sizing Check (Phase 1b Gate: Capacity/Q_design must be in [1.2, 1.5])
        assert 1.2 <= self.sizing_safety_ratio <= 1.5, (
            f"Capacity/Q_design ratio {self.sizing_safety_ratio:.4f} must be in [1.2, 1.5]"
        )

        # Pre-compute constant coil/physics scalars and DataFrame fast row cache
        self._cached_design_load_kw = self.design_cooling_load_kw
        self._cached_w_coil_sat = humidity_ratio_from_rh(12.0, 100.0)
        self._cached_v_coil_m3_h = 0.08 * self._cached_design_load_kw * 3600.0
        self._cached_df_id = None
        self._cached_df_len = -1
        self._ensure_row_cache()

    def _ensure_row_cache(self) -> None:
        """Build O(1) Python list cache for self.data_df columns (zero pandas.iloc overhead in step)."""
        df = self.data_df
        if self._cached_df_id == id(df) and self._cached_df_len == len(df):
            return
        n = len(df)
        cols = set(df.columns)
        self._c_hour = [int(x) for x in df["hour"].tolist()] if "hour" in cols else [12] * n
        if "dayofweek" in cols:
            self._c_dow = [int(x) for x in df["dayofweek"].tolist()]
        elif "day_of_week" in cols:
            self._c_dow = [int(x) for x in df["day_of_week"].tolist()]
        else:
            self._c_dow = [0] * n
        if "is_workhour" in cols:
            self._c_work = [int(x) for x in df["is_workhour"].tolist()]
        else:
            self._c_work = [1 if (d < 5 and 8 <= h < 18) else 0 for d, h in zip(self._c_dow, self._c_hour)]
        self._c_temp = [float(x) for x in df["temp_c"].tolist()] if "temp_c" in cols else [25.5] * n
        self._c_rh = [float(x) for x in df["humidity_rh"].tolist()] if "humidity_rh" in cols else [55.0] * n
        self._c_amb_t = [float(x) for x in df["ambient_temp_est"].tolist()] if "ambient_temp_est" in cols else [32.0] * n
        self._c_amb_rh = [float(x) for x in df["ambient_rh_est"].tolist()] if "ambient_rh_est" in cols else [65.0] * n
        self._c_amb_co2 = [float(x) for x in df["ambient_co2_est"].tolist()] if "ambient_co2_est" in cols else [415.0] * n
        self._c_amb_pm25 = [float(x) for x in df["ambient_pm25_est"].tolist()] if "ambient_pm25_est" in cols else [25.0] * n
        if "internal_heat_scaled_kw" in cols:
            self._c_p_int = [float(x) for x in df["internal_heat_scaled_kw"].tolist()]
            self._c_raw_p_int = [float(x) for x in df["internal_heat_kw"].tolist()] if "internal_heat_kw" in cols else list(self._c_p_int)
        elif "internal_heat_kw" in cols:
            raw_list = [float(x) for x in df["internal_heat_kw"].tolist()]
            self._c_raw_p_int = raw_list
            s = self.internal_heat_scale
            self._c_p_int = [x * s for x in raw_list]
        else:
            self._c_p_int = [2.0 * self.internal_heat_scale] * n
            self._c_raw_p_int = [2.0] * n
        if "occupancy_est" in cols:
            self._c_occ = [int(x) for x in df["occupancy_est"].tolist()]
        else:
            self._c_occ = [3 if w == 1 else 0 for w in self._c_work]
        self._cached_df_id = id(df)
        self._cached_df_len = n

    @property
    def design_cooling_load_kw(self) -> float:
        """Calculate peak design cooling load (Q_design) under standard ASHRAE design conditions."""
        delta_t = 11.0  # 35°C outdoor - 24°C indoor
        q_envelope = (self.room.u_envelope * self.room.envelope_area_m2 * delta_t) / 1000.0
        q_vent = (1.21 * (178.0 * (self.room.floor_area_m2 / 50.0)) * delta_t) / 3600.0
        q_internal = 0.035 * self.room.floor_area_m2
        n_occ_max = round(self.room.floor_area_m2 / 6.0)
        q_occupants = n_occ_max * 0.075
        return float(q_envelope + q_vent + q_internal + q_occupants)

    @property
    def max_cooling_capacity_kw(self) -> float:
        """Active maximum thermal cooling capacity at stage HIGH."""
        active_kw = max(0.0, self.power_specs["ac_high"] - self.power_specs["ac_off"])
        return float(active_kw * self.power_specs["ac_cop"])

    @property
    def sizing_safety_ratio(self) -> float:
        """Ratio of maximum cooling capacity to design cooling load (Capacity / Q_design)."""
        return float(self.max_cooling_capacity_kw / self.design_cooling_load_kw)

    def _generate_synthetic_scenario_df(
        self,
        days: int = 7,
        scenario_type: str = "standard"
    ) -> pd.DataFrame:
        """
        Generate high-resolution synthetic diurnal weather timeseries for standalone mode.
        Supported scenarios:
        - 'standard': Standard tropical Bangkok-like diurnal cycle.
        - 'heatwave': Severe diurnal heatwave with high ambient temperatures (30-38°C).
        - 'large_occupancy': Large room schedule with high occupant density (8-10 occupants during work hours).
        - 'smog_pollution': Severe outdoor PM2.5 pollution (>60 µg/m³) and high humidity (>75% RH).
        """
        n_hours = days * 24
        hours = np.arange(n_hours)
        dows = (hours // 24) % 7
        hours_of_day = hours % 24

        # Work hours: weekdays (dows < 5) from 08:00 to 18:00
        is_workhour = np.where((dows < 5) & (hours_of_day >= 8) & (hours_of_day < 18), 1, 0)

        if scenario_type == "heatwave":
            # Extreme heatwave: min 29°C at night, peaking at 38°C at 14:00
            amb_temp = 33.5 + 4.5 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))
            amb_rh = 60.0 - 15.0 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))
            amb_pm25 = 25.0 + 8.0 * np.sin(hours_of_day * 0.5)
            occ_count = np.where(is_workhour == 1, 2, 0)
            p_int = np.where(is_workhour == 1, 1.2 + 0.4 * np.sin(hours_of_day), 0.2)
        elif scenario_type == "large_occupancy":
            # High occupant load: 8-10 occupants in large meeting/open space
            amb_temp = 30.0 + 4.0 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))
            amb_rh = 68.0 - 15.0 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))
            amb_pm25 = 25.0 + 10.0 * np.sin(hours_of_day * 0.5)
            occ_count = np.where(is_workhour == 1, 9, 0)
            p_int = np.where(is_workhour == 1, 5.0 + 1.2 * np.sin(hours_of_day), 0.4)
        elif scenario_type == "smog_pollution":
            # Severe outdoor smog: ambient PM2.5 consistently high (> 60 µg/m³, peaking at 85 µg/m³)
            amb_temp = 28.0 + 3.0 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))
            amb_rh = 80.0 + 8.0 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))  # Rain/high humidity
            # Smog PM2.5 strictly >= 55, mean ~ 72, peak ~ 88 µg/m³
            amb_pm25 = 70.0 + 18.0 * np.sin(hours_of_day * (2.0 * np.pi / 24.0))
            occ_count = np.where(is_workhour == 1, 4, 0)
            p_int = np.where(is_workhour == 1, 2.5 + 0.8 * np.sin(hours_of_day), 0.3)
        else:
            # Standard Bangkok-like diurnal temperature: min at 05:00 (26°C), peak at 14:00 (34°C)
            amb_temp = 30.0 + 4.0 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))
            amb_rh = 68.0 - 15.0 * np.sin((hours_of_day - 9.0) * (2.0 * np.pi / 24.0))
            amb_pm25 = 25.0 + 10.0 * np.sin(hours_of_day * 0.5)
            occ_count = np.where(is_workhour == 1, 4, 0)
            p_int = np.where(is_workhour == 1, 2.5 + 0.8 * np.sin(hours_of_day), 0.3)

        return pd.DataFrame({
            "hour": hours_of_day,
            "dayofweek": dows,
            "day_of_week": dows,
            "is_workhour": is_workhour,
            "ambient_temp_est": amb_temp,
            "ambient_rh_est": amb_rh,
            "ambient_co2_est": np.full(n_hours, 415.0),
            "ambient_pm25_est": amb_pm25,
            "internal_heat_kw": p_int,
            "occupancy_est": occ_count,
            "temp_c": amb_temp - 3.0,
            "humidity_rh": amb_rh - 5.0
        })

    def reset(
        self,
        start_idx: Optional[int] = None,
        initial_temp: Optional[float] = None,
        initial_rh: Optional[float] = None,
        initial_co2: Optional[float] = None,
        initial_pm25: Optional[float] = None,
        occupancy: Optional[int] = None,
        seed: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset environment to clean initial conditions with complete episode isolation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.window_state = 0
        self.last_action = None
        self.episode_switching_count = 0
        for k in self.cumul_energy:
            self.cumul_energy[k] = 0.0
        for k in self.instant_power:
            self.instant_power[k] = 0.0

        self._ensure_row_cache()
        max_start = self._cached_df_len - self.episode_length - 2
        if max_start <= 0:
            self.data_idx = 0
        elif start_idx is None:
            self.data_idx = int(self.rng.integers(0, max_start))
        else:
            self.data_idx = min(max(0, start_idx), max_start)

        idx = self.data_idx
        self.current_temp = float(initial_temp) if initial_temp is not None else self._c_temp[idx]
        self.current_rh = float(initial_rh) if initial_rh is not None else self._c_rh[idx]
        self.current_co2 = float(initial_co2) if initial_co2 is not None else 650.0
        self.current_pm25 = float(initial_pm25) if initial_pm25 is not None else 18.0

        if occupancy is not None:
            self.occupancy_count = int(occupancy)
        else:
            self.occupancy_count = self._c_occ[idx]

        p_int = self._c_p_int[idx]
        hour = self._c_hour[idx]
        dow = self._c_dow[idx]
        is_workhour = self._c_work[idx]

        state_tuple = discretize_state_720(
            self.current_temp, self.current_rh, self.current_co2, self.current_pm25, p_int, hour, dow
        )
        self.current_state_idx = state_tuple_to_index_720(state_tuple)

        info = self._build_info_dict(
            state_tuple, action=None, hour=hour, dow=dow, is_workhour=is_workhour, idx=idx
        )
        return self.current_state_idx, info

    def _calculate_actuator_powers_and_effects(
        self,
        template: ActionTemplate,
        current_temp: float,
        current_rh: float,
        ambient_temp: float,
        ambient_rh: float
    ) -> Dict[str, Any]:
        """
        Determine exact electrical power drawn and physical thermal/latent/air effects
        for each of the 7 actuator groups according to the chosen ActionTemplate.
        """
        powers = {}
        effects = {}
        spec = self.power_specs

        # 1. Air Conditioner (Cooling)
        cooling_thermal_kw = 0.0
        cooling_latent_removal_kg_h = 0.0
        ac_stage = "OFF"
        ac_standby_power = spec["ac_off"]

        if template.ac_mode != "OFF" and template.ac_setpoint_c is not None:
            gap = current_temp - template.ac_setpoint_c
            if gap <= 0.0:
                # Room is already cooler than or at setpoint -> compressor idle in standby
                ac_power = ac_standby_power
                ac_stage = "OFF"
                cooling_thermal_kw = 0.0
                cooling_latent_removal_kg_h = 0.0
            elif 0.0 < gap <= 0.4:
                ac_power = spec["ac_low"]
                ac_stage = "LOW"
            elif 0.4 < gap <= 0.8:
                ac_power = spec["ac_med"]
                ac_stage = "MED"
            else:
                ac_power = spec["ac_high"]
                ac_stage = "HIGH"

            if ac_stage != "OFF":
                # Active cooling capacity derived strictly from active compressor power (excluding standby electronics)
                active_electrical_kw = max(0.0, ac_power - ac_standby_power)
                cooling_thermal_kw = active_electrical_kw * spec["ac_cop"]
                # Physical Cooling Coil Dehumidification Model (ASHRAE Handbook of Fundamentals):
                # Moisture removal: m_latent (kg/h) = rho_air * V_coil * (w_room - w_coil_sat) * PLF
                # Parameters:
                # - T_coil = 12.0°C (leaving coil dew point temperature in [11, 14]°C)
                # - V_coil_m3_h = 0.08 m³/s per kW design load * 3600 (recirculated evaporator airflow)
                # - PLF = active_electrical_kw / (spec["ac_high"] - ac_standby_power) (part-load fraction)
                # - Upper bound: latent removal cannot exceed total active cooling power: Q_sens * 3600 / 2450
                w_room = humidity_ratio_from_rh(current_temp, current_rh)
                t_coil = 12.0  # °C leaving coil dew point
                w_coil_sat = humidity_ratio_from_rh(t_coil, 100.0)
                delta_w = max(0.0, w_room - w_coil_sat)

                rho_air = 1.204  # kg/m³
                v_coil_spec = 0.08  # m³/s per kW design load
                v_coil_m3_h = v_coil_spec * self.design_cooling_load_kw * 3600.0

                max_active_kw = max(0.01, spec["ac_high"] - ac_standby_power)
                part_load_fraction = min(1.0, active_electrical_kw / max_active_kw)

                m_coil_removal_kg_h = rho_air * v_coil_m3_h * delta_w * part_load_fraction
                max_latent_capacity_kg_h = (cooling_thermal_kw * 3600.0) / 2450.0  # h_fg = 2450 kJ/kg
                cooling_latent_removal_kg_h = min(m_coil_removal_kg_h, max_latent_capacity_kg_h)
        else:
            ac_power = ac_standby_power
            ac_stage = "OFF"
            cooling_thermal_kw = 0.0
            cooling_latent_removal_kg_h = 0.0

        powers["cooling_kw"] = ac_power
        effects["cooling_thermal_kw"] = cooling_thermal_kw
        effects["ac_latent_kg_h"] = cooling_latent_removal_kg_h
        effects["ac_stage"] = ac_stage

        # 2. Heating
        heater_thermal_kw = 0.0
        heater_stage = "OFF"
        if template.heater_mode != "OFF" and template.heater_setpoint_c is not None:
            gap_h = template.heater_setpoint_c - current_temp
            if gap_h > 0.0:
                if template.heater_mode == "HIGH" or gap_h > 2.0:
                    heater_p = spec["heater_high"]
                    heater_stage = "HIGH"
                else:
                    heater_p = spec["heater_low"]
                    heater_stage = "LOW"
                heater_thermal_kw = heater_p * 0.98  # Electric resistance / heat pump
            else:
                heater_p = spec["heater_off"]
        else:
            heater_p = spec["heater_off"]

        powers["heating_kw"] = heater_p
        effects["heater_thermal_kw"] = heater_thermal_kw
        effects["heater_stage"] = heater_stage

        # 3. Mechanical Ventilation (Supply + Exhaust Fans)
        vent_flow = template.vent_flow_m3_h
        if template.vent_fan_speed == "OFF":
            vent_p = 0.0
        elif template.vent_fan_speed == "LOW":
            vent_p = spec["vent_fan_low"] + spec["exhaust_fan_power"] * 0.5
        elif template.vent_fan_speed == "MED":
            vent_p = spec["vent_fan_med"] + spec["exhaust_fan_power"]
        else:
            vent_p = spec["vent_fan_high"] + spec["exhaust_fan_power"] * 1.5

        powers["ventilation_kw"] = vent_p
        effects["vent_flow_m3_h"] = vent_flow

        # 4. Window / Damper Natural Airflow
        # If window is open, calculate buoyancy & wind driven exchange: V_win = C_d * A_win * sqrt(g * H * deltaT / T)
        if template.window_open == 1:
            delta_t_win = abs(ambient_temp - current_temp)
            # Flow rate estimate: approx 150 m³/h per open m² under mild temperature difference
            window_flow_m3_h = self.room.window_area_m2 * (100.0 + 35.0 * math.sqrt(max(0.1, delta_t_win)))
        else:
            window_flow_m3_h = 0.0
        effects["window_flow_m3_h"] = window_flow_m3_h

        # 5. Humidifier
        if template.humidifier_mode == "LOW":
            humid_p = spec["humidifier_low"]
            moisture_add_kg_h = template.humidifier_rate_kg_h
        elif template.humidifier_mode in ["MED", "HIGH"]:
            humid_p = spec["humidifier_high"]
            moisture_add_kg_h = template.humidifier_rate_kg_h
        else:
            humid_p = 0.0
            moisture_add_kg_h = 0.0

        powers["humidifier_kw"] = humid_p
        effects["moisture_add_kg_h"] = moisture_add_kg_h

        # 6. Dehumidifier
        if template.dehumidifier_mode != "OFF":
            dehumid_p = spec["dehumidifier_power"]
            moisture_remove_kg_h = template.dehumidifier_rate_kg_h
        else:
            dehumid_p = 0.0
            moisture_remove_kg_h = 0.0

        powers["dehumidifier_kw"] = dehumid_p
        effects["moisture_remove_kg_h"] = moisture_remove_kg_h

        # 7. Air Filter (Purifier)
        if template.filter_speed == "LOW":
            filter_p = spec["filter_low"]
        elif template.filter_speed == "MED":
            filter_p = spec["filter_med"]
        elif template.filter_speed == "HIGH":
            filter_p = spec["filter_high"]
        else:
            filter_p = 0.0

        powers["filtration_kw"] = filter_p
        effects["filter_cadr_m3_h"] = template.filter_cadr_m3_h

        powers["total_kw"] = sum(powers.values())
        return {"powers": powers, "effects": effects}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Execute one discrete simulation step (default 1 hour).
        Simulates physical mass & energy balances across all 7 device groups.
        """
        if action not in MULTI_DEVICE_ACTION_TEMPLATES:
            raise ValueError(f"Invalid action {action}! Valid actions are 0 to {len(MULTI_DEVICE_ACTION_TEMPLATES)-1}")

        template = MULTI_DEVICE_ACTION_TEMPLATES[action]
        action_changed = (action != self.last_action) if self.last_action is not None else False
        if action_changed:
            self.episode_switching_count += 1
        self.last_action = action
        self.window_state = template.window_open

        self._ensure_row_cache()
        idx = self.data_idx
        p_int = self._c_p_int[idx]
        amb_t = self._c_amb_t[idx]
        amb_rh = self._c_amb_rh[idx]
        amb_co2 = self._c_amb_co2[idx]
        amb_pm25 = self._c_amb_pm25[idx]
        hour = self._c_hour[idx]
        dow = self._c_dow[idx]
        is_workhour = self._c_work[idx]

        # -------------------------------------------------------------------
        # Multi-rate Physics Integration (12 Sub-steps of 5 min = dt_phys 300s)
        # Prevents Euler numerical instability, eliminates PM2.5/CO2 sawtooth oscillation,
        # and allows realistic mid-hour compressor staging upon reaching setpoint.
        # Control decision timestep remains 1 hour (24 steps / episode).
        # -------------------------------------------------------------------
        n_sub = 12
        dt_sub = self.step_duration_hours / n_sub  # 1.0 / 12 = 0.08333... h (300 seconds)
        noise_scale = 1.0 / math.sqrt(n_sub)

        vol = self.room.volume_m3
        c_thermal = self.room.total_heat_capacity_kj_k
        v_infil = self.room.base_infiltration_ach * vol
        env_coeff_kw_k = (self.room.u_envelope * self.room.envelope_area_m2) / 1000.0
        air_mass_kg = self.room.air_mass_kg
        w_outdoor = humidity_ratio_from_rh(amb_t, amb_rh)

        substep_powers = {k: 0.0 for k in self.instant_power}
        last_effects = {}

        for _ in range(n_sub):
            # Recalculate actuator powers and effects at current sub-step conditions
            act_calc = self._calculate_actuator_powers_and_effects(
                template, self.current_temp, self.current_rh, amb_t, amb_rh
            )
            powers = act_calc["powers"]
            effects = act_calc["effects"]
            last_effects = effects

            # Accumulate energy & track mean power across sub-steps
            for dev, p_kw in powers.items():
                kwh_key = dev.replace("_kw", "_kwh")
                self.cumul_energy[kwh_key] += p_kw * dt_sub
                substep_powers[dev] += p_kw / n_sub

            # A. Total Airflows (m³/h)
            v_vent = effects["vent_flow_m3_h"]
            v_win = effects["window_flow_m3_h"]
            total_fresh_air_m3_h = v_infil + v_vent + v_win

            # B. Sensible Thermal Balance (including occupant sensible heat: 75 W / person)
            q_envelope_kw = env_coeff_kw_k * (amb_t - self.current_temp)
            q_vent_kw = (1.21 * total_fresh_air_m3_h * (amb_t - self.current_temp)) / 3600.0
            q_occupants_kw = 0.075 * self.occupancy_count
            q_internal_kw = p_int + q_occupants_kw
            q_cooling_kw = effects["cooling_thermal_kw"]
            q_heating_kw = effects["heater_thermal_kw"]

            net_heat_rate_kw = q_envelope_kw + q_vent_kw + q_internal_kw + q_heating_kw - q_cooling_kw
            delta_temp = (net_heat_rate_kw * 3600.0 * dt_sub) / c_thermal
            delta_temp += float(self.rng.normal(0.0, 0.05 * noise_scale))
            self.current_temp = _clip(self.current_temp + delta_temp, 15.0, 40.0)

            # C. Psychrometric Moisture Balance
            w_indoor = humidity_ratio_from_rh(self.current_temp, self.current_rh)
            m_occ_kg_h = self.occupancy_count * 0.050
            m_humid_kg_h = effects["moisture_add_kg_h"]
            m_dehumid_kg_h = effects["moisture_remove_kg_h"]
            m_ac_latent_kg_h = effects["ac_latent_kg_h"]
            m_vent_kg_h = 1.204 * total_fresh_air_m3_h * (w_outdoor - w_indoor)

            net_moisture_rate_kg_h = (
                m_occ_kg_h + m_humid_kg_h + m_vent_kg_h - m_dehumid_kg_h - m_ac_latent_kg_h
            )
            delta_w = (net_moisture_rate_kg_h * dt_sub) / air_mass_kg
            next_w = max(1e-5, w_indoor + delta_w)
            next_rh = rh_from_humidity_ratio(self.current_temp, next_w)
            self.current_rh = _clip(next_rh + float(self.rng.normal(0.0, 0.2 * noise_scale)), 15.0, 95.0)

            # D. CO2 Mass Balance
            g_co2_ppm_m3_h = self.occupancy_count * 18000.0
            delta_co2 = ((g_co2_ppm_m3_h - total_fresh_air_m3_h * (self.current_co2 - amb_co2)) / vol) * dt_sub
            self.current_co2 = _clip(self.current_co2 + delta_co2 + float(self.rng.normal(0.0, 5.0 * noise_scale)), 400.0, 3500.0)

            # E. PM2.5 Particulate Mass Balance
            g_pm25 = 15.0 if self.occupancy_count > 0 else 2.0
            cadr = effects["filter_cadr_m3_h"]
            vent_filtered_supply = v_vent * (1.0 - 0.40)
            unfiltered_air = v_infil + v_win + vent_filtered_supply
            k_dep = 0.15
            removal_rate_m3_h = cadr + k_dep * vol + total_fresh_air_m3_h
            delta_pm = ((g_pm25 * vol + unfiltered_air * amb_pm25 - removal_rate_m3_h * self.current_pm25) / vol) * dt_sub
            self.current_pm25 = _clip(self.current_pm25 + delta_pm + float(self.rng.normal(0.0, 0.8 * noise_scale)), 3.0, 200.0)

        # Update instantaneous power tracking
        self.instant_power = substep_powers

        # -------------------------------------------------------------------
        # Clock & State Advancement
        # -------------------------------------------------------------------
        self.current_step += 1
        self.data_idx += 1

        done = (self.current_step >= self.episode_length)
        truncated = False

        next_idx = self.data_idx if self.data_idx < self._cached_df_len else (self._cached_df_len - 1)
        # Dynamic occupancy update from next row (Fixing F3)
        self.occupancy_count = self._c_occ[next_idx]
        next_p_int = self._c_p_int[next_idx]
        next_hour = self._c_hour[next_idx]
        next_dow = self._c_dow[next_idx]

        next_state_tuple = discretize_state_720(
            self.current_temp, self.current_rh, self.current_co2, self.current_pm25,
            next_p_int, next_hour, next_dow
        )
        self.current_state_idx = state_tuple_to_index_720(next_state_tuple)

        info = self._build_info_dict(
            next_state_tuple, action=action, hour=hour, dow=dow, is_workhour=is_workhour, idx=idx,
            powers=substep_powers, effects=last_effects, action_changed=action_changed
        )

        # Temporary placeholder reward; true reward computed via MultiDeviceRewardFunction
        reward = 0.0
        return self.current_state_idx, reward, done, truncated, info

    def _build_info_dict(
        self,
        state_tuple: Tuple[int, int, int, int, int, int],
        action: Optional[int],
        hour: int,
        dow: int,
        is_workhour: Optional[int] = None,
        idx: Optional[int] = None,
        row: Optional[pd.Series] = None,
        powers: Optional[Dict[str, float]] = None,
        effects: Optional[Dict[str, Any]] = None,
        action_changed: bool = False
    ) -> Dict[str, Any]:
        """Assemble comprehensive diagnostic telemetry with provenance classification."""
        powers = powers or self.instant_power
        effects = effects or {}
        template = MULTI_DEVICE_ACTION_TEMPLATES[action] if action is not None else None
        if idx is not None:
            wk = is_workhour if is_workhour is not None else self._c_work[idx]
            amb_t = self._c_amb_t[idx]
            amb_rh = self._c_amb_rh[idx]
            amb_co2 = self._c_amb_co2[idx]
            amb_pm = self._c_amb_pm25[idx]
            p_raw = self._c_raw_p_int[idx]
        else:
            wk = int(row.get("is_workhour", 1 if (dow < 5 and 8 <= hour < 18) else 0)) if row is not None else (1 if (dow < 5 and 8 <= hour < 18) else 0)
            amb_t = float(row.get("ambient_temp_est", 32.0)) if row is not None else 32.0
            amb_rh = float(row.get("ambient_rh_est", 65.0)) if row is not None else 65.0
            amb_co2 = float(row.get("ambient_co2_est", 415.0)) if row is not None else 415.0
            amb_pm = float(row.get("ambient_pm25_est", 25.0)) if row is not None else 25.0
            p_raw = float(row.get("internal_heat_kw", 2.0)) if row is not None else 2.0

        return {
            "step": self.current_step,
            "hour": hour,
            "dayofweek": dow,
            "is_workhour": wk,
            "temp_c": round(self.current_temp, 2),
            "humidity_rh": round(self.current_rh, 1),
            "co2_ppm": round(self.current_co2, 1),
            "pm25_ug_m3": round(self.current_pm25, 1),
            "occupancy_count": self.occupancy_count,
            "window_open": self.window_state,
            "action": action,
            "action_name": template.name if template else "INITIAL_STATE",
            "state_tuple": state_tuple,
            "ambient_temp": round(amb_t, 2),
            "ambient_rh": round(amb_rh, 1),
            "ambient_co2": round(amb_co2, 1),
            "ambient_pm25": round(amb_pm, 1),
            "internal_heat_kw": round(p_raw, 2),
            "powers_kw": {k: round(v, 3) for k, v in powers.items()},
            "cumul_energy_kwh": {k: round(v, 3) for k, v in self.cumul_energy.items()},
            "effects": effects,
            "action_changed": action_changed,
            "episode_switching_count": self.episode_switching_count,
            "room_geometry": self.room.to_dict(),
            "provenance": {
                "temp_c": ProvenanceType.SIMULATED.value,
                "humidity_rh": ProvenanceType.SIMULATED.value,
                "co2_ppm": ProvenanceType.SIMULATED.value,
                "pm25_ug_m3": ProvenanceType.SIMULATED.value,
                "occupancy_count": ProvenanceType.SIMULATED_ASSUMED.value,
                "window_open": ProvenanceType.SIMULATED_ASSUMED.value,
                "ambient_temp": ProvenanceType.SIMULATED_ASSUMED.value,
                "ambient_rh": ProvenanceType.SIMULATED_ASSUMED.value,
                "ambient_co2": ProvenanceType.SIMULATED_ASSUMED.value,
                "ambient_pm25": ProvenanceType.SIMULATED_ASSUMED.value,
                "internal_heat_kw": ProvenanceType.DERIVED.value if not self.is_synthetic_scenario else ProvenanceType.SIMULATED_ASSUMED.value,
                "powers_kw": ProvenanceType.SIMULATED.value,
                "cumul_energy_kwh": ProvenanceType.SIMULATED.value,
                "room_geometry": ProvenanceType.SIMULATED_ASSUMED.value,
                "is_workhour": ProvenanceType.DERIVED.value if not self.is_synthetic_scenario else ProvenanceType.SIMULATED_ASSUMED.value
            }
        }
