"""
Preprocessing and feature engineering module for SmartHVACControlByDoubleQLearning.
Prepares continuous and discretized state representations for Reinforcement Learning.
"""

from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd


# Default temperature bins (in deg C)
DEFAULT_TEMP_BINS = [21.0, 22.5, 24.5, 26.0]

# Default internal heat load bins (in kW)
DEFAULT_LOAD_BINS = [2.0, 8.0]

# Default time bins
# 0: Night/Off-hours (20:00 - 07:00)
# 1: Work morning (07:00 - 12:00)
# 2: Work afternoon (12:00 - 20:00)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal, thermal load, and occupancy features to processed dataframe.
    """
    df = df.copy()
    if not np.issubdtype(df["datetime"].dtype, np.datetime64):
        df["datetime"] = pd.to_datetime(df["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Work hours: Weekdays 08:00 - 18:00
    df["is_workhour"] = (
        (df["is_weekend"] == 0) & (df["hour"] >= 8) & (df["hour"] < 18)
    ).astype(int)

    # Occupancy proxy from lux and plug power
    df["occupancy_proxy"] = (
        (df["light_lux"] > 30) | (df["plug_kw"] > 2.0)
    ).astype(int)

    # Ambient outdoor temperature estimation (Bangkok typical diurnal curve)
    # T_amb ~ 27°C base + 6°C peak at 14:00
    solar_phase = 2 * np.pi * (df["hour"] - 14) / 24.0
    df["ambient_temp_est"] = 27.5 + 5.5 * np.cos(solar_phase)

    # Time binning for state space
    def get_time_bin(hour: int) -> int:
        if 7 <= hour < 12:
            return 1  # Morning
        elif 12 <= hour < 20:
            return 2  # Afternoon / Peak
        else:
            return 0  # Night / Off-hours

    df["time_bin"] = df["hour"].apply(get_time_bin)

    return df


def discretize_state(
    temp_c: float,
    internal_heat_kw: float,
    hour: int,
    temp_bins: list = DEFAULT_TEMP_BINS,
    load_bins: list = DEFAULT_LOAD_BINS
) -> Tuple[int, int, int]:
    """
    Discretize continuous environmental variables into a tuple (temp_idx, load_idx, time_idx).
    
    Returns:
    --------
    (temp_bin, load_bin, time_bin): Tuple of discrete bin indices
    """
    # 1. Temperature bin (0 to len(temp_bins))
    temp_bin = int(np.digitize(temp_c, temp_bins))

    # 2. Load bin (0 to len(load_bins))
    load_bin = int(np.digitize(internal_heat_kw, load_bins))

    # 3. Time bin (0, 1, 2)
    if 7 <= hour < 12:
        time_bin = 1
    elif 12 <= hour < 20:
        time_bin = 2
    else:
        time_bin = 0

    return (temp_bin, load_bin, time_bin)


def state_to_index(
    state_tuple: Tuple[int, int, int],
    n_temp_bins: int = 5,
    n_load_bins: int = 3,
    n_time_bins: int = 3
) -> int:
    """
    Convert 3D discrete state tuple into a single 1D flat integer index for Q-tables.
    """
    t_idx, l_idx, h_idx = state_tuple
    # Clamping for bounds safety
    t_idx = min(max(t_idx, 0), n_temp_bins - 1)
    l_idx = min(max(l_idx, 0), n_load_bins - 1)
    h_idx = min(max(h_idx, 0), n_time_bins - 1)

    return t_idx * (n_load_bins * n_time_bins) + l_idx * n_time_bins + h_idx


def index_to_state(
    state_idx: int,
    n_temp_bins: int = 5,
    n_load_bins: int = 3,
    n_time_bins: int = 3
) -> Tuple[int, int, int]:
    """
    Convert 1D flat state index back to 3D state tuple.
    """
    h_idx = state_idx % n_time_bins
    rem = state_idx // n_time_bins
    l_idx = rem % n_load_bins
    t_idx = rem // n_load_bins
    return (t_idx, l_idx, h_idx)


def get_state_space_size(
    n_temp_bins: int = 5,
    n_load_bins: int = 3,
    n_time_bins: int = 3
) -> int:
    """Return total number of discrete states."""
    return n_temp_bins * n_load_bins * n_time_bins
