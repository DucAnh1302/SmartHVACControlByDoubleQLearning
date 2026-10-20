"""
Data loader module for CU-BEMS (Smart Building Energy and IAQ Data).
Handles reading raw CSV files, cleaning missing values, and resampling.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np


def get_default_data_path() -> Path:
    """Resolve default raw dataset directory."""
    # Try looking in ../../dataset/archive first
    curr_dir = Path(__file__).resolve().parent
    candidates = [
        curr_dir.parents[2] / "dataset" / "archive",
        Path(r"C:\Users\LENOVO\Desktop\SmartHVAC\dataset\archive"),
        curr_dir.parents[1] / "data" / "raw",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.csv")):
            return c
    return candidates[0]


def load_raw_floor_data(
    year: int = 2019,
    floor: int = 2,
    raw_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Load raw CSV for a given year and floor from CU-BEMS dataset.
    """
    if raw_dir is None:
        data_dir = get_default_data_path()
    else:
        data_dir = Path(raw_dir)

    filename = f"{year}Floor{floor}.csv"
    filepath = data_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"[DataLoader] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def preprocess_cubems_data(
    df_raw: pd.DataFrame,
    zone: str = "z1",
    resample_rule: str = "1h",
    drop_unmetered_start: bool = True
) -> pd.DataFrame:
    """
    Extract key environmental features for the designated zone,
    handle missing values, compute total loads, and resample time-series.
    
    Parameters:
    -----------
    df_raw: Raw DataFrame with Date and sensor columns
    zone: Target zone prefix (e.g., 'z1', 'z2')
    resample_rule: Pandas frequency alias ('1h' for hourly, '15min' for 15-minute)
    drop_unmetered_start: Whether to drop initial period before sensors came online
    """
    df = df_raw.copy()
    df.set_index("Date", inplace=True)

    temp_col = f"{zone}_S1(degC)"
    rh_col = f"{zone}_S1(RH%)"
    lux_col = f"{zone}_S1(lux)"
    light_col = f"{zone}_Light(kW)"
    plug_col = f"{zone}_Plug(kW)"
    ac_col = f"{zone}_AC1(kW)"

    # Identify all AC columns across the floor for total floor consumption
    all_ac_cols = [c for c in df.columns if "AC" in c]
    df["total_floor_hvac_kw"] = df[all_ac_cols].sum(axis=1, min_count=1)

    # Check required columns
    required = [temp_col, rh_col, lux_col, light_col, plug_col, ac_col]
    available = [c for c in required if c in df.columns]
    if temp_col not in available:
        raise ValueError(f"Temperature column {temp_col} not found in dataset!")

    # If drop_unmetered_start is True, slice starting from first valid sensor reading
    if drop_unmetered_start:
        first_valid = df[temp_col].first_valid_index()
        if first_valid is not None:
            df = df.loc[first_valid:]

    # Resample to regular intervals using mean
    resampled = df[available + ["total_floor_hvac_kw"]].resample(resample_rule).mean()

    # Interpolate missing values for small sensor dropouts
    resampled = resampled.interpolate(method="time", limit=6)
    resampled = resampled.bfill().ffill()

    # Standardize column naming
    rename_dict = {
        temp_col: "temp_c",
        rh_col: "humidity_rh",
        lux_col: "light_lux",
        light_col: "light_kw",
        plug_col: "plug_kw",
        ac_col: "hvac_kw"
    }
    clean_df = resampled.rename(columns=rename_dict).reset_index()
    clean_df.rename(columns={"Date": "datetime"}, inplace=True)

    # Derived heat and occupancy features
    clean_df["internal_heat_kw"] = clean_df["light_kw"] + clean_df["plug_kw"]
    
    # Clip physically impossible sensor outliers if any
    clean_df["temp_c"] = clean_df["temp_c"].clip(lower=15.0, upper=40.0)
    clean_df["humidity_rh"] = clean_df["humidity_rh"].clip(lower=20.0, upper=100.0)
    clean_df["hvac_kw"] = clean_df["hvac_kw"].clip(lower=0.0)
    clean_df["plug_kw"] = clean_df["plug_kw"].clip(lower=0.0)
    clean_df["light_kw"] = clean_df["light_kw"].clip(lower=0.0)

    print(f"[DataLoader] Processed {len(clean_df)} records at frequency '{resample_rule}'. Date range: {clean_df['datetime'].min()} to {clean_df['datetime'].max()}")
    return clean_df


def save_processed_data(
    df: pd.DataFrame,
    output_filename: str,
    output_dir: Optional[str] = None
) -> Path:
    """Save processed dataframe to CSV."""
    if output_dir is None:
        curr_dir = Path(__file__).resolve().parent
        target_dir = curr_dir.parents[1] / "data" / "processed"
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / output_filename
    df.to_csv(out_path, index=False)
    print(f"[DataLoader] Saved processed data to: {out_path}")
    return out_path
