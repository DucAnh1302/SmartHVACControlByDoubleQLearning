"""
Domain-aware Data Loader and Preprocessor for CU-BEMS.
Supports multi-domain loading, sensor outage trimming, temporal integrity,
and non-leaking train/test splitting.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.domain_config import DomainSpec, get_domain_spec, DOMAINS
from src.data.provenance import ProvenanceType


def get_archive_dir() -> Path:
    """Resolve archive directory containing CU-BEMS CSVs."""
    candidates = [
        Path(r"C:\Users\LENOVO\Desktop\SmartHVAC\dataset\archive"),
        Path(__file__).resolve().parents[3] / "dataset" / "archive",
        Path(__file__).resolve().parents[2] / "data" / "raw"
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.csv")):
            return c
    return candidates[0]


def get_processed_dir() -> Path:
    """Resolve data/processed directory."""
    d = Path(__file__).resolve().parents[2] / "data" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_domain_raw_dataframe(spec: DomainSpec, archive_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load raw CSV corresponding to a DomainSpec."""
    adir = archive_dir or get_archive_dir()
    filename = f"{spec.year}Floor{spec.floor}.csv"
    filepath = adir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    needed_cols = set(["Date", spec.temp_col, spec.rh_col, spec.lux_col, spec.plug_col, spec.light_col])
    for ac in spec.ac_cols:
        needed_cols.add(ac)

    df = pd.read_csv(filepath, usecols=lambda c: c in needed_cols)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def preprocess_domain_timeseries(
    spec: DomainSpec,
    df_raw: pd.DataFrame,
    resample_rule: str = "1h"
) -> pd.DataFrame:
    """
    Process raw timeseries for a given domain into a standardized,
    resampled, leak-free continuous environmental dataframe.
    """
    df = df_raw.copy()
    df.set_index("Date", inplace=True)

    # 1. Check temperature validity window
    temp_valid = df[spec.temp_col].notnull()
    if not temp_valid.any():
        raise ValueError(f"Domain {spec.domain_id} has no valid temperature records!")

    first_valid = df[temp_valid].index.min()
    last_valid = df[temp_valid].index.max()

    # Slice only the valid sensor active period
    df_active = df.loc[first_valid:last_valid].copy()

    # Sum multiple AC columns if present
    df_active["hvac_kw"] = df_active[list(spec.ac_cols)].sum(axis=1, min_count=1)

    # 2. Resample to regular intervals using mean
    cols_to_resample = ["hvac_kw", spec.temp_col, spec.rh_col, spec.lux_col, spec.plug_col, spec.light_col]
    resampled = df_active[cols_to_resample].resample(resample_rule).mean()

    # 3. Handle small dropouts via time-interpolation (max gap 6 hours)
    resampled = resampled.interpolate(method="time", limit=6)
    resampled = resampled.bfill().ffill()

    # Drop any remaining unfillable NaN
    resampled = resampled.dropna().reset_index()
    resampled.rename(columns={"Date": "datetime"}, inplace=True)

    # 4. Standardize column names
    col_mapping = {
        spec.temp_col: "temp_c",
        spec.rh_col: "humidity_rh",
        spec.lux_col: "light_lux",
        spec.plug_col: "plug_kw",
        spec.light_col: "light_kw"
    }
    clean_df = resampled.rename(columns=col_mapping)

    # 5. Physics and Sensor bounds
    clean_df["temp_c"] = clean_df["temp_c"].clip(lower=15.0, upper=40.0)
    clean_df["humidity_rh"] = clean_df["humidity_rh"].clip(lower=20.0, upper=100.0)
    clean_df["hvac_kw"] = clean_df["hvac_kw"].clip(lower=0.0)
    clean_df["plug_kw"] = clean_df["plug_kw"].clip(lower=0.0)
    clean_df["light_kw"] = clean_df["light_kw"].clip(lower=0.0)

    # 6. Derived internal heat gain
    clean_df["internal_heat_kw"] = clean_df["plug_kw"] + clean_df["light_kw"]

    # 7. Temporal & calendar features
    clean_df["hour"] = clean_df["datetime"].dt.hour
    clean_df["dayofweek"] = clean_df["datetime"].dt.dayofweek
    clean_df["is_weekend"] = (clean_df["dayofweek"] >= 5).astype(int)
    clean_df["is_workhour"] = (
        (clean_df["is_weekend"] == 0) & (clean_df["hour"] >= 8) & (clean_df["hour"] < 18)
    ).astype(int)

    # 8. Occupancy proxy
    clean_df["occupancy_proxy"] = (
        (clean_df["light_lux"] > 30) | (clean_df["plug_kw"] > 2.0)
    ).astype(int)

    # 9. External Weather Proxy (Bangkok tropical diurnal model aligned to local hour)
    solar_phase = 2 * np.pi * (clean_df["hour"] - 14) / 24.0
    clean_df["ambient_temp_est"] = 27.5 + 5.5 * np.cos(solar_phase)
    clean_df["ambient_rh_est"] = 75.0 - 15.0 * np.cos(solar_phase)

    # Attach domain metadata attribute
    clean_df.attrs["domain_id"] = spec.domain_id
    clean_df.attrs["year"] = spec.year
    clean_df.attrs["floor"] = spec.floor
    clean_df.attrs["zone"] = spec.zone
    clean_df.attrs["is_primary"] = spec.is_primary

    return clean_df


def get_domain_dataset(
    domain_id: str = "2018_F2_Z1",
    resample_rule: str = "1h",
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Load preprocessed domain dataset from cache, or recompute and cache.
    """
    spec = get_domain_spec(domain_id)
    cache_path = get_processed_dir() / f"{domain_id}_{resample_rule}.csv"

    if cache_path.exists() and not force_recompute:
        df = pd.read_csv(cache_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.attrs["domain_id"] = spec.domain_id
        df.attrs["year"] = spec.year
        df.attrs["floor"] = spec.floor
        df.attrs["zone"] = spec.zone
        df.attrs["is_primary"] = spec.is_primary
        return df

    # Otherwise compute fresh
    df_raw = load_domain_raw_dataframe(spec)
    clean_df = preprocess_domain_timeseries(spec, df_raw, resample_rule=resample_rule)
    clean_df.to_csv(cache_path, index=False)
    print(f"[DomainLoader] Cached {spec.domain_id} ({len(clean_df)} records) to {cache_path}")
    return clean_df


def split_temporal_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict sequential temporal split (NO shuffling, NO data leakage).
    First `train_ratio` fraction is Train, remaining is Test.
    """
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)
    df_test = df.iloc[n_train:].copy().reset_index(drop=True)

    df_train.attrs = dict(df.attrs)
    df_test.attrs = dict(df.attrs)

    return df_train, df_test
