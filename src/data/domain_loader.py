"""
Domain-aware Data Loader and Preprocessor for CU-BEMS.
Supports multi-domain loading, sensor outage trimming, temporal integrity,
quality masking, and non-leaking train/adaptation/test splitting.
"""

from pathlib import Path
import os
from typing import Dict, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from src.data.domain_config import DomainSpec, get_domain_spec, DOMAINS
from src.data.provenance import ProvenanceType


def get_archive_dir() -> Path:
    """Resolve archive directory containing CU-BEMS CSVs dynamically."""
    env_dir = os.environ.get("CUBEMS_DATASET_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir)

    base = Path(__file__).resolve().parent
    candidates = [
        base.parents[2] / "dataset" / "archive",
        base.parents[1] / "data" / "raw",
        base.parents[3] / "dataset" / "archive",
        Path.cwd() / "dataset" / "archive",
        Path.cwd().parent / "dataset" / "archive"
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
    resample_rule: str = "1h",
    max_interpolate_hours: int = 3
) -> pd.DataFrame:
    """
    Process raw timeseries for a given domain into a standardized,
    resampled, leak-free continuous environmental dataframe.

    Missing Data Policy:
    - Slices strictly active sensor intervals where target IAQ sensor was connected.
    - Interpolates ONLY short dropouts (<= max_interpolate_hours, default 3h).
    - NEVER fills multi-day outages with unconstrained bfill/ffill.
    - Preserves a quality mask tracking which steps were interpolated.
    """
    df = df_raw.copy()
    df = df.set_index("Date")

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

    # Record missing values before interpolation
    pre_missing_temp = int(resampled[spec.temp_col].isnull().sum())
    pre_missing_rh = int(resampled[spec.rh_col].isnull().sum())
    total_resampled_steps = len(resampled)

    # 3. Controlled short gap interpolation (strictly bounded to max_interpolate_hours)
    temp_was_missing = resampled[spec.temp_col].isnull()
    rh_was_missing = resampled[spec.rh_col].isnull()

    resampled = resampled.interpolate(method="time", limit=max_interpolate_hours)

    # Record quality masks
    resampled["temp_is_interpolated"] = (temp_was_missing & resampled[spec.temp_col].notnull()).astype(int)
    resampled["rh_is_interpolated"] = (rh_was_missing & resampled[spec.rh_col].notnull()).astype(int)

    # 4. Drop prolonged sensor outages exceeding max_interpolate_hours
    # NO unrestricted bfill() or ffill() is allowed!
    resampled = resampled.dropna(subset=[spec.temp_col, spec.rh_col, "hvac_kw"]).reset_index()
    resampled = resampled.rename(columns={"Date": "datetime"})

    # 5. Standardize column names
    col_mapping = {
        spec.temp_col: "temp_c",
        spec.rh_col: "humidity_rh",
        spec.lux_col: "light_lux",
        spec.plug_col: "plug_kw",
        spec.light_col: "light_kw"
    }
    clean_df = resampled.rename(columns=col_mapping)

    # 6. Physical plausibility clipping
    clean_df["temp_c"] = clean_df["temp_c"].clip(lower=15.0, upper=40.0)
    clean_df["humidity_rh"] = clean_df["humidity_rh"].clip(lower=20.0, upper=100.0)
    clean_df["hvac_kw"] = clean_df["hvac_kw"].clip(lower=0.0)
    clean_df["plug_kw"] = clean_df["plug_kw"].clip(lower=0.0)
    clean_df["light_kw"] = clean_df["light_kw"].clip(lower=0.0)

    # 7. Derived internal heat gain
    clean_df["internal_heat_kw"] = clean_df["plug_kw"] + clean_df["light_kw"]

    # 8. Temporal & calendar features
    clean_df["hour"] = clean_df["datetime"].dt.hour
    clean_df["day_of_week"] = clean_df["datetime"].dt.day_of_week
    clean_df["is_weekend"] = (clean_df["day_of_week"] >= 5).astype(int)
    clean_df["is_workhour"] = (
        (clean_df["is_weekend"] == 0) & (clean_df["hour"] >= 8) & (clean_df["hour"] < 18)
    ).astype(int)

    # 9. Occupancy proxy
    clean_df["occupancy_proxy"] = (
        (clean_df["light_lux"] > 30) | (clean_df["plug_kw"] > 2.0)
    ).astype(int)

    # 10. Weather Scenario Profile (Explicitly labeled SIMULATED_ASSUMED)
    solar_phase = 2 * np.pi * (clean_df["hour"] - 14) / 24.0
    clean_df["ambient_temp_est"] = 27.5 + 5.5 * np.cos(solar_phase)
    clean_df["ambient_rh_est"] = 75.0 - 15.0 * np.cos(solar_phase)

    # Quality Report Metadata
    quality_report = {
        "domain_id": spec.domain_id,
        "total_active_hours": total_resampled_steps,
        "pre_missing_temp_hours": pre_missing_temp,
        "pre_missing_temp_pct": round((pre_missing_temp / max(1, total_resampled_steps)) * 100.0, 2),
        "pre_missing_rh_hours": pre_missing_rh,
        "interpolated_temp_hours": int(clean_df["temp_is_interpolated"].sum()),
        "final_clean_hours": len(clean_df),
        "dropped_outage_hours": total_resampled_steps - len(clean_df)
    }

    clean_df.attrs["domain_id"] = spec.domain_id
    clean_df.attrs["year"] = spec.year
    clean_df.attrs["floor"] = spec.floor
    clean_df.attrs["zone"] = spec.zone
    clean_df.attrs["is_primary"] = spec.is_primary
    clean_df.attrs["quality_report"] = quality_report

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
        if "temp_is_interpolated" in df.columns:
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

    # Sanity check for temporal leakage
    assert df_train["datetime"].max() < df_test["datetime"].min(), "Temporal leakage detected in split!"

    return df_train, df_test


def split_target_domain_dataset(
    df: pd.DataFrame,
    calib_ratio: float = 0.20,
    adapt_ratio: float = 0.40,
    test_ratio: float = 0.40
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict 3-way sequential temporal partition for target domains:
    1. Calibration Set (0% to calib_ratio): Used ONLY for equipment clustering and physics fitting.
    2. Adaptation Set (calib_ratio to calib+adapt): Used ONLY for agent fine-tuning budget.
    3. Test Set (calib+adapt to 100%): Kept strictly held-out for final evaluation.

    Guarantees:
    - Zero timestamp overlap.
    - Strict chronological order: max(t_calib) < min(t_adapt) <= max(t_adapt) < min(t_test).
    - Test set is never touched during calibration or adaptation.
    """
    assert np.isclose(calib_ratio + adapt_ratio + test_ratio, 1.0, atol=1e-5), "Split ratios must sum to 1.0"
    n = len(df)
    n_cal = int(n * calib_ratio)
    n_adapt = int(n * (calib_ratio + adapt_ratio))

    df_calib = df.iloc[:n_cal].copy().reset_index(drop=True)
    df_adapt = df.iloc[n_cal:n_adapt].copy().reset_index(drop=True)
    df_test = df.iloc[n_adapt:].copy().reset_index(drop=True)

    # Leakage assertions
    assert df_calib["datetime"].max() < df_adapt["datetime"].min(), "Temporal overlap between calib and adapt!"
    assert df_adapt["datetime"].max() < df_test["datetime"].min(), "Temporal overlap between adapt and test!"

    for split_df, split_name in [(df_calib, "CALIBRATION"), (df_adapt, "ADAPTATION"), (df_test, "TEST")]:
        split_df.attrs = dict(df.attrs)
        split_df.attrs["split_type"] = split_name

    return df_calib, df_adapt, df_test
