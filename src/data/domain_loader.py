"""
Domain-aware Data Loader and Preprocessor for CU-BEMS.
Supports multi-domain loading, sensor outage trimming, temporal integrity,
quality masking, and non-leaking train/adaptation/test splitting.
"""

from pathlib import Path
import os
import hashlib
from typing import Dict, Optional, Tuple, Union, Any, List
import numpy as np
import pandas as pd

from src.data.domain_config import DomainSpec, get_domain_spec, DOMAINS
from src.data.provenance import ProvenanceType

# Scaling and occupancy constants derived strictly from primary training domain (2018_F2_Z1 train split)
P95_TRAIN_ZONE_INTERNAL_HEAT_KW: float = 12.7947  # p95 of raw (plug + light) in train set (kW)
NOMINAL_ROOM_AREA_M2: float = 50.0                # Medium office benchmark geometry (m²)
TARGET_INTERNAL_LOAD_DENSITY_KW_M2: float = 0.035  # Standard office equipment/lighting density: 35 W/m²
INTERNAL_HEAT_SCALING_FACTOR: float = (TARGET_INTERNAL_LOAD_DENSITY_KW_M2 * NOMINAL_ROOM_AREA_M2) / P95_TRAIN_ZONE_INTERNAL_HEAT_KW  # ~0.136775


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

    # 7. Derived internal heat gain (Raw Zone Level & Scaled Room Level)
    clean_df["internal_heat_raw_kw"] = clean_df["plug_kw"] + clean_df["light_kw"]
    clean_df["internal_heat_kw"] = clean_df["internal_heat_raw_kw"]
    clean_df["internal_heat_scaled_kw"] = clean_df["internal_heat_raw_kw"] * INTERNAL_HEAT_SCALING_FACTOR

    # 8. Temporal & calendar features
    clean_df["hour"] = clean_df["datetime"].dt.hour
    clean_df["day_of_week"] = clean_df["datetime"].dt.day_of_week
    clean_df["dayofweek"] = clean_df["day_of_week"]  # Standardized alias for consistency
    clean_df["is_weekend"] = (clean_df["day_of_week"] >= 5).astype(int)
    clean_df["is_workhour"] = (
        (clean_df["is_weekend"] == 0) & (clean_df["hour"] >= 8) & (clean_df["hour"] < 18)
    ).astype(int)

    # 9. Dynamic Occupancy estimation (Explicitly labeled DERIVED)
    clean_df["occupancy_proxy"] = (
        (clean_df["light_lux"] > 30) | (clean_df["plug_kw"] > 2.0)
    ).astype(int)
    n_max_occ = int(round(NOMINAL_ROOM_AREA_M2 / 6.0))  # 8 persons max for 50 m²
    heat_ratio = (clean_df["internal_heat_kw"] / P95_TRAIN_ZONE_INTERNAL_HEAT_KW).clip(lower=0.3, upper=1.0)
    clean_df["occupancy_est"] = np.round(n_max_occ * clean_df["occupancy_proxy"] * heat_ratio).astype(int)

    # 10. Weather Scenario Profile (Explicitly labeled SIMULATED_ASSUMED)
    # Synthetic Bangkok Climatology (SIMULATED_ASSUMED, tham số hiệu chuẩn trên train): mean temp 27.5°C, mean RH 68.0%
    solar_phase = 2 * np.pi * (clean_df["hour"] - 14) / 24.0
    base_ambient_temp = 27.5 + 5.5 * np.cos(solar_phase)
    base_ambient_rh = 68.0 - 15.0 * np.cos(solar_phase)

    # Deterministic daily weather offsets (hash of date string guarantees reproducible daily variations)
    dates_str = clean_df["datetime"].dt.strftime("%Y-%m-%d")
    unique_dates = dates_str.unique()
    temp_offset_map = {}
    rh_offset_map = {}
    for d_str in unique_dates:
        h = int(hashlib.md5(d_str.encode("utf-8")).hexdigest()[:8], 16)
        day_rng = np.random.default_rng(h)
        temp_offset_map[d_str] = float(day_rng.normal(0.0, 1.5))
        rh_offset_map[d_str] = float(day_rng.normal(0.0, 5.0))

    temp_offsets = dates_str.map(temp_offset_map).values
    rh_offsets = dates_str.map(rh_offset_map).values

    clean_df["ambient_temp_est"] = np.clip(base_ambient_temp + temp_offsets, 15.0, 42.0)
    clean_df["ambient_rh_est"] = np.clip(base_ambient_rh + rh_offsets, 20.0, 95.0)
    clean_df["ambient_co2_est"] = 415.0
    clean_df["ambient_pm25_est"] = 25.0

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
        if "temp_is_interpolated" in df.columns and "occupancy_est" in df.columns and "ambient_co2_est" in df.columns:
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
    train_ratio: float = 0.60,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    buffer_hours: int = 0
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Strict sequential temporal split (NO shuffling, NO data leakage).
    Supports:
    - 2-way split: (train_df, test_df) when val_ratio is None.
    - 3-way split: (train_df, val_df, test_df) with optional non-overlapping buffer_hours.
    """
    n_total = len(df)

    if val_ratio is None:
        # 2-way split (backward compatible)
        n_train = int(n_total * train_ratio)
        df_train = df.iloc[:n_train].copy().reset_index(drop=True)
        df_test = df.iloc[n_train:].copy().reset_index(drop=True)

        df_train.attrs = dict(df.attrs)
        df_test.attrs = dict(df.attrs)
        df_train.attrs["split_type"] = "TRAIN"
        df_test.attrs["split_type"] = "TEST"

        assert df_train["datetime"].max() < df_test["datetime"].min(), "Temporal leakage detected in 2-way split!"
        return df_train, df_test

    # 3-way split with temporal buffer
    eff_test_ratio = test_ratio if test_ratio is not None else (1.0 - train_ratio - val_ratio)
    assert train_ratio > 0 and val_ratio > 0 and eff_test_ratio >= 0, "Ratios must be positive"

    n_train = int(n_total * train_ratio)
    df_train = df.iloc[:n_train].copy().reset_index(drop=True)

    val_start = n_train + buffer_hours
    n_val = int(n_total * val_ratio)
    val_end = min(n_total, val_start + n_val)
    df_val = df.iloc[val_start:val_end].copy().reset_index(drop=True)

    test_start = val_end + buffer_hours
    df_test = df.iloc[test_start:].copy().reset_index(drop=True)

    for split_df, split_name in [(df_train, "TRAIN"), (df_val, "VAL"), (df_test, "TEST")]:
        split_df.attrs = dict(df.attrs)
        split_df.attrs["split_type"] = split_name

    # Strict leakage and chronological order assertions
    assert df_train["datetime"].max() < df_val["datetime"].min(), "Temporal overlap between train and val!"
    assert df_val["datetime"].max() < df_test["datetime"].min(), "Temporal overlap between val and test!"

    if buffer_hours > 0:
        gap_train_val = (df_val["datetime"].min() - df_train["datetime"].max()).total_seconds() / 3600.0
        gap_val_test = (df_test["datetime"].min() - df_val["datetime"].max()).total_seconds() / 3600.0
        assert gap_train_val >= buffer_hours, f"Train-Val buffer {gap_train_val:.1f}h < {buffer_hours}h"
        assert gap_val_test >= buffer_hours, f"Val-Test buffer {gap_val_test:.1f}h < {buffer_hours}h"

    return df_train, df_val, df_test


def compute_train_normalization_stats(
    train_df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute normalization statistics (mean, std, min, max) STRICTLY from Train data.
    Never exposes test or validation distributions.
    """
    target_features = features or ["temp_c", "humidity_rh", "internal_heat_kw", "ambient_temp_est", "ambient_rh_est"]
    stats = {}
    for col in target_features:
        if col in train_df.columns:
            s = train_df[col].dropna()
            mean = float(s.mean())
            std = float(s.std()) if float(s.std()) > 1e-6 else 1.0
            stats[col] = {
                "mean": mean,
                "std": std,
                "min": float(s.min()),
                "max": float(s.max())
            }
    return stats


class TemporalFeatureScaler:
    """
    Feature scaler fitted strictly on training split to guarantee zero future/test leakage.
    """
    def __init__(self, features: Optional[List[str]] = None):
        self.features = features
        self.stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "TemporalFeatureScaler":
        self.stats = compute_train_normalization_stats(train_df, self.features)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("TemporalFeatureScaler must be fitted on train_df before transform!")
        out_df = df.copy()
        for col, s in self.stats.items():
            if col in out_df.columns:
                out_df[f"{col}_zscore"] = (out_df[col] - s["mean"]) / s["std"]
                rng = s["max"] - s["min"] if (s["max"] - s["min"]) > 1e-6 else 1.0
                out_df[f"{col}_minmax"] = (out_df[col] - s["min"]) / rng
        return out_df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


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
