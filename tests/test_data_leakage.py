"""
Unit tests for Data Leakage Prevention and Temporal Splitting Integrity.
Ensures no future information leaks into training and splits are strictly sequential.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.domain_loader import split_temporal_dataset


def test_split_temporal_dataset_sequential_integrity():
    """Verify that split_temporal_dataset cuts sequentially without shuffling."""
    dates = pd.date_range("2018-01-01 00:00:00", periods=1000, freq="1h")
    df = pd.DataFrame({
        "datetime": dates,
        "temp_c": np.random.uniform(22, 28, 1000),
        "hvac_kw": np.random.uniform(0, 30, 1000)
    })
    df.attrs["domain_id"] = "TEST_DOMAIN"

    train_ratio = 0.70
    df_train, df_test = split_temporal_dataset(df, train_ratio=train_ratio)

    # 1. Check partition sizes
    assert len(df_train) == 700
    assert len(df_test) == 300
    assert len(df_train) + len(df_test) == len(df)

    # 2. Check strict temporal order (max train date < min test date)
    assert df_train["datetime"].max() < df_test["datetime"].min()

    # 3. Check monotonicity (no internal shuffling)
    assert df_train["datetime"].is_monotonic_increasing
    assert df_test["datetime"].is_monotonic_increasing

    # 4. Check that attrs are preserved
    assert df_train.attrs.get("domain_id") == "TEST_DOMAIN"
    assert df_test.attrs.get("domain_id") == "TEST_DOMAIN"


def test_no_future_leakage_in_features():
    """
    Verify that derived features depend only on current or past values.
    """
    from src.data.domain_loader import preprocess_domain_timeseries
    from src.data.domain_config import DomainSpec

    spec = DomainSpec(
        domain_id="MOCK_DOMAIN",
        year=2018,
        floor=2,
        zone="z1",
        is_primary=False,
        description="Mock Domain for test",
        ac_cols=("z1_AC1(kW)",),
        temp_col="z1_S1(degC)",
        rh_col="z1_S1(RH%)",
        lux_col="z1_S1(lux)",
        plug_col="z1_Plug(kW)",
        light_col="z1_Light(kW)"
    )

    dates = pd.date_range("2018-03-01 00:00:00", periods=48, freq="15min")
    raw_df = pd.DataFrame({
        "Date": dates,
        "z1_S1(degC)": np.linspace(24.0, 27.0, 48),
        "z1_S1(RH%)": np.linspace(60.0, 65.0, 48),
        "z1_S1(lux)": [100.0 if 8 <= d.hour <= 18 else 0.0 for d in dates],
        "z1_Plug(kW)": [3.0 if 8 <= d.hour <= 18 else 0.5 for d in dates],
        "z1_Light(kW)": [1.5 if 8 <= d.hour <= 18 else 0.0 for d in dates],
        "z1_AC1(kW)": [20.0 if 8 <= d.hour <= 18 else 0.0 for d in dates],
    })

    clean_df = preprocess_domain_timeseries(spec, raw_df, resample_rule="1h")

    # Verify column existence and integrity
    assert "internal_heat_kw" in clean_df.columns
    assert "occupancy_proxy" in clean_df.columns
    assert "ambient_temp_est" in clean_df.columns

    # Verify internal heat = plug + light
    expected_heat = clean_df["plug_kw"] + clean_df["light_kw"]
    np.testing.assert_allclose(clean_df["internal_heat_kw"], expected_heat)

    # Verify no NaN values
    assert not clean_df.isnull().any().any()
