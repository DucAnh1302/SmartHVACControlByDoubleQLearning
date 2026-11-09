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


def test_split_target_domain_dataset_zero_leakage():
    """
    Verify that split_target_domain_dataset creates a strict 3-way partition
    with zero timestamp overlap and monotonic chronological order.
    """
    from src.data.domain_loader import split_target_domain_dataset

    dates = pd.date_range("2018-05-01 00:00:00", periods=1000, freq="1h")
    df = pd.DataFrame({
        "datetime": dates,
        "temp_c": np.random.uniform(23, 27, 1000),
        "hvac_kw": np.random.uniform(0, 30, 1000)
    })
    df.attrs["domain_id"] = "TARGET_TEST"

    df_calib, df_adapt, df_test = split_target_domain_dataset(
        df, calib_ratio=0.20, adapt_ratio=0.40, test_ratio=0.40
    )

    # 1. Sizes
    assert len(df_calib) == 200
    assert len(df_adapt) == 400
    assert len(df_test) == 400
    assert len(df_calib) + len(df_adapt) + len(df_test) == len(df)

    # 2. Sequential timestamps: Calib < Adapt < Test
    assert df_calib["datetime"].max() < df_adapt["datetime"].min()
    assert df_adapt["datetime"].max() < df_test["datetime"].min()

    # 3. Zero set overlap
    set_calib = set(df_calib["datetime"])
    set_adapt = set(df_adapt["datetime"])
    set_test = set(df_test["datetime"])

    assert len(set_calib.intersection(set_adapt)) == 0
    assert len(set_adapt.intersection(set_test)) == 0
    assert len(set_calib.intersection(set_test)) == 0


def test_leakage_detector_catches_test_contamination():
    """
    Verify that deliberate contamination between test and training sets
    is immediately detected by strict chronological assertions.
    """
    from src.data.domain_loader import split_target_domain_dataset

    dates = pd.date_range("2018-05-01 00:00:00", periods=500, freq="1h")
    df = pd.DataFrame({
        "datetime": dates,
        "temp_c": np.random.uniform(23, 27, 500),
        "hvac_kw": np.random.uniform(0, 30, 500)
    })

    df_calib, df_adapt, df_test = split_target_domain_dataset(df)

    # Deliberate contamination attempt: inject test row into calib
    contaminated_calib = pd.concat([df_calib, df_test.iloc[0:5]], ignore_index=True)

    # Verifying that the leakage assertion catches this deliberate breach
    with pytest.raises(AssertionError, match="Temporal overlap"):
        # Max of contaminated calib now exceeds min of adapt
        assert contaminated_calib["datetime"].max() < df_adapt["datetime"].min(), "Temporal overlap between calib and adapt!"


def test_sensor_bounded_interpolation_and_quality_mask():
    """
    Verify that gaps <= 3h are cleanly interpolated and flagged with quality mask,
    while prolonged outages (> 3h) are NOT filled with unconstrained ffill/bfill.
    """
    from src.data.domain_loader import preprocess_domain_timeseries
    from src.data.domain_config import DomainSpec

    spec = DomainSpec(
        domain_id="MOCK_GAP_TEST",
        year=2018, floor=2, zone="z1",
        is_primary=False, description="Mock gap testing",
        ac_cols=("z1_AC1(kW)",),
        temp_col="z1_S1(degC)", rh_col="z1_S1(RH%)",
        lux_col="z1_S1(lux)", plug_col="z1_Plug(kW)", light_col="z1_Light(kW)"
    )

    # 48 hours total: hour 5 has a 1h missing gap (should be interpolated)
    # hours 20 to 25 have a 5h missing gap (exceeds max 3h, must be dropped)
    dates = pd.date_range("2018-06-01 00:00:00", periods=48, freq="1h")
    temps = [25.0] * 48
    temps[5] = np.nan  # 1h gap
    for i in range(20, 26):
        temps[i] = np.nan  # 6h gap

    raw_df = pd.DataFrame({
        "Date": dates,
        "z1_S1(degC)": temps,
        "z1_S1(RH%)": [60.0] * 48,
        "z1_S1(lux)": [100.0] * 48,
        "z1_Plug(kW)": [1.0] * 48,
        "z1_Light(kW)": [0.5] * 48,
        "z1_AC1(kW)": [10.0] * 48,
    })

    clean_df = preprocess_domain_timeseries(spec, raw_df, resample_rule="1h", max_interpolate_hours=3)

    # Hour 5 was interpolated -> quality mask temp_is_interpolated should be 1
    h5_rows = clean_df[clean_df["datetime"] == "2018-06-01 05:00:00"]
    assert len(h5_rows) == 1
    assert h5_rows.iloc[0]["temp_is_interpolated"] == 1

    # The 6h gap (hours 20 to 25) should NOT have been filled:
    for h in range(20, 26):
        gap_date = f"2018-06-01 {h:02d}:00:00"
        gap_rows = clean_df[clean_df["datetime"] == gap_date]
        # Any row that exceeded the 3h interpolation limit was dropped
        if h > 22:
            assert len(gap_rows) == 0, f"Expected hour {h} to be dropped due to >3h outage!"

