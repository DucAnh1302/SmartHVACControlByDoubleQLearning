"""
Unit tests for Phase 2: Data Pipeline, Temporal 60/20/20 Buffer Split,
and Strict Train-Only Normalization.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.domain_loader import (
    get_domain_dataset,
    split_temporal_dataset,
    compute_train_normalization_stats,
    TemporalFeatureScaler,
    get_domain_spec,
    load_domain_raw_dataframe,
    preprocess_domain_timeseries
)


def test_phase2_split_temporal_60_20_20_with_24h_buffer():
    """Verify 3-way temporal partition with 24h buffer has zero overlap."""
    df = get_domain_dataset("2018_F2_Z1")
    train_df, val_df, test_df = split_temporal_dataset(
        df,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        buffer_hours=24
    )

    # 1. Assert chronological progression
    assert train_df["datetime"].max() < val_df["datetime"].min()
    assert val_df["datetime"].max() < test_df["datetime"].min()

    # 2. Assert minimum 24-hour buffer gap between partitions
    gap_train_val = (val_df["datetime"].min() - train_df["datetime"].max()).total_seconds() / 3600.0
    gap_val_test = (test_df["datetime"].min() - val_df["datetime"].max()).total_seconds() / 3600.0
    assert gap_train_val >= 24.0, f"Train-Val buffer was {gap_train_val}h, expected >= 24h"
    assert gap_val_test >= 24.0, f"Val-Test buffer was {gap_val_test}h, expected >= 24h"

    # 3. Assert partition monotonicity
    assert train_df["datetime"].is_monotonic_increasing
    assert val_df["datetime"].is_monotonic_increasing
    assert test_df["datetime"].is_monotonic_increasing


def test_phase2_normalization_stats_strictly_from_train():
    """Verify that normalization statistics are derived solely from train split."""
    df = get_domain_dataset("2018_F2_Z1")
    train_df, val_df, test_df = split_temporal_dataset(
        df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, buffer_hours=24
    )

    scaler = TemporalFeatureScaler(features=["temp_c", "humidity_rh"])
    scaler.fit(train_df)

    # Verify scaler stats equal train_df stats
    assert np.isclose(scaler.stats["temp_c"]["mean"], train_df["temp_c"].mean(), atol=1e-5)
    assert np.isclose(scaler.stats["temp_c"]["std"], train_df["temp_c"].std(), atol=1e-5)

    # Verify scaler stats are NOT influenced by test_df
    assert not np.isclose(scaler.stats["temp_c"]["mean"], test_df["temp_c"].mean(), atol=1e-2)

    # Transform test_df and verify transformed column uses train mean
    test_transformed = scaler.transform(test_df)
    assert "temp_c_zscore" in test_transformed.columns
    expected_zscore = (test_df["temp_c"] - scaler.stats["temp_c"]["mean"]) / scaler.stats["temp_c"]["std"]
    assert np.allclose(test_transformed["temp_c_zscore"], expected_zscore, atol=1e-5)


def test_phase2_raw_vs_processed_reproducibility():
    """Verify that raw ingestion matches cached processed data within 1%."""
    spec = get_domain_spec("2018_F2_Z1")
    df_raw = load_domain_raw_dataframe(spec)
    fresh_df = preprocess_domain_timeseries(spec, df_raw)
    cached_df = get_domain_dataset("2018_F2_Z1")

    # Assert row count match
    assert len(fresh_df) == len(cached_df)

    # Assert mean values within 1.0%
    diff_temp = abs(fresh_df["temp_c"].mean() - cached_df["temp_c"].mean()) / cached_df["temp_c"].mean()
    diff_rh = abs(fresh_df["humidity_rh"].mean() - cached_df["humidity_rh"].mean()) / cached_df["humidity_rh"].mean()
    assert diff_temp < 0.01, f"Temperature mean diff {diff_temp*100:.3f}% >= 1.0%"
    assert diff_rh < 0.01, f"Humidity mean diff {diff_rh*100:.3f}% >= 1.0%"
