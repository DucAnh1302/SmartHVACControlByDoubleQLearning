"""
Unit tests for Domain Transfer, Cross-Domain Benchmarking, and Evaluation Protocol.
Verifies domain specifications, calibration loading, and evaluation pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.domain_config import get_domain_spec, DOMAINS
from src.data.calibration import EquipmentStageCalibration, EquipmentCalibrator
from src.environment.target_env import TargetHVACEnv
from src.environment.greybox_model import GreyBoxThermalModel
from src.agent.target_double_q_agent import TargetDoubleQLearningAgent
from src.evaluation.benchmark_suite import evaluate_policy_on_env


def test_domain_specs_consistency():
    """Verify all 4 core domains are defined and valid."""
    required_domains = ["2018_F2_Z1", "2019_F2_Z1", "2018_F2_Z2", "2018_F3_Z1"]
    for d_id in required_domains:
        assert d_id in DOMAINS
        spec = get_domain_spec(d_id)
        assert spec.year in [2018, 2019]
        assert spec.floor in [2, 3]
        assert spec.temp_col
        assert len(spec.ac_cols) >= 1


def test_mock_environment_evaluation():
    """Verify evaluate_policy_on_env executes cleanly on mock timeseries."""
    dates = pd.date_range("2018-05-01 00:00:00", periods=200, freq="1h")
    mock_df = pd.DataFrame({
        "datetime": dates,
        "temp_c": np.random.uniform(23.5, 26.5, 200),
        "humidity_rh": np.random.uniform(55.0, 70.0, 200),
        "hvac_kw": np.random.uniform(0.0, 30.0, 200),
        "internal_heat_kw": np.random.uniform(0.5, 3.0, 200),
        "ambient_temp_est": np.random.uniform(28.0, 34.0, 200),
        "ambient_rh_est": np.random.uniform(60.0, 80.0, 200),
        "hour": [d.hour for d in dates],
        "is_weekend": [1 if d.day_of_week >= 5 else 0 for d in dates],
        "is_workhour": [1 if d.day_of_week < 5 and 8 <= d.hour < 18 else 0 for d in dates],
        "occupancy_proxy": [1 if 8 <= d.hour < 18 else 0 for d in dates],
    })

    calib = EquipmentStageCalibration(
        domain_id="MOCK_DOMAIN",
        year=2018,
        floor=2,
        zone="z1",
        off_power_kw=0.0,
        low_power_kw=12.5,
        med_power_kw=21.6,
        high_power_kw=31.8,
        off_frequency=0.25,
        total_samples=1000,
        active_samples=750,
        cluster_centers=[12.5, 21.6, 31.8],
        cluster_boundaries=[17.0, 26.5],
        inertia=120.5
    )

    env = TargetHVACEnv(
        data_df=mock_df,
        calibration=calib,
        episode_length=24
    )

    agent = TargetDoubleQLearningAgent(n_states=600, n_actions=5, seed=42)

    # Run quick evaluation of 2 episodes
    results = evaluate_policy_on_env(
        agent=agent,
        env=env,
        n_episodes=2,
        seed=42,
        use_safety_gateway=True
    )

    # Assert required output metric keys
    assert "mean_episode_reward" in results
    assert "comfort_violation_pct" in results
    assert "severe_violation_pct" in results
    assert "avg_energy_kwh_per_day" in results
    assert "action_distribution_pct" in results
    assert "safety_override_rate_pct" in results
