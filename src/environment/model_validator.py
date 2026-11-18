"""
Comprehensive Transition Model Validation and Benchmarking Suite.
Evaluates Persistence Baseline, Random Forest, and Grey-Box Model on unseen test data,
multi-step rollouts, error slices, and strict physical consistency tests.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.environment.transition_base import BaseTransitionModel
from src.environment.persistence_model import PersistenceTransitionModel
from src.environment.rf_transition_model import RandomForestTransitionModel
from src.environment.greybox_model import GreyBoxThermalModel


def evaluate_single_step(
    model: BaseTransitionModel,
    df_test: pd.DataFrame
) -> Dict[str, float]:
    """Evaluate 1-step prediction metrics on held-out test split."""
    n = len(df_test) - 1
    if n <= 0:
        raise ValueError("Test dataframe has insufficient rows!")

    df_in = df_test.iloc[:-1].copy()
    y_true_temp = df_test["temp_c"].iloc[1:].values
    y_true_rh = df_test["humidity_rh"].iloc[1:].values

    y_pred_temp, y_pred_rh = model.predict_batch(df_in)

    mae_t = float(mean_absolute_error(y_true_temp, y_pred_temp))
    rmse_t = float(np.sqrt(mean_squared_error(y_true_temp, y_pred_temp)))
    r2_t = float(r2_score(y_true_temp, y_pred_temp))

    mae_rh = float(mean_absolute_error(y_true_rh, y_pred_rh))
    rmse_rh = float(np.sqrt(mean_squared_error(y_true_rh, y_pred_rh)))
    r2_rh = float(r2_score(y_true_rh, y_pred_rh))

    return {
        "temp_mae": round(mae_t, 4),
        "temp_rmse": round(rmse_t, 4),
        "temp_r2": round(r2_t, 4),
        "rh_mae": round(mae_rh, 4),
        "rh_rmse": round(rmse_rh, 4),
        "rh_r2": round(r2_rh, 4)
    }


def evaluate_multistep_rollout(
    model: BaseTransitionModel,
    df_test: pd.DataFrame,
    horizon: int = 24,
    n_windows: int = 15
) -> Dict[str, float]:
    """
    Evaluate multi-step autoregressive rollout error over non-overlapping windows.
    Tests how accumulated prediction errors compound over time.
    """
    max_start = len(df_test) - horizon - 1
    if max_start <= 0:
        return {"rollout_mae_24h": -1.0, "rollout_rmse_24h": -1.0}

    step_gap = max(1, max_start // n_windows)
    start_indices = list(range(0, max_start, step_gap))[:n_windows]

    all_errors_t = []
    all_errors_rh = []

    for start_idx in start_indices:
        sub_df = df_test.iloc[start_idx : start_idx + horizon].copy().reset_index(drop=True)
        init_t = float(sub_df["temp_c"].iloc[0])
        init_rh = float(sub_df["humidity_rh"].iloc[0])

        hvac_powers = sub_df["hvac_kw"].values
        pred_t, pred_rh = model.rollout(
            initial_temp=init_t,
            initial_rh=init_rh,
            df_trajectory=sub_df,
            hvac_powers_kw=hvac_powers
        )

        true_t = sub_df["temp_c"].values
        true_rh = sub_df["humidity_rh"].values

        all_errors_t.extend(np.abs(true_t - pred_t))
        all_errors_rh.extend(np.abs(true_rh - pred_rh))

    return {
        f"rollout_mae_{horizon}h": round(float(np.mean(all_errors_t)), 4),
        f"rollout_rmse_{horizon}h": round(float(np.sqrt(np.mean(np.array(all_errors_t)**2))), 4),
        f"rollout_rh_mae_{horizon}h": round(float(np.mean(all_errors_rh)), 4)
    }


def run_physical_consistency_checks(model: BaseTransitionModel) -> Dict[str, Any]:
    """
    Verify fundamental laws of building thermodynamics:
    1. Cooling Direction: higher HVAC power must produce lower or equal temperature.
    2. Heating Direction: higher internal heat must produce higher or equal temperature.
    3. Outdoor Thermal Tracking: higher outdoor temp must raise or maintain indoor temp.
    4. Boundedness: extreme inputs must not produce diverging numerical runaway.
    """
    curr_t = 25.0
    curr_rh = 60.0
    amb_t = 30.0
    amb_rh = 75.0
    hour = 12

    # 1. Cooling monotonicity
    t_off, _ = model.predict_step(curr_t, curr_rh, 0.0, 2.0, amb_t, amb_rh, hour)
    t_low, _ = model.predict_step(curr_t, curr_rh, 10.0, 2.0, amb_t, amb_rh, hour)
    t_med, _ = model.predict_step(curr_t, curr_rh, 20.0, 2.0, amb_t, amb_rh, hour)
    t_high, _ = model.predict_step(curr_t, curr_rh, 35.0, 2.0, amb_t, amb_rh, hour)

    cooling_monotonic = (t_off >= t_low - 0.05) and (t_low >= t_med - 0.05) and (t_med >= t_high - 0.05)

    # 2. Internal heat gain monotonicity
    t_zero_heat, _ = model.predict_step(curr_t, curr_rh, 10.0, 0.0, amb_t, amb_rh, hour)
    t_high_heat, _ = model.predict_step(curr_t, curr_rh, 10.0, 8.0, amb_t, amb_rh, hour)
    heat_monotonic = (t_high_heat >= t_zero_heat - 0.05)

    # 3. Ambient heat transfer monotonicity
    t_cold_amb, _ = model.predict_step(curr_t, curr_rh, 10.0, 2.0, 20.0, amb_rh, hour)
    t_hot_amb, _ = model.predict_step(curr_t, curr_rh, 10.0, 2.0, 36.0, amb_rh, hour)
    ambient_monotonic = (t_hot_amb >= t_cold_amb - 0.05)

    # 4. Extreme input safety bounding
    t_extreme, rh_extreme = model.predict_step(38.0, 95.0, 0.0, 15.0, 45.0, 95.0, hour)
    is_bounded = (15.0 <= t_extreme <= 42.0) and (15.0 <= rh_extreme <= 100.0)

    all_passed = cooling_monotonic and heat_monotonic and ambient_monotonic and is_bounded

    return {
        "cooling_monotonic": bool(cooling_monotonic),
        "heat_gain_monotonic": bool(heat_monotonic),
        "ambient_transfer_monotonic": bool(ambient_monotonic),
        "extreme_bounded": bool(is_bounded),
        "physical_consistency_passed": bool(all_passed),
        "cooling_trajectory_samples": [round(t, 2) for t in [t_off, t_low, t_med, t_high]]
    }


def benchmark_all_transition_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_unseen_domain: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run full benchmark across Persistence, Random Forest, and Grey-Box models.
    """
    models: List[BaseTransitionModel] = [
        PersistenceTransitionModel(),
        RandomForestTransitionModel().fit(df_train),
        GreyBoxThermalModel().fit(df_train)
    ]

    benchmark_rows = []
    full_reports = {}

    for m in models:
        # 1-step test metrics
        single_step = evaluate_single_step(m, df_test)
        # 24h rollout metrics
        rollout_24h = evaluate_multistep_rollout(m, df_test, horizon=24, n_windows=10)
        # 6h rollout metrics
        rollout_6h = evaluate_multistep_rollout(m, df_test, horizon=6, n_windows=10)
        # Physics checks
        physics = run_physical_consistency_checks(m)

        # Cross-domain zero-shot evaluation if unseen domain provided
        cross_domain_metrics = {}
        if df_unseen_domain is not None:
            cd_single = evaluate_single_step(m, df_unseen_domain)
            cross_domain_metrics = {
                "cross_domain_temp_mae": cd_single["temp_mae"],
                "cross_domain_temp_rmse": cd_single["temp_rmse"]
            }

        row = {
            "Model": m.name,
            "1-Step Temp MAE (°C)": single_step["temp_mae"],
            "1-Step Temp RMSE (°C)": single_step["temp_rmse"],
            "1-Step Temp R²": single_step["temp_r2"],
            "6-Hour Rollout MAE (°C)": rollout_6h["rollout_mae_6h"],
            "24-Hour Rollout MAE (°C)": rollout_24h["rollout_mae_24h"],
            "Physics Consistency": "PASS" if physics["physical_consistency_passed"] else "FAIL"
        }
        if cross_domain_metrics:
            row["Cross-Domain MAE (°C)"] = cross_domain_metrics["cross_domain_temp_mae"]

        benchmark_rows.append(row)
        full_reports[m.name] = {
            "single_step": single_step,
            "rollout_6h": rollout_6h,
            "rollout_24h": rollout_24h,
            "physics_checks": physics,
            "cross_domain": cross_domain_metrics
        }

    df_summary = pd.DataFrame(benchmark_rows)
    return df_summary, full_reports
