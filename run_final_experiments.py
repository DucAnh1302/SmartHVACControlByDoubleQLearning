"""
Run Final Experiments - Comprehensive Benchmark Pipeline (Phase 3, Protocol v2).
Executes:
1. Hyperparameter grid search on validation split (27 configs x 3 seeds x 2 algorithms)
   using a target-fraction exponential epsilon decay reaching epsilon_min=0.02 cleanly.
2. Model training across 10 random seeds (0..9) on train split (parallelized via 8 workers).
3. Comparative evaluation of 6 controllers on Test-A, Test-B, and Test-C.
4. Statistical hypothesis testing (Paired t-test, Wilcoxon, Cohen's d, Bootstrap 95% CI,
   Levene variance test, and Holm-Bonferroni baseline comparisons).
5. Maximization bias verification (Toy MDP + HVAC Monte Carlo bias).
6. Full real-data artifact exports (manifest.json, seed_level_results.csv, benchmark_results.csv,
   training_history.csv, action_distribution.csv, trajectory_24h.csv, maxbias_toy_curves.csv,
   test_c_summary.csv).
"""

import os
import sys
import json
import time
import pickle
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set project root in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.environment.multi_device_env import MultiDeviceHVACEnv, ROOM_PRESETS, MULTI_DEVICE_ACTION_TEMPLATES
from src.reward.multi_device_reward import MultiDeviceRewardFunction
from src.safety.multi_device_safety_gateway import MultiDeviceSafetyGateway
from src.agent.multi_device_double_q_agent import MultiDeviceDoubleQAgent, SingleQMultiDeviceAgent
from src.agent.multi_device_baselines import (
    FixedSetpointMultiDevice, ConventionalThermostatMultiDevice,
    RuleBasedSmartHVACMultiDevice, RandomMultiDevicePolicy
)
from src.data.domain_loader import get_domain_dataset, split_temporal_dataset
from experiments.maxbias_toy import run_maximization_bias_toy


# ---------------------------------------------------------------------------
# Multiprocessing Global Worker Cache (Windows spawn safe)
# ---------------------------------------------------------------------------
_WORKER_TRAIN_DF: Optional[pd.DataFrame] = None
_WORKER_VAL_DF: Optional[pd.DataFrame] = None


def _init_worker(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> None:
    global _WORKER_TRAIN_DF, _WORKER_VAL_DF
    _WORKER_TRAIN_DF = train_df
    _WORKER_VAL_DF = val_df


# ---------------------------------------------------------------------------
# Evaluation Utility Functions
# ---------------------------------------------------------------------------

def evaluate_episode(
    policy,
    env: MultiDeviceHVACEnv,
    reward_fn: MultiDeviceRewardFunction,
    safety_gateway: Optional[MultiDeviceSafetyGateway],
    start_idx: Optional[int] = None,
    seed: int = 42,
    record_actions: bool = False,
    record_trajectory: bool = False,
    controller_name: str = ""
) -> Dict[str, Any]:
    """Evaluates a single 24-hour episode under greedy control."""
    state, info = env.reset(start_idx=start_idx, seed=seed)
    env._ensure_row_cache()
    if safety_gateway:
        safety_gateway.reset()

    tot_reward = 0.0
    switches = 0
    overrides = 0
    occ_hours = 0
    t_viol = 0
    rh_viol = 0
    co2_viol = 0
    pm25_viol = 0
    last_act = None
    action_counts = [0] * 12 if record_actions else None
    trajectory_rows = [] if record_trajectory else None

    done = False
    while not done:
        raw_action = policy.select_action(state, evaluate=True)
        idx = env.data_idx

        if safety_gateway:
            safe_act, reason = safety_gateway.filter_action(
                agent_action=raw_action,
                indoor_temp=env.current_temp,
                indoor_rh=env.current_rh,
                indoor_co2=env.current_co2,
                indoor_pm25=env.current_pm25,
                ambient_temp=env._c_amb_t[idx],
                ambient_rh=env._c_amb_rh[idx],
                ambient_pm25=env._c_amb_pm25[idx],
                hour=env._c_hour[idx],
                step=env.current_step
            )
            safety_viol = (safe_act != raw_action)
            if safety_viol:
                overrides += 1
            act_to_exec = safe_act
        else:
            reason = "SAFE"
            safety_viol = False
            act_to_exec = raw_action

        if action_counts is not None:
            action_counts[act_to_exec] += 1

        if last_act is not None and act_to_exec != last_act:
            switches += 1
        last_act = act_to_exec

        next_state, _, done, trunc, step_info = env.step(act_to_exec)
        r, _ = reward_fn.calculate_reward(
            temp_c=step_info["temp_c"],
            humidity_rh=step_info["humidity_rh"],
            co2_ppm=step_info["co2_ppm"],
            pm25_ug_m3=step_info["pm25_ug_m3"],
            total_power_kw=step_info["powers_kw"]["total_kw"],
            hour=step_info["hour"],
            dayofweek=step_info["dayofweek"],
            is_workhour=step_info.get("is_workhour"),
            action_changed=step_info["action_changed"],
            safety_violation=safety_viol
        )
        tot_reward += r

        if trajectory_rows is not None:
            trajectory_rows.append({
                "controller": controller_name,
                "step": env.current_step,
                "hour": step_info["hour"],
                "dayofweek": step_info["dayofweek"],
                "is_workhour": step_info.get("is_workhour", 0),
                "occupancy_count": step_info.get("occupancy_count", 0),
                "ambient_temp_c": step_info.get("ambient_temp_c", env._c_amb_t[idx]),
                "ambient_rh_pct": step_info.get("ambient_rh_pct", env._c_amb_rh[idx]),
                "ambient_pm25_ug_m3": env._c_amb_pm25[idx],
                "temp_c": step_info["temp_c"],
                "humidity_rh": step_info["humidity_rh"],
                "co2_ppm": step_info["co2_ppm"],
                "pm25_ug_m3": step_info["pm25_ug_m3"],
                "raw_action": raw_action,
                "exec_action": act_to_exec,
                "action_name": MULTI_DEVICE_ACTION_TEMPLATES[act_to_exec].name,
                "safety_override": int(safety_viol),
                "override_reason": reason,
                "total_kw": step_info["powers_kw"]["total_kw"],
                "cumul_kwh": env.cumul_energy["total_kwh"],
                "step_reward": r,
            })

        # Evaluate comfort and IAQ only during occupied hours (Methodological Invariant)
        occ = step_info.get("occupancy_count", 0)
        if occ > 0:
            occ_hours += 1
            t = step_info["temp_c"]
            rh = step_info["humidity_rh"]
            co2 = step_info["co2_ppm"]
            pm = step_info["pm25_ug_m3"]

            if t < 24.0 or t > 26.0:
                t_viol += 1
            if rh < 40.0 or rh > 65.0:
                rh_viol += 1
            if co2 > 1000.0:
                co2_viol += 1
            if pm > 35.0:
                pm25_viol += 1

        state = next_state

    energy_kwh = env.cumul_energy["total_kwh"]

    out = {
        "reward": tot_reward,
        "energy_kwh": energy_kwh,
        "switches": switches,
        "overrides": overrides,
        "total_steps": 24,
        "occ_hours": occ_hours,
        "t_viol": t_viol,
        "rh_viol": rh_viol,
        "co2_viol": co2_viol,
        "pm25_viol": pm25_viol
    }
    if action_counts is not None:
        out["action_counts"] = action_counts
    if trajectory_rows is not None:
        out["trajectory_rows"] = trajectory_rows
    return out


def evaluate_policy_on_scenario(
    policy,
    data_df: Optional[pd.DataFrame],
    scenario_type: str,
    reward_fn: MultiDeviceRewardFunction,
    safety_gateway: Optional[MultiDeviceSafetyGateway],
    n_episodes: int = 10,
    base_seed: int = 100,
    record_actions: bool = False
) -> Dict[str, Any]:
    """Evaluates a policy on a specified dataset or synthetic scenario over multiple episodes."""
    if data_df is not None and len(data_df) > 0:
        env = MultiDeviceHVACEnv(data_df=data_df, episode_length=24, seed=base_seed)
        max_start = max(1, len(data_df) - 26)
        step_stride = max(1, max_start // n_episodes)
        start_indices = [min(i * step_stride, max_start - 1) for i in range(n_episodes)]
    else:
        env = MultiDeviceHVACEnv(scenario_type=scenario_type, episode_length=24, seed=base_seed)
        start_indices = [0 for _ in range(n_episodes)]

    ep_results = []
    agg_actions = [0] * 12 if record_actions else None
    for ep in range(n_episodes):
        res = evaluate_episode(
            policy=policy,
            env=env,
            reward_fn=reward_fn,
            safety_gateway=safety_gateway,
            start_idx=start_indices[ep],
            seed=base_seed + ep,
            record_actions=record_actions
        )
        if agg_actions is not None:
            for a_i, cnt in enumerate(res["action_counts"]):
                agg_actions[a_i] += cnt
        ep_results.append(res)

    tot_energy = sum(r["energy_kwh"] for r in ep_results)
    tot_reward = sum(r["reward"] for r in ep_results)
    tot_switches = sum(r["switches"] for r in ep_results)
    tot_overrides = sum(r["overrides"] for r in ep_results)
    tot_steps = sum(r["total_steps"] for r in ep_results)
    tot_occ = sum(r["occ_hours"] for r in ep_results)

    tot_t_viol = sum(r["t_viol"] for r in ep_results)
    tot_rh_viol = sum(r["rh_viol"] for r in ep_results)
    tot_co2_viol = sum(r["co2_viol"] for r in ep_results)
    tot_pm_viol = sum(r["pm25_viol"] for r in ep_results)

    denom_occ = max(1, tot_occ)

    summary = {
        "energy_kwh_per_day": float(tot_energy / n_episodes),
        "mean_reward": float(tot_reward / n_episodes),
        "comfort_violation_pct": float((tot_t_viol / denom_occ) * 100.0),
        "rh_violation_pct": float((tot_rh_viol / denom_occ) * 100.0),
        "co2_violation_pct": float((tot_co2_viol / denom_occ) * 100.0),
        "pm25_violation_pct": float((tot_pm_viol / denom_occ) * 100.0),
        "switching_per_day": float(tot_switches / n_episodes),
        "override_pct": float((tot_overrides / tot_steps) * 100.0)
    }
    if agg_actions is not None:
        summary["action_counts"] = agg_actions
    return summary


# ---------------------------------------------------------------------------
# Training Worker Functions
# ---------------------------------------------------------------------------

def compute_epsilon_decay(epsilon_start: float, epsilon_min: float, target_decay_frac: float, n_episodes: int) -> float:
    """Computes multiplicative decay factor reaching epsilon_min after target_decay_frac * n_episodes."""
    decay_steps = max(1.0, float(target_decay_frac) * float(n_episodes))
    return float((epsilon_min / epsilon_start) ** (1.0 / decay_steps))


def train_agent_run(
    agent_type: str,
    lr: float,
    epsilon_decay: float,
    n_episodes: int,
    seed: int,
    train_df: pd.DataFrame,
    epsilon_min: float = 0.02,
    record_history: bool = False
) -> Any:
    """Trains a single agent with in-loop Safety Gateway feedback on train_df."""
    env = MultiDeviceHVACEnv(data_df=train_df, episode_length=24, seed=seed)
    env._ensure_row_cache()
    reward_fn = MultiDeviceRewardFunction()
    safety_gateway = MultiDeviceSafetyGateway()

    if agent_type == "dql":
        agent = MultiDeviceDoubleQAgent(
            n_states=720, n_actions=12,
            learning_rate=lr, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            seed=seed
        )
    else:
        agent = SingleQMultiDeviceAgent(
            n_states=720, n_actions=12,
            learning_rate=lr, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
            seed=seed
        )

    history = [] if record_history else None

    for ep in range(1, n_episodes + 1):
        state, info = env.reset(seed=seed + ep)
        safety_gateway.reset()
        done = False
        ep_reward = 0.0
        ep_overrides = 0
        ep_occ_hours = 0
        ep_t_viol = 0
        ep_co2_viol = 0

        while not done:
            raw_action = agent.select_action(state, evaluate=False)
            idx = env.data_idx
            safe_act, _ = safety_gateway.filter_action(
                agent_action=raw_action,
                indoor_temp=env.current_temp,
                indoor_rh=env.current_rh,
                indoor_co2=env.current_co2,
                indoor_pm25=env.current_pm25,
                ambient_temp=env._c_amb_t[idx],
                ambient_rh=env._c_amb_rh[idx],
                ambient_pm25=env._c_amb_pm25[idx],
                hour=env._c_hour[idx],
                step=env.current_step
            )
            safety_viol = (safe_act != raw_action)
            if safety_viol:
                ep_overrides += 1
            act_to_exec = safe_act

            next_state, _, done, trunc, step_info = env.step(act_to_exec)
            r, _ = reward_fn.calculate_reward(
                temp_c=step_info["temp_c"],
                humidity_rh=step_info["humidity_rh"],
                co2_ppm=step_info["co2_ppm"],
                pm25_ug_m3=step_info["pm25_ug_m3"],
                total_power_kw=step_info["powers_kw"]["total_kw"],
                hour=step_info["hour"],
                dayofweek=step_info["dayofweek"],
                is_workhour=step_info.get("is_workhour"),
                action_changed=step_info["action_changed"],
                safety_violation=safety_viol
            )
            agent.update(state, act_to_exec, r, next_state, done)
            ep_reward += r
            occ = step_info.get("occupancy_count", 0)
            if occ > 0:
                ep_occ_hours += 1
                if step_info["temp_c"] < 24.0 or step_info["temp_c"] > 26.0:
                    ep_t_viol += 1
                if step_info["co2_ppm"] > 1000.0:
                    ep_co2_viol += 1
            state = next_state

        if history is not None:
            denom_occ = max(1, ep_occ_hours)
            history.append({
                "algorithm": "Double Q-Learning" if agent_type == "dql" else "Single Q-Learning",
                "seed": seed,
                "episode": ep,
                "reward": float(ep_reward),
                "energy_kwh": float(env.cumul_energy["total_kwh"]),
                "epsilon": float(agent.epsilon),
                "t_viol_pct": float(ep_t_viol / denom_occ * 100.0),
                "co2_viol_pct": float(ep_co2_viol / denom_occ * 100.0),
                "overrides": int(ep_overrides),
                "coverage_pct": float((np.sum(agent.visit_counts > 0) / (agent.n_states * agent.n_actions)) * 100.0),
            })

        agent.decay_epsilon()

    if record_history:
        return agent, history
    return agent


def grid_search_worker(task: Tuple) -> Dict[str, Any]:
    """Worker task for multiprocessing grid search."""
    agent_type, lr, target_frac, eps_decay, n_episodes, seed = task
    train_df = _WORKER_TRAIN_DF
    val_df = _WORKER_VAL_DF

    agent = train_agent_run(
        agent_type=agent_type, lr=lr, epsilon_decay=eps_decay,
        n_episodes=n_episodes, seed=seed, train_df=train_df,
        epsilon_min=0.02, record_history=False
    )

    reward_fn = MultiDeviceRewardFunction()
    safety_gateway = MultiDeviceSafetyGateway()
    val_res = evaluate_policy_on_scenario(
        policy=agent,
        data_df=val_df,
        scenario_type="standard",
        reward_fn=reward_fn,
        safety_gateway=safety_gateway,
        n_episodes=8,
        base_seed=seed + 999
    )

    return {
        "agent_type": agent_type,
        "learning_rate": lr,
        "target_decay_frac": target_frac,
        "epsilon_decay": eps_decay,
        "n_episodes": n_episodes,
        "seed": seed,
        "val_mean_reward": val_res["mean_reward"],
        "val_energy_kwh": val_res["energy_kwh_per_day"],
        "val_comfort_violation_pct": val_res["comfort_violation_pct"],
        "val_co2_violation_pct": val_res["co2_violation_pct"],
    }


def final_train_worker(task: Tuple) -> Tuple[str, int, Any, List[Dict[str, Any]]]:
    """Worker task to train final multi-seed agent + log per-episode history."""
    agent_type, lr, eps_decay, n_episodes, seed = task
    agent, history = train_agent_run(
        agent_type=agent_type, lr=lr, epsilon_decay=eps_decay,
        n_episodes=n_episodes, seed=seed, train_df=_WORKER_TRAIN_DF,
        epsilon_min=0.02, record_history=True
    )
    return agent_type, seed, agent, history


# ---------------------------------------------------------------------------
# Monte Carlo Return Bias Measurement for HVAC
# ---------------------------------------------------------------------------

def measure_hvac_mc_bias(
    agent_dql: MultiDeviceDoubleQAgent,
    agent_ql: SingleQMultiDeviceAgent,
    env_eval: MultiDeviceHVACEnv,
    reward_fn: MultiDeviceRewardFunction,
    n_sample_states: int = 20,
    mc_rollouts: int = 50,
    gamma: float = 0.95
) -> Tuple[float, float]:
    """
    Measures empirical maximization bias on the HVAC environment:
    Bias = max_a Q(s0, a) - G_MC(s0)
    where G_MC(s0) is the average discounted return of greedy policy from state s0.
    """
    dql_biases = []
    ql_biases = []

    sampled_indices = np.linspace(10, max(20, len(env_eval.data_df) - 30), n_sample_states, dtype=int)

    for s_idx in sampled_indices:
        # Measure QL
        state_ql, _ = env_eval.reset(start_idx=int(s_idx), seed=42)
        q_vals_ql = agent_ql.q_table[state_ql]
        max_q_ql = float(np.max(q_vals_ql))

        returns_ql = []
        for ro in range(mc_rollouts):
            env_eval.reset(start_idx=int(s_idx), seed=1000 + ro)
            s = state_ql
            g = 0.0
            discount = 1.0
            for _ in range(24):
                a = agent_ql.select_action(s, evaluate=True)
                s_next, _, done, _, info = env_eval.step(a)
                r, _ = reward_fn.calculate_reward(
                    temp_c=info["temp_c"], humidity_rh=info["humidity_rh"],
                    co2_ppm=info["co2_ppm"], pm25_ug_m3=info["pm25_ug_m3"],
                    total_power_kw=info["powers_kw"]["total_kw"],
                    hour=info["hour"], dayofweek=info["dayofweek"],
                    is_workhour=info.get("is_workhour"), action_changed=info["action_changed"]
                )
                g += discount * r
                discount *= gamma
                s = s_next
                if done:
                    break
            returns_ql.append(g)

        g_mc_ql = float(np.mean(returns_ql))
        ql_biases.append(max_q_ql - g_mc_ql)

        # Measure DQL (using combined Q table)
        state_dql, _ = env_eval.reset(start_idx=int(s_idx), seed=42)
        combined_q = (agent_dql.q_table_a + agent_dql.q_table_b) / 2.0
        max_q_dql = float(np.max(combined_q[state_dql]))

        returns_dql = []
        for ro in range(mc_rollouts):
            env_eval.reset(start_idx=int(s_idx), seed=1000 + ro)
            s = state_dql
            g = 0.0
            discount = 1.0
            for _ in range(24):
                a = agent_dql.select_action(s, evaluate=True)
                s_next, _, done, _, info = env_eval.step(a)
                r, _ = reward_fn.calculate_reward(
                    temp_c=info["temp_c"], humidity_rh=info["humidity_rh"],
                    co2_ppm=info["co2_ppm"], pm25_ug_m3=info["pm25_ug_m3"],
                    total_power_kw=info["powers_kw"]["total_kw"],
                    hour=info["hour"], dayofweek=info["dayofweek"],
                    is_workhour=info.get("is_workhour"), action_changed=info["action_changed"]
                )
                g += discount * r
                discount *= gamma
                s = s_next
                if done:
                    break
            returns_dql.append(g)

        g_mc_dql = float(np.mean(returns_dql))
        dql_biases.append(max_q_dql - g_mc_dql)

    return float(np.mean(ql_biases)), float(np.mean(dql_biases))


def holm_bonferroni(p_vals: Dict[str, float]) -> Dict[str, float]:
    """Applies Holm-Bonferroni step-down correction to a dict of {name: p_value}."""
    sorted_items = sorted(p_vals.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    adjusted = {}
    running_max = 0.0
    for rank, (k, p) in enumerate(sorted_items):
        adj = min(1.0, p * (m - rank))
        running_max = max(running_max, adj)
        adjusted[k] = float(running_max)
    return adjusted


def _find_representative_weekday_start(df: pd.DataFrame) -> int:
    """Finds a representative weekday 24h window in df starting near midnight 00:00."""
    df_reset = df.reset_index(drop=True)
    max_start = max(1, len(df_reset) - 26)
    for idx in range(0, max_start):
        window = df_reset.iloc[idx:idx + 24]
        if int(window.iloc[0]["hour"]) == 0 and int(window.iloc[0]["dayofweek"]) < 5:
            if window["is_workhour"].sum() >= 8:
                return int(idx)
    return 0


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("  SMART HVAC CONTROL: PHASE 3 COMPREHENSIVE BENCHMARK (PROTOCOL v2)")
    print("=" * 80)
    start_time_all = time.time()

    # 1. Check git tag protocol-locked / protocol-locked-v2
    res_tags = subprocess.run(["git", "tag", "-l"], capture_output=True, text=True, cwd=str(REPO_ROOT))
    tag_list = res_tags.stdout.splitlines()
    active_tag = "protocol-locked-v2" if "protocol-locked-v2" in tag_list else (
        "protocol-locked" if "protocol-locked" in tag_list else "unlocked"
    )
    if "protocol-locked" not in tag_list:
        print("[!] WARNING: Git tag 'protocol-locked' not detected! Gating requires this tag.")
    else:
        print(f"  Active protocol tag: {active_tag}")

    # 2. Load Datasets and create 60/20/20 temporal split with 24h buffer
    print("\n[Step 1] Loading Datasets and Creating Temporal Split...")
    df_2018 = get_domain_dataset("2018_F2_Z1")
    train_df, val_df, test_df = split_temporal_dataset(
        df_2018, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, buffer_hours=24
    )
    df_2019 = get_domain_dataset("2019_F2_Z1")

    print(f"  Train Split (2018): {len(train_df)} hours ({train_df['datetime'].min()} to {train_df['datetime'].max()})")
    print(f"  Val Split   (2018): {len(val_df)} hours ({val_df['datetime'].min()} to {val_df['datetime'].max()})")
    print(f"  Test-A      (2018): {len(test_df)} hours ({test_df['datetime'].min()} to {test_df['datetime'].max()})")
    print(f"  Test-B      (2019): {len(df_2019)} hours ({df_2019['datetime'].min()} to {df_2019['datetime'].max()})")

    # 3. Hyperparameter Grid Search on Validation Split (Target-Fraction Schedule)
    print("\n[Step 2] Executing Hyperparameter Grid Search on Validation Split...")
    grid_alphas = [0.05, 0.10, 0.15]
    grid_target_fracs = [0.65, 0.75, 0.85]
    grid_episodes = [1200, 2000, 3000]
    val_seeds = [101, 102, 103]

    tasks = []
    for agent_type in ["dql", "ql"]:
        for lr in grid_alphas:
            for tf in grid_target_fracs:
                for nep in grid_episodes:
                    ed = compute_epsilon_decay(1.0, 0.02, tf, nep)
                    for s in val_seeds:
                        tasks.append((agent_type, lr, tf, ed, nep, s))

    n_workers = min(8, os.cpu_count() or 6)
    print(f"  Total grid tasks: {len(tasks)} (27 configs x 3 seeds x 2 algorithms).")
    print(f"  Dispatching to ProcessPoolExecutor (max_workers={n_workers})...")

    t_grid_start = time.time()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(train_df, val_df)
    ) as executor:
        grid_results = list(executor.map(grid_search_worker, tasks))
    t_grid_end = time.time()
    print(f"  Grid Search completed in {t_grid_end - t_grid_start:.1f}s ({(t_grid_end - t_grid_start)/60:.1f} min).")

    df_grid = pd.DataFrame(grid_results)
    out_dir = REPO_ROOT / "result" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(out_dir / "hyperparam_search.csv", index=False)

    # Aggregate by configuration to select best for DQL and QL
    agg_grid = df_grid.groupby(
        ["agent_type", "learning_rate", "target_decay_frac", "epsilon_decay", "n_episodes"]
    )["val_mean_reward"].mean().reset_index()

    best_dql_row = agg_grid[agg_grid["agent_type"] == "dql"].sort_values("val_mean_reward", ascending=False).iloc[0]
    best_ql_row = agg_grid[agg_grid["agent_type"] == "ql"].sort_values("val_mean_reward", ascending=False).iloc[0]

    best_dql_params = {
        "learning_rate": float(best_dql_row["learning_rate"]),
        "target_decay_frac": float(best_dql_row["target_decay_frac"]),
        "epsilon_decay": float(best_dql_row["epsilon_decay"]),
        "epsilon_min": 0.02,
        "n_episodes": int(best_dql_row["n_episodes"]),
        "val_mean_reward": float(best_dql_row["val_mean_reward"]),
    }
    best_ql_params = {
        "learning_rate": float(best_ql_row["learning_rate"]),
        "target_decay_frac": float(best_ql_row["target_decay_frac"]),
        "epsilon_decay": float(best_ql_row["epsilon_decay"]),
        "epsilon_min": 0.02,
        "n_episodes": int(best_ql_row["n_episodes"]),
        "val_mean_reward": float(best_ql_row["val_mean_reward"]),
    }

    print("\n[Optimal Hyperparameters Selected on Validation]:")
    print(f"  DQL Best: lr={best_dql_params['learning_rate']}, tf={best_dql_params['target_decay_frac']}, "
          f"eps_decay={best_dql_params['epsilon_decay']:.6f}, episodes={best_dql_params['n_episodes']} "
          f"(Val Reward: {best_dql_params['val_mean_reward']:.2f})")
    print(f"  QL Best : lr={best_ql_params['learning_rate']}, tf={best_ql_params['target_decay_frac']}, "
          f"eps_decay={best_ql_params['epsilon_decay']:.6f}, episodes={best_ql_params['n_episodes']} "
          f"(Val Reward: {best_ql_params['val_mean_reward']:.2f})")

    # 4. Final Training Across 10 Seeds (0..9) on Train Split (parallelized)
    print("\n[Step 3] Training Final Models Across 10 Seeds (0..9) on Train Split...")
    eval_seeds = list(range(10))
    train_tasks = []
    for s in eval_seeds:
        train_tasks.append(("dql", best_dql_params["learning_rate"], best_dql_params["epsilon_decay"], best_dql_params["n_episodes"], s))
        train_tasks.append(("ql", best_ql_params["learning_rate"], best_ql_params["epsilon_decay"], best_ql_params["n_episodes"], s))

    t_train_start = time.time()
    dql_agents: Dict[int, MultiDeviceDoubleQAgent] = {}
    ql_agents: Dict[int, SingleQMultiDeviceAgent] = {}
    all_history_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(train_df, None)
    ) as executor:
        for agent_type, seed, agent, hist in executor.map(final_train_worker, train_tasks):
            if agent_type == "dql":
                dql_agents[seed] = agent
            else:
                ql_agents[seed] = agent
            all_history_rows.extend(hist)

    t_train_end = time.time()
    print(f"  Final training (20 models across {n_workers} workers) completed in {t_train_end - t_train_start:.1f}s.")

    df_history = pd.DataFrame(all_history_rows)
    df_history.to_csv(out_dir / "training_history.csv", index=False)
    print(f"  Saved real training history ({len(df_history)} rows) to {out_dir / 'training_history.csv'}")

    # Save checkpoints for seed 0
    model_dir = REPO_ROOT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    dql_agents[0].save(str(model_dir / "final_dql_seed0.pkl"))
    with open(model_dir / "final_ql_seed0.pkl", "wb") as f:
        pickle.dump(ql_agents[0], f)

    # 5. Comparative Evaluation of 6 Controllers on Test Scenarios
    print("\n[Step 4] Running Benchmark Evaluation on Test Scenarios (Test-A, Test-B, Test-C)...")
    reward_fn = MultiDeviceRewardFunction()
    safety_gateway = MultiDeviceSafetyGateway()

    fixed_policy = FixedSetpointMultiDevice()
    thermostat_policy = ConventionalThermostatMultiDevice()
    rule_based_policy = RuleBasedSmartHVACMultiDevice()
    random_policy = RandomMultiDevicePolicy()

    scenarios = {
        "test-A": {"df": test_df, "type": "standard"},
        "test-B": {"df": df_2019, "type": "standard"},
        "test-C_heatwave": {"df": None, "type": "heatwave"},
        "test-C_large_occupancy": {"df": None, "type": "large_occupancy"},
        "test-C_smog_pollution": {"df": None, "type": "smog_pollution"}
    }

    controllers_names = [
        "Double Q-Learning",
        "Single Q-Learning",
        "Fixed Setpoint (25°C Cool)",
        "Conventional Thermostat",
        "Rule-Based Smart HVAC",
        "Random Policy"
    ]

    seed_records = []
    action_totals_testA = {c_name: np.zeros(12, dtype=int) for c_name in controllers_names}

    for s in eval_seeds:
        ctrls = {
            "Double Q-Learning": dql_agents[s],
            "Single Q-Learning": ql_agents[s],
            "Fixed Setpoint (25°C Cool)": fixed_policy,
            "Conventional Thermostat": thermostat_policy,
            "Rule-Based Smart HVAC": rule_based_policy,
            "Random Policy": random_policy
        }

        for scen_key, scen_info in scenarios.items():
            rec_acts = (scen_key == "test-A")
            for c_name, c_inst in ctrls.items():
                res = evaluate_policy_on_scenario(
                    policy=c_inst,
                    data_df=scen_info["df"],
                    scenario_type=scen_info["type"],
                    reward_fn=reward_fn,
                    safety_gateway=safety_gateway,
                    n_episodes=8 if scen_info["df"] is not None else 5,
                    base_seed=s * 100 + 42,
                    record_actions=rec_acts
                )
                if rec_acts:
                    action_totals_testA[c_name] += np.array(res.pop("action_counts"), dtype=int)
                res["controller"] = c_name
                res["scenario"] = scen_key
                res["seed"] = s
                seed_records.append(res)

    df_seed_results = pd.DataFrame(seed_records)
    df_seed_results.to_csv(out_dir / "seed_level_results.csv", index=False)

    # Export Test-A Action Distribution CSV
    act_rows = []
    for c_name in controllers_names:
        counts = action_totals_testA[c_name]
        tot_cnt = max(1, int(counts.sum()))
        for a_id in range(12):
            act_rows.append({
                "controller": c_name,
                "action_id": a_id,
                "action_name": MULTI_DEVICE_ACTION_TEMPLATES[a_id].name,
                "count": int(counts[a_id]),
                "pct": float(counts[a_id] / tot_cnt * 100.0)
            })
    df_actions = pd.DataFrame(act_rows)
    df_actions.to_csv(out_dir / "action_distribution.csv", index=False)
    print(f"  Saved Test-A real action distribution to {out_dir / 'action_distribution.csv'}")

    # Export Representative 24-Hour Real Rollout Trajectory on Test-A (Seed 0)
    rep_start = _find_representative_weekday_start(test_df)
    traj_env = MultiDeviceHVACEnv(data_df=test_df, episode_length=24, seed=42)
    traj_ctrls = {
        "Double Q-Learning": dql_agents[0],
        "Single Q-Learning": ql_agents[0],
        "Fixed Setpoint (25°C Cool)": fixed_policy,
        "Conventional Thermostat": thermostat_policy,
        "Rule-Based Smart HVAC": rule_based_policy,
    }
    traj_all = []
    for c_name, c_inst in traj_ctrls.items():
        ep_res = evaluate_episode(
            policy=c_inst, env=traj_env, reward_fn=reward_fn,
            safety_gateway=safety_gateway, start_idx=rep_start, seed=42,
            record_trajectory=True, controller_name=c_name
        )
        traj_all.extend(ep_res["trajectory_rows"])
    df_traj = pd.DataFrame(traj_all)
    df_traj.to_csv(out_dir / "trajectory_24h.csv", index=False)
    print(f"  Saved real 24h Test-A trajectory ({len(df_traj)} rows, start_idx={rep_start}) to {out_dir / 'trajectory_24h.csv'}")

    # Export per-subscenario Test-C breakdown
    test_c_sub = df_seed_results[df_seed_results["scenario"].str.startswith("test-C_")].copy()
    sub_rows = []
    test_c_breakdown = {}
    for c_name in controllers_names:
        test_c_breakdown[c_name] = {}
        for scen_sub in ["test-C_heatwave", "test-C_large_occupancy", "test-C_smog_pollution"]:
            sub = test_c_sub[(test_c_sub["controller"] == c_name) & (test_c_sub["scenario"] == scen_sub)]
            row_dict = {
                "controller": c_name,
                "scenario": scen_sub,
                "energy_kwh_mean": float(sub["energy_kwh_per_day"].mean()),
                "energy_kwh_std": float(sub["energy_kwh_per_day"].std()),
                "comfort_violation_pct_mean": float(sub["comfort_violation_pct"].mean()),
                "comfort_violation_pct_std": float(sub["comfort_violation_pct"].std()),
                "co2_violation_pct_mean": float(sub["co2_violation_pct"].mean()),
                "co2_violation_pct_std": float(sub["co2_violation_pct"].std()),
                "pm25_violation_pct_mean": float(sub["pm25_violation_pct"].mean()),
                "pm25_violation_pct_std": float(sub["pm25_violation_pct"].std()),
                "rh_violation_pct_mean": float(sub["rh_violation_pct"].mean()),
                "rh_violation_pct_std": float(sub["rh_violation_pct"].std()),
                "switching_per_day_mean": float(sub["switching_per_day"].mean()),
                "override_pct_mean": float(sub["override_pct"].mean()),
                "mean_reward_mean": float(sub["mean_reward"].mean()),
                "mean_reward_std": float(sub["mean_reward"].std()),
            }
            sub_rows.append(row_dict)
            test_c_breakdown[c_name][scen_sub] = {
                "energy_kwh_per_day": f"{row_dict['energy_kwh_mean']:.2f} ± {row_dict['energy_kwh_std']:.2f}",
                "comfort_violation_pct": f"{row_dict['comfort_violation_pct_mean']:.1f}% ± {row_dict['comfort_violation_pct_std']:.1f}%",
                "co2_violation_pct": f"{row_dict['co2_violation_pct_mean']:.1f}% ± {row_dict['co2_violation_pct_std']:.1f}%",
                "pm25_violation_pct": f"{row_dict['pm25_violation_pct_mean']:.1f}% ± {row_dict['pm25_violation_pct_std']:.1f}%",
                "mean_reward": f"{row_dict['mean_reward_mean']:.2f} ± {row_dict['mean_reward_std']:.2f}",
            }
    pd.DataFrame(sub_rows).to_csv(out_dir / "test_c_summary.csv", index=False)

    # Aggregate Test-C components into combined "test-C" for benchmark_results.csv / Gate 3 compatibility
    test_c_rows = df_seed_results[df_seed_results["scenario"].str.startswith("test-C_")].copy()
    test_c_rows["scenario"] = "test-C"
    df_all_eval = pd.concat(
        [df_seed_results[~df_seed_results["scenario"].str.startswith("test-C_")], test_c_rows],
        ignore_index=True
    )

    # 6. Statistical Hypothesis Testing on Test-A (DQL vs QL + Baselines across 10 paired seeds)
    print("\n[Step 5] Running Statistical Hypothesis Tests on Test-A...")
    dql_testA = df_all_eval[(df_all_eval["controller"] == "Double Q-Learning") & (df_all_eval["scenario"] == "test-A")].sort_values("seed")
    ql_testA = df_all_eval[(df_all_eval["controller"] == "Single Q-Learning") & (df_all_eval["scenario"] == "test-A")].sort_values("seed")

    dql_r = dql_testA["mean_reward"].values
    ql_r = ql_testA["mean_reward"].values
    dql_energy = dql_testA["energy_kwh_per_day"].values
    ql_energy = ql_testA["energy_kwh_per_day"].values

    t_stat_r, p_val_t_r = stats.ttest_rel(dql_r, ql_r)
    w_stat_r, p_val_w_r = stats.wilcoxon(dql_r, ql_r)
    diff_r = dql_r - ql_r
    sd_diff_r = float(np.std(diff_r, ddof=1))
    d_r = float(np.mean(diff_r) / (sd_diff_r if sd_diff_r > 1e-8 else 1.0))

    t_stat_e, p_val_t_e = stats.ttest_rel(dql_energy, ql_energy)
    diff_e = dql_energy - ql_energy
    sd_diff_e = float(np.std(diff_e, ddof=1))
    d_e = float(np.mean(diff_e) / (sd_diff_e if sd_diff_e > 1e-8 else 1.0))

    rng = np.random.default_rng(42)
    boot_means = [float(np.mean(rng.choice(diff_r, size=len(diff_r), replace=True))) for _ in range(10000)]
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    lev_stat, p_val_lev = stats.levene(dql_r, ql_r)

    # Holm-Bonferroni comparisons: DQL vs QL & all 4 Baselines on Test-A mean_reward
    raw_baseline_p = {"Single Q-Learning": float(p_val_t_r)}
    baseline_diffs = {}
    for b_name in [
        "Fixed Setpoint (25°C Cool)",
        "Conventional Thermostat",
        "Rule-Based Smart HVAC",
        "Random Policy"
    ]:
        b_df = df_all_eval[(df_all_eval["controller"] == b_name) & (df_all_eval["scenario"] == "test-A")].sort_values("seed")
        b_r = b_df["mean_reward"].values
        t_b, p_b = stats.ttest_rel(dql_r, b_r)
        raw_baseline_p[b_name] = float(p_b)
        diff_b = dql_r - b_r
        sd_b = float(np.std(diff_b, ddof=1))
        baseline_diffs[b_name] = {
            "mean_reward_diff": float(np.mean(diff_b)),
            "t_statistic": float(t_b),
            "raw_p_value": float(p_b),
            "cohens_dz": float(np.mean(diff_b) / (sd_b if sd_b > 1e-8 else 1.0)),
        }
    adj_p = holm_bonferroni(raw_baseline_p)
    for b_name in baseline_diffs:
        baseline_diffs[b_name]["holm_bonferroni_p"] = adj_p[b_name]

    statistical_tests = {
        "paired_t": {
            "statistic": float(t_stat_r),
            "p_value": float(p_val_t_r),
            "holm_p_value": float(adj_p["Single Q-Learning"]),
            "metric": "mean_reward"
        },
        "paired_t_energy": {
            "statistic": float(t_stat_e),
            "p_value": float(p_val_t_e),
            "mean_diff_kwh": float(np.mean(diff_e)),
            "cohens_dz": d_e,
            "metric": "energy_kwh_per_day"
        },
        "wilcoxon": {
            "statistic": float(w_stat_r),
            "p_value": float(p_val_w_r),
            "metric": "mean_reward"
        },
        "cohens_d": {
            "effect_size": d_r,
            "definition": "paired_dz",
            "interpretation": "large" if abs(d_r) >= 0.8 else ("medium" if abs(d_r) >= 0.5 else ("small" if abs(d_r) >= 0.2 else "negligible"))
        },
        "bootstrap_ci": {
            "confidence_level": 0.95,
            "mean_difference": float(np.mean(diff_r)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "resamples": 10000
        },
        "levene": {
            "statistic": float(lev_stat),
            "p_value": float(p_val_lev),
            "dql_variance": float(np.var(dql_r, ddof=1)),
            "ql_variance": float(np.var(ql_r, ddof=1))
        },
        "holm_bonferroni_baselines": baseline_diffs,
    }

    print(f"  Paired t-test (Reward): t={t_stat_r:.4f}, p={p_val_t_r:.4e} (Holm p={adj_p['Single Q-Learning']:.4e})")
    print(f"  Paired t-test (Energy): t={t_stat_e:.4f}, p={p_val_t_e:.4e}, diff={np.mean(diff_e):+.3f} kWh/day")
    print(f"  Wilcoxon test (Reward): W={w_stat_r:.4f}, p={p_val_w_r:.4e}")
    print(f"  Cohen's d_z   (Reward): d={d_r:.4f}")
    print(f"  Bootstrap 95% CI      : [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Levene variance test  : stat={lev_stat:.4f}, p={p_val_lev:.4e}")

    if p_val_t_r < 0.05 and d_r >= 0.5:
        interpretation_branch = 1
        branch_desc = "Branch 1: DQL significantly outperforms QL (p < 0.05, d >= 0.5)"
    elif p_val_t_r < 0.05 and d_r <= -0.5:
        interpretation_branch = 3
        branch_desc = "Branch 3: QL significantly outperforms DQL (p < 0.05, d <= -0.5)"
    else:
        interpretation_branch = 2
        branch_desc = "Branch 2: DQL and QL are statistically indistinguishable (p >= 0.05 or |d| < 0.2)"

    print(f"\n[*] INTERPRETATION BRANCH SELECTED: {interpretation_branch} ({branch_desc})")

    # 7. Maximization Bias Verification
    print("\n[Step 6] Running Maximization Bias Verification Experiments...")
    toy_bias = run_maximization_bias_toy(n_runs=2000, n_episodes=300, output_dir=out_dir)

    # Save real Toy MaxBias curves
    df_toy_curves = pd.DataFrame({
        "episode": list(range(1, len(toy_bias["ql_left_pct_curve"]) + 1)),
        "ql_left_pct": toy_bias["ql_left_pct_curve"],
        "dql_left_pct": toy_bias["dql_left_pct_curve"],
        "optimal_eps_half_pct": [5.0] * len(toy_bias["ql_left_pct_curve"]),
    })
    df_toy_curves.to_csv(out_dir / "maxbias_toy_curves.csv", index=False)

    env_eval = MultiDeviceHVACEnv(data_df=test_df, episode_length=24, seed=42)
    hvac_ql_bias, hvac_dql_bias = measure_hvac_mc_bias(
        agent_dql=dql_agents[0], agent_ql=ql_agents[0],
        env_eval=env_eval, reward_fn=reward_fn,
        n_sample_states=20, mc_rollouts=50
    )
    print(f"  HVAC MC Bias: QL={hvac_ql_bias:.4f}, DQL={hvac_dql_bias:.4f}")

    # 8. Build Aggregate Results Table
    summary_rows = []
    for c_name in controllers_names:
        for scen in ["test-A", "test-B", "test-C"]:
            sub = df_all_eval[(df_all_eval["controller"] == c_name) & (df_all_eval["scenario"] == scen)]
            summary_rows.append({
                "controller": c_name,
                "scenario": scen,
                "energy_kwh_mean": float(sub["energy_kwh_per_day"].mean()),
                "energy_kwh_std": float(sub["energy_kwh_per_day"].std()),
                "comfort_violation_pct_mean": float(sub["comfort_violation_pct"].mean()),
                "comfort_violation_pct_std": float(sub["comfort_violation_pct"].std()),
                "co2_violation_pct_mean": float(sub["co2_violation_pct"].mean()),
                "co2_violation_pct_std": float(sub["co2_violation_pct"].std()),
                "pm25_violation_pct_mean": float(sub["pm25_violation_pct"].mean()),
                "pm25_violation_pct_std": float(sub["pm25_violation_pct"].std()),
                "rh_violation_pct_mean": float(sub["rh_violation_pct"].mean()),
                "rh_violation_pct_std": float(sub["rh_violation_pct"].std()),
                "switching_per_day_mean": float(sub["switching_per_day"].mean()),
                "switching_per_day_std": float(sub["switching_per_day"].std()),
                "override_pct_mean": float(sub["override_pct"].mean()),
                "override_pct_std": float(sub["override_pct"].std()),
                "mean_reward_mean": float(sub["mean_reward"].mean()),
                "mean_reward_std": float(sub["mean_reward"].std())
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_dir / "benchmark_results.csv", index=False)

    # 9. Build Manifest JSON
    print("\n[Step 7] Generating Final Manifest JSON...")
    try:
        commit_res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        commit_hash = commit_res.stdout.strip()
    except Exception:
        commit_hash = "unknown"

    manifest_controllers = {}
    for c_name in controllers_names:
        manifest_controllers[c_name] = {}
        for scen in ["test-A", "test-B", "test-C"]:
            row = df_summary[(df_summary["controller"] == c_name) & (df_summary["scenario"] == scen)].iloc[0]
            manifest_controllers[c_name][scen] = {
                "energy_kwh_per_day": f"{row['energy_kwh_mean']:.2f} ± {row['energy_kwh_std']:.2f}",
                "comfort_violation_pct": f"{row['comfort_violation_pct_mean']:.1f}% ± {row['comfort_violation_pct_std']:.1f}%",
                "co2_violation_pct": f"{row['co2_violation_pct_mean']:.1f}% ± {row['co2_violation_pct_std']:.1f}%",
                "pm25_violation_pct": f"{row['pm25_violation_pct_mean']:.1f}% ± {row['pm25_violation_pct_std']:.1f}%",
                "rh_violation_pct": f"{row['rh_violation_pct_mean']:.1f}% ± {row['rh_violation_pct_std']:.1f}%",
                "switching_per_day": f"{row['switching_per_day_mean']:.1f} ± {row['switching_per_day_std']:.1f}",
                "override_pct": f"{row['override_pct_mean']:.1f}% ± {row['override_pct_std']:.1f}%",
                "mean_reward": f"{row['mean_reward_mean']:.2f} ± {row['mean_reward_std']:.2f}"
            }

    manifest = {
        "version": "2.0.0",
        "generated_timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_hash": commit_hash,
        "git_tag": active_tag,
        "controllers": manifest_controllers,
        "test_c_breakdown": test_c_breakdown,
        "scenarios": {
            "test-A": "2018_F2_Z1 held-out temporal partition (in-distribution)",
            "test-B": "2019_F2_Z1 cross-year domain transfer",
            "test-C": "Synthetic stress scenarios (heatwave, large_occupancy, smog_pollution)"
        },
        "statistical_tests": statistical_tests,
        "interpretation_branch": interpretation_branch,
        "bias_experiment": {
            "ql_mean_bias": toy_bias["ql_mean_bias"],
            "dql_mean_bias": toy_bias["dql_mean_bias"],
            "hvac_mc_ql_bias": hvac_ql_bias,
            "hvac_mc_dql_bias": hvac_dql_bias
        },
        "hyperparameters": {
            "double_q_learning": best_dql_params,
            "single_q_learning": best_ql_params
        },
        "sample_sizes": {
            "train_hours": len(train_df),
            "val_hours": len(val_df),
            "test_a_hours": len(test_df),
            "test_b_hours": len(df_2019),
            "eval_seeds": len(eval_seeds)
        }
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written to: {manifest_path}")

    # 10. Generate Quick Test-A Comparison Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    df_testA_summary = df_summary[df_summary["scenario"] == "test-A"]
    colors = ["#0B6E70", "#D9534F", "#337AB7", "#F0AD4E", "#5CB85C", "#777777"]
    axes[0].bar(df_testA_summary["controller"], df_testA_summary["energy_kwh_mean"], yerr=df_testA_summary["energy_kwh_std"], capsize=5, color=colors)
    axes[0].set_title("Daily Energy Consumption (Test-A)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("kWh / day", fontsize=11)
    axes[0].set_xticks(range(len(df_testA_summary["controller"])))
    axes[0].set_xticklabels(df_testA_summary["controller"], rotation=25, ha="right", fontsize=9)
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    axes[1].bar(df_testA_summary["controller"], df_testA_summary["comfort_violation_pct_mean"], yerr=df_testA_summary["comfort_violation_pct_std"], capsize=5, color=colors)
    axes[1].set_title("Thermal Discomfort Violation % (Test-A)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("% Hours Outside [24°C, 26°C]", fontsize=11)
    axes[1].set_xticks(range(len(df_testA_summary["controller"])))
    axes[1].set_xticklabels(df_testA_summary["controller"], rotation=25, ha="right", fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_dir / "benchmark_comparison_testA.png")
    plt.close(fig)

    total_elapsed = time.time() - start_time_all
    print(f"\n=======================================================")
    print(f"  PHASE 3 BENCHMARK COMPLETED SUCCESSFULLY in {total_elapsed:.1f}s ({total_elapsed/60:.1f} mins)")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
