"""
Evaluation Script for Multi-Device Smart HVAC Control (720 states / 12 actions).
Loads trained checkpoints (model/final_dql_seed0.pkl and model/final_ql_seed0.pkl)
plus the 4 deterministic/random baselines and evaluates them on held-out Test-A data.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
import pandas as pd

# Setup project root in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.domain_loader import get_domain_dataset, split_temporal_dataset
from src.reward.multi_device_reward import MultiDeviceRewardFunction
from src.safety.multi_device_safety_gateway import MultiDeviceSafetyGateway
from src.agent.multi_device_double_q_agent import MultiDeviceDoubleQAgent
from src.agent.multi_device_baselines import (
    FixedSetpointMultiDevice,
    ConventionalThermostatMultiDevice,
    RuleBasedSmartHVACMultiDevice,
    RandomMultiDevicePolicy,
)
from run_final_experiments import evaluate_policy_on_scenario


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Multi-Device HVAC controllers on Test-A.")
    parser.add_argument("--episodes", type=int, default=16, help="Number of 24h evaluation episodes on Test-A.")
    parser.add_argument("--seed", type=int, default=42, help="Base evaluation seed.")
    args = parser.parse_args()

    print("=" * 68)
    print("  SmartHVAC: Multi-Device Model Evaluation on Held-Out Test-A")
    print("=" * 68)

    df_2018 = get_domain_dataset("2018_F2_Z1")
    _, _, test_df = split_temporal_dataset(
        df_2018, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, buffer_hours=24
    )
    print(f"Loaded Test-A Split: {len(test_df)} hours ({test_df['datetime'].min()} to {test_df['datetime'].max()})")

    model_dir = project_root / "model"
    dql_path = model_dir / "final_dql_seed0.pkl"
    ql_path = model_dir / "final_ql_seed0.pkl"
    if not dql_path.exists() or not ql_path.exists():
        raise FileNotFoundError(
            "Trained checkpoints model/final_dql_seed0.pkl and model/final_ql_seed0.pkl not found! "
            "Run python run_final_experiments.py first."
        )

    agent_dql = MultiDeviceDoubleQAgent.load_checkpoint(dql_path)
    with open(ql_path, "rb") as f:
        agent_ql = pickle.load(f)

    controllers = {
        "Double Q-Learning": agent_dql,
        "Single Q-Learning": agent_ql,
        "Fixed Setpoint (25°C Cool)": FixedSetpointMultiDevice(),
        "Conventional Thermostat": ConventionalThermostatMultiDevice(),
        "Rule-Based Smart HVAC": RuleBasedSmartHVACMultiDevice(),
        "Random Policy": RandomMultiDevicePolicy(seed=args.seed),
    }

    reward_fn = MultiDeviceRewardFunction()
    safety_gateway = MultiDeviceSafetyGateway()

    rows = []
    for name, ctrl in controllers.items():
        res = evaluate_policy_on_scenario(
            policy=ctrl,
            data_df=test_df,
            scenario_type="standard",
            reward_fn=reward_fn,
            safety_gateway=safety_gateway,
            n_episodes=args.episodes,
            base_seed=args.seed,
        )
        res["controller"] = name
        rows.append(res)

    df_results = pd.DataFrame(rows)[
        [
            "controller",
            "energy_kwh_per_day",
            "mean_reward",
            "comfort_violation_pct",
            "co2_violation_pct",
            "pm25_violation_pct",
            "rh_violation_pct",
            "switching_per_day",
            "override_pct",
        ]
    ]

    print("\n=== TEST-A EVALUATION SUMMARY ===")
    print(df_results.to_string(index=False))

    log_dir = project_root / "result" / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(log_dir / "evaluation_summary.csv", index=False)
    with open(log_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=2)
    print(f"\n[Logging] Saved evaluation summary to: {log_dir / 'evaluation_summary.csv'}")


if __name__ == "__main__":
    main()
