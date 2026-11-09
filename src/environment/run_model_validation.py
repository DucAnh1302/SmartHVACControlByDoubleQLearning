"""
Script to run transition model benchmarking and save reports.
"""

from pathlib import Path
import sys
import json

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.domain_loader import get_domain_dataset, split_temporal_dataset
from src.environment.model_validator import benchmark_all_transition_models


def main():
    print("=== Running Comprehensive Transition Model Benchmark ===")
    df18 = get_domain_dataset("2018_F2_Z1")
    df_train, df_test = split_temporal_dataset(df18, 0.70)
    df_unseen_2019 = get_domain_dataset("2019_F2_Z1")

    df_summary, full_reports = benchmark_all_transition_models(df_train, df_test, df_unseen_2019)

    print("\nBenchmark Summary Table:")
    print(df_summary.to_string(index=False))

    out_dir = project_root / "result" / "log"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "transition_model_benchmark.json"
    out_csv = out_dir / "transition_model_benchmark.csv"

    df_summary.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_reports, f, indent=2)

    print(f"\n[Saved] Benchmark summary to: {out_csv}")
    print(f"[Saved] Full validation report to: {out_json}")


if __name__ == "__main__":
    main()
