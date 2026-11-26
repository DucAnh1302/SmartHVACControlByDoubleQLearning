"""
Master Execution Script for Smart HVAC Control Benchmark.
One command to reproduce the entire Phase 3 experimental pipeline:
1. Executes Phase 3 experiments (validation grid search, 10 seeds training, 6 controllers x 3 test scenarios).
2. Runs toy and HVAC maximization bias verification.
3. Generates all final tables, plots, and manifest.json.
4. Executes automated Gate 3 verification.

Estimated runtime: ~12-15 minutes on multi-core CPU.
"""

import sys
import time
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 80)
    print("  SMART HVAC CONTROL: REPRODUCE ALL BENCHMARK EXPERIMENTS")
    print("  Author: Nhom 2 (Machine Learning Capstone Project)")
    print("=" * 80)

    t0 = time.time()

    print("\n>>> Step 1/2: Running Comprehensive Benchmark (run_final_experiments.py)...")
    cmd_exp = [sys.executable, str(REPO_ROOT / "run_final_experiments.py")]
    res_exp = subprocess.run(cmd_exp, cwd=str(REPO_ROOT))
    if res_exp.returncode != 0:
        print("[!] Error in run_final_experiments.py! Exiting.")
        sys.exit(res_exp.returncode)

    print("\n>>> Step 2/2: Running Automated Gate 3 Verification (tools/run_gate.py 3)...")
    cmd_gate = [sys.executable, str(REPO_ROOT / "tools" / "run_gate.py"), "3"]
    res_gate = subprocess.run(cmd_gate, cwd=str(REPO_ROOT))
    if res_gate.returncode != 0:
        print("[!] Gate 3 verification failed! Exiting.")
        sys.exit(res_gate.returncode)

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"  ALL BENCHMARKS AND GATE CHECKS PASSED in {elapsed:.1f}s ({elapsed/60:.1f} mins)!")
    print("=" * 80)

if __name__ == "__main__":
    main()
