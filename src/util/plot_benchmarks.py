"""
Plotting module for Benchmark Results:
- Baselines Benchmark
- Domain Shift (3 Transfer Modes)
- Robustness and Plant Uncertainty
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

plt.style.use("default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def plot_baselines_summary(df_base: pd.DataFrame, save_path: Path) -> None:
    """Plot bar chart comparing all evaluated baseline policies with multi-seed statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    policies = df_base["Policy"].tolist()
    n_p = len(policies)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_p)]

    # 1. Daily Energy
    axes[0].bar(range(len(policies)), df_base["Daily Energy (kWh)"], color=colors, width=0.55)
    axes[0].set_xticks(range(len(policies)))
    axes[0].set_xticklabels(policies, rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Daily HVAC Energy (kWh)", fontweight="bold")
    axes[0].set_title("Energy Consumption (Lower is better)", fontweight="bold")
    for i, v in enumerate(df_base["Daily Energy (kWh)"]):
        axes[0].text(i, v + (df_base["Daily Energy (kWh)"].max() * 0.02), f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Severe Comfort Violations
    axes[1].bar(range(len(policies)), df_base["Severe Violation (%)"], color=colors, width=0.55)
    axes[1].set_xticks(range(len(policies)))
    axes[1].set_xticklabels(policies, rotation=35, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Severe Violation Rate (%)", fontweight="bold")
    axes[1].set_title("Severe Discomfort Rate (<21°C or >29°C)", fontweight="bold")
    for i, v in enumerate(df_base["Severe Violation (%)"]):
        axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Mean Reward
    axes[2].bar(range(len(policies)), df_base["Mean Reward"], color=colors, width=0.55)
    axes[2].set_xticks(range(len(policies)))
    axes[2].set_xticklabels(policies, rotation=35, ha="right", fontsize=8.5)
    axes[2].set_ylabel("Mean Episode Reward", fontweight="bold")
    axes[2].set_title("Overall Reward (Higher is better)", fontweight="bold")
    for i, v in enumerate(df_base["Mean Reward"]):
        axes[2].text(i, v - (abs(df_base["Mean Reward"].min()) * 0.04), f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Comprehensive Controller Multi-Seed Benchmark on Unseen 2018 Test Period", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved baselines comparison to: {save_path}")


def plot_domain_shift_modes(df_shift: pd.DataFrame, save_path: Path) -> None:
    """Plot grouped bar chart comparing Zero-Shot, Calibration-Only, and Fine-Tuning across budgets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # Identify available transfer modes
    all_modes = list(df_shift["Transfer Mode"].unique())
    # Desired logical ordering
    preferred_order = ["ZERO-SHOT", "CALIBRATION-ONLY", "FINE-TUNING-10EP", "FINE-TUNING-25EP", "FINE-TUNING"]
    mode_order = [m for m in preferred_order if m in all_modes]
    for m in all_modes:
        if m not in mode_order:
            mode_order.append(m)

    # Group by Target Domain and Transfer Mode
    pivot_r = df_shift.pivot(index="Target Domain", columns="Transfer Mode", values="Mean Reward")
    pivot_e = df_shift.pivot(index="Target Domain", columns="Transfer Mode", values="Energy (kWh/day)")

    pivot_r = pivot_r.reindex(columns=mode_order)
    pivot_e = pivot_e.reindex(columns=mode_order)

    x = np.arange(len(pivot_r))
    n_modes = len(mode_order)
    width = 0.8 / max(1, n_modes)

    palette = ["#ef4444", "#f59e0b", "#3b82f6", "#10b981", "#8b5cf6"]

    # 1. Mean Reward across Transfer Modes
    for idx, mode in enumerate(mode_order):
        offset = (idx - (n_modes - 1) / 2) * width
        color = palette[idx % len(palette)]
        ax1.bar(x + offset, pivot_r[mode], width, label=mode, color=color, alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot_r.index, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Mean Daily Reward", fontweight="bold")
    ax1.set_title("Reward Comparison by Transfer Mode (Higher is better)", fontweight="bold")
    ax1.legend(fontsize=8.5)
    ax1.grid(axis="y", alpha=0.3)

    # 2. Daily Energy across Transfer Modes
    for idx, mode in enumerate(mode_order):
        offset = (idx - (n_modes - 1) / 2) * width
        color = palette[idx % len(palette)]
        ax2.bar(x + offset, pivot_e[mode], width, label=mode, color=color, alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_e.index, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Daily HVAC Energy (kWh)", fontweight="bold")
    ax2.set_title("Energy Consumption by Transfer Mode", fontweight="bold")
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Domain Shift Transfer Performance: Zero-Shot vs Calibration-Only vs Multi-Budget Fine-Tuning", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved domain shift comparison to: {save_path}")
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Domain Shift Transfer Performance: Zero-Shot vs Calibration-Only vs Fine-Tuning", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved domain shift comparison to: {save_path}")


def plot_robustness_perturbations(df_rob: pd.DataFrame, save_path: Path) -> None:
    """Plot horizontal bar chart of robustness degradation."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    y_pos = np.arange(len(df_rob))
    ax.barh(y_pos, df_rob["Mean Reward"], xerr=df_rob["Reward Std Across Seeds"], align="center", color="#0275d8", alpha=0.85, capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_rob["Perturbation Scenario"], fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mean Daily Reward ± Std across seeds", fontweight="bold")
    ax.set_title("Plant Perturbation & Environmental Robustness Analysis", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for i, v in enumerate(df_rob["Mean Reward"]):
        ax.text(v - 5, i, f"{v:.1f}", va="center", ha="right", fontsize=9, fontweight="bold", color="black")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved robustness comparison to: {save_path}")


def main():
    log_dir = project_root / "result" / "log"
    fig_dir = project_root / "result" / "figure"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_csv = log_dir / "baselines_benchmark.csv"
    shift_csv = log_dir / "domain_shift_benchmark.csv"
    rob_csv = log_dir / "robustness_benchmark.csv"

    if base_csv.exists():
        plot_baselines_summary(pd.read_csv(base_csv), fig_dir / "baselines_comparison.png")
    if shift_csv.exists():
        plot_domain_shift_modes(pd.read_csv(shift_csv), fig_dir / "domain_shift_comparison.png")
    if rob_csv.exists():
        plot_robustness_perturbations(pd.read_csv(rob_csv), fig_dir / "robustness_comparison.png")

    print("[Complete] All benchmark figures generated successfully!")


if __name__ == "__main__":
    main()
