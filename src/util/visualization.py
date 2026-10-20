"""
Visualization module for SmartHVACControlByDoubleQLearning.
Provides high-quality figures for EDA, training convergence, and performance comparisons.
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set clean styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def get_default_figure_dir() -> Path:
    """Resolve project result/figure directory."""
    curr_dir = Path(__file__).resolve().parent
    fig_dir = curr_dir.parents[1] / "result" / "figure"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_time_series(
    df: pd.DataFrame,
    start_idx: int = 100,
    n_hours: int = 168,  # 1 week of data
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 5-panel time-series of Temperature, Humidity, Illuminance, Plug Loads, and HVAC Power.
    """
    sub_df = df.iloc[start_idx : start_idx + n_hours].copy()

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    # 1. Temperature
    axes[0].plot(sub_df["datetime"], sub_df["temp_c"], color="#d9534f", lw=2, label="Indoor Temp (°C)")
    axes[0].axhspan(22.0, 24.5, color="#5cb85c", alpha=0.25, label="Comfort Zone (22.0 - 24.5°C)")
    axes[0].set_ylabel("Temp (°C)", fontsize=11, fontweight="bold")
    axes[0].set_title("CU-BEMS Environmental & Energy Profile (1-Week Sample)", fontsize=14, fontweight="bold", pad=10)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # 2. Humidity
    axes[1].plot(sub_df["datetime"], sub_df["humidity_rh"], color="#0275d8", lw=1.8, label="Relative Humidity (%RH)")
    axes[1].set_ylabel("RH (%)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    # 3. Light Illuminance
    axes[2].plot(sub_df["datetime"], sub_df["light_lux"], color="#f0ad4e", lw=1.8, label="Lighting (Lux)")
    axes[2].set_ylabel("Lux", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # 4. Plug Power
    axes[3].plot(sub_df["datetime"], sub_df["plug_kw"], color="#5bc0de", lw=1.8, label="Plug Load (kW)")
    axes[3].set_ylabel("Plug (kW)", fontsize=11, fontweight="bold")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    # 5. HVAC Power
    axes[4].plot(sub_df["datetime"], sub_df["hvac_kw"], color="#292b2c", lw=2, label="HVAC Cooling Power (kW)")
    axes[4].fill_between(sub_df["datetime"], 0, sub_df["hvac_kw"], color="#292b2c", alpha=0.2)
    axes[4].set_ylabel("HVAC (kW)", fontsize=11, fontweight="bold")
    axes[4].set_xlabel("Date & Time", fontsize=11, fontweight="bold")
    axes[4].legend(loc="upper right")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()

    out_file = Path(save_path) if save_path else get_default_figure_dir() / "eda_time_series.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved time-series plot to: {out_file}")
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot annotated heatmap of correlations."""
    cols = ["temp_c", "humidity_rh", "light_lux", "light_kw", "plug_kw", "internal_heat_kw", "hvac_kw"]
    available_cols = [c for c in cols if c in df.columns]

    corr = df[available_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("CU-BEMS Correlation Matrix (Temperature, IAQ & Energy)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_file = Path(save_path) if save_path else get_default_figure_dir() / "eda_correlation_matrix.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved correlation matrix to: {out_file}")
    return fig


def plot_diurnal_patterns(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot 24-hour diurnal patterns for weekdays vs weekends."""
    df_plot = df.copy()
    if "hour" not in df_plot.columns:
        df_plot["hour"] = df_plot["datetime"].dt.hour
    if "is_weekend" not in df_plot.columns:
        df_plot["is_weekend"] = (df_plot["datetime"].dt.dayofweek >= 5).astype(int)

    weekday_grp = df_plot[df_plot["is_weekend"] == 0].groupby("hour").mean(numeric_only=True)
    weekend_grp = df_plot[df_plot["is_weekend"] == 1].groupby("hour").mean(numeric_only=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Temperature Diurnal
    axes[0].plot(weekday_grp.index, weekday_grp["temp_c"], "o-", color="#d9534f", lw=2, label="Weekday")
    axes[0].plot(weekend_grp.index, weekend_grp["temp_c"], "s--", color="#6c757d", lw=2, label="Weekend")
    axes[0].axhspan(22.0, 24.5, color="#5cb85c", alpha=0.2, label="Comfort (22-24.5°C)")
    axes[0].set_title("Indoor Temperature (°C)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Hour of Day (0-23)", fontsize=10)
    axes[0].set_ylabel("Temp (°C)", fontsize=10)
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Internal Heat Loads
    axes[1].plot(weekday_grp.index, weekday_grp["internal_heat_kw"], "o-", color="#f0ad4e", lw=2, label="Weekday")
    axes[1].plot(weekend_grp.index, weekend_grp["internal_heat_kw"], "s--", color="#6c757d", lw=2, label="Weekend")
    axes[1].set_title("Internal Heat (Plug + Light kW)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Hour of Day (0-23)", fontsize=10)
    axes[1].set_ylabel("kW", fontsize=10)
    axes[1].set_xticks(range(0, 24, 3))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: HVAC Power
    axes[2].plot(weekday_grp.index, weekday_grp["hvac_kw"], "o-", color="#0275d8", lw=2, label="Weekday")
    axes[2].plot(weekend_grp.index, weekend_grp["hvac_kw"], "s--", color="#6c757d", lw=2, label="Weekend")
    axes[2].set_title("HVAC Cooling Power (kW)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Hour of Day (0-23)", fontsize=10)
    axes[2].set_ylabel("HVAC kW", fontsize=10)
    axes[2].set_xticks(range(0, 24, 3))
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("CU-BEMS Diurnal Profiles (Weekday Occupied vs. Weekend)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = Path(save_path) if save_path else get_default_figure_dir() / "eda_diurnal_patterns.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved diurnal profiles to: {out_file}")
    return fig


def plot_comfort_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot histogram and density distribution of temperatures vs ASHRAE comfort bounds."""
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        df["temp_c"],
        kde=True,
        color="#0275d8",
        bins=35,
        stat="density",
        ax=ax,
        alpha=0.4
    )

    ax.axvspan(22.0, 24.5, color="#5cb85c", alpha=0.25, label="ASHRAE Comfort Zone (22.0 - 24.5°C)")
    ax.axvline(22.0, color="#5cb85c", linestyle="--", lw=1.5)
    ax.axvline(24.5, color="#5cb85c", linestyle="--", lw=1.5)

    ax.set_title("CU-BEMS Indoor Temperature Distribution vs ASHRAE Standard 55", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Indoor Temperature (°C)", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_file = Path(save_path) if save_path else get_default_figure_dir() / "eda_comfort_distribution.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved comfort distribution plot to: {out_file}")
    return fig


def plot_training_convergence(
    history_ql: Dict[str, List[float]],
    history_dql: Dict[str, List[float]],
    window: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curves comparing Q-Learning Baseline vs Double Q-Learning:
    - Episode Reward convergence (smoothed)
    - Average Maximum Q-value (illustrating overestimation bias)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    episodes_ql = np.arange(1, len(history_ql["rewards"]) + 1)
    episodes_dql = np.arange(1, len(history_dql["rewards"]) + 1)

    # 1. Episode Rewards
    def rolling_mean(arr, w):
        return pd.Series(arr).rolling(window=w, min_periods=1).mean().values

    r_ql_smooth = rolling_mean(history_ql["rewards"], window)
    r_dql_smooth = rolling_mean(history_dql["rewards"], window)

    ax1.plot(episodes_ql, history_ql["rewards"], color="#0275d8", alpha=0.15)
    ax1.plot(episodes_ql, r_ql_smooth, color="#0275d8", lw=2, label="Q-Learning Baseline")

    ax1.plot(episodes_dql, history_dql["rewards"], color="#d9534f", alpha=0.15)
    ax1.plot(episodes_dql, r_dql_smooth, color="#d9534f", lw=2.5, label="Double Q-Learning (Proposed)")

    ax1.set_title(f"Episode Reward Convergence (Smoothed, w={window})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode", fontsize=10)
    ax1.set_ylabel("Total Episode Reward", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # 2. Maximum Q-Value (Overestimation Analysis)
    if "avg_max_q" in history_ql and "avg_max_q" in history_dql:
        q_ql = rolling_mean(history_ql["avg_max_q"], window)
        q_dql = rolling_mean(history_dql["avg_max_q"], window)

        ax2.plot(episodes_ql, q_ql, color="#0275d8", lw=2, label="Q-Learning (Overestimated)")
        ax2.plot(episodes_dql, q_dql, color="#d9534f", lw=2.5, label="Double Q-Learning (Unbiased)")
        ax2.set_title("Estimated Action-Value (Q-Value Overestimation Bias)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Episode", fontsize=10)
        ax2.set_ylabel("Average Q(s, a)", fontsize=10)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Training Performance & Convergence Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = Path(save_path) if save_path else get_default_figure_dir() / "training_convergence.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved training convergence plot to: {out_file}")
    return fig


def plot_evaluation_comparison(
    metrics_summary: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar charts comparing key performance indicators:
    - Comfort Violation Rate (%)
    - Total Energy Consumption (kWh)
    - Average Episode Reward
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = metrics_summary["Model"].tolist()
    colors = ["#6c757d", "#0275d8", "#5cb85c"]  # Grey (Rule-Based), Blue (Q-Learning), Green (Double Q-Learning)

    # 1. Total Reward
    axes[0].bar(models, metrics_summary["Total Reward"], color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Total Cumulative Reward (Higher is better)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Reward", fontsize=10)
    for i, v in enumerate(metrics_summary["Total Reward"]):
        axes[0].text(i, v / 2, f"{v:.1f}", ha="center", va="center", color="white", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # 2. Comfort Violation Rate (%)
    axes[1].bar(models, metrics_summary["Comfort Violation (%)"], color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Thermal Comfort Violation % (Lower is better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Violation Rate (%)", fontsize=10)
    for i, v in enumerate(metrics_summary["Comfort Violation (%)"]):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[1].set_ylim(0, max(metrics_summary["Comfort Violation (%)"]) * 1.25)
    axes[1].grid(axis="y", alpha=0.3)

    # 3. Energy Consumption (kWh)
    axes[2].bar(models, metrics_summary["Total Energy (kWh)"], color=colors, width=0.55, edgecolor="black", linewidth=0.5)
    axes[2].set_title("Total HVAC Energy Consumption (Lower is better)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Energy (kWh)", fontsize=10)
    for i, v in enumerate(metrics_summary["Total Energy (kWh)"]):
        axes[2].text(i, v + 10, f"{v:.1f} kWh", ha="center", va="bottom", fontweight="bold")
    axes[2].set_ylim(0, max(metrics_summary["Total Energy (kWh)"]) * 1.25)
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Comprehensive Controller Evaluation: Baseline vs Q-Learning vs Double Q-Learning", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_file = Path(save_path) if save_path else get_default_figure_dir() / "evaluation_comparison.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved evaluation comparison plot to: {out_file}")
    return fig


def plot_control_trajectory(
    trajectories: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 48-hour simulation control trajectory comparing Rule-Based, Q-Learning, and Double Q-Learning.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors = {
        "Rule-Based": "#6c757d",
        "Q-Learning": "#0275d8",
        "Double Q-Learning": "#28a745"
    }

    # Upper panel: Indoor Temperature
    ax1.axhspan(22.0, 24.5, color="#5cb85c", alpha=0.2, label="Comfort Zone (22.0 - 24.5°C)")
    for name, df_traj in trajectories.items():
        c = colors.get(name, "#333333")
        lw = 2.5 if name == "Double Q-Learning" else 1.8
        ax1.plot(df_traj["step"], df_traj["temp_c"], label=name, color=c, lw=lw)

    ax1.set_ylabel("Indoor Temp (°C)", fontsize=11, fontweight="bold")
    ax1.set_title("48-Hour HVAC Control Trajectory & Thermal Regulation", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Lower panel: HVAC Power
    for name, df_traj in trajectories.items():
        c = colors.get(name, "#333333")
        lw = 2.2 if name == "Double Q-Learning" else 1.5
        ax2.step(df_traj["step"], df_traj["hvac_kw"], where="post", label=name, color=c, lw=lw)

    ax2.set_xlabel("Simulation Step (Hours)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("HVAC Power (kW)", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_file = Path(save_path) if save_path else get_default_figure_dir() / "control_trajectory_comparison.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved control trajectory plot to: {out_file}")
    return fig
