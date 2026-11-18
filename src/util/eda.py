"""
Script to generate all Exploratory Data Analysis (EDA) figures from CU-BEMS dataset.
"""

from pathlib import Path
import sys
import pandas as pd

# Add project root to sys.path
curr_dir = Path(__file__).resolve().parent
project_root = curr_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.util.visualization import (
    plot_time_series,
    plot_correlation_matrix,
    plot_diurnal_patterns,
    plot_comfort_distribution
)


def run_eda():
    data_path = project_root / "data" / "processed" / "cubems_zone1_hourly.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found at: {data_path}")

    print(f"[EDA] Loading processed data from: {data_path}")
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    print("\n=== Dataset Summary Statistics ===")
    stats_cols = ["temp_c", "humidity_rh", "light_lux", "plug_kw", "light_kw", "internal_heat_kw", "hvac_kw"]
    print(df[stats_cols].describe().round(2))

    print("\n=== Generating EDA Visualizations ===")
    plot_time_series(df, start_idx=200, n_hours=168)
    plot_correlation_matrix(df)
    plot_diurnal_patterns(df)
    plot_comfort_distribution(df)

    print("\n=== All EDA Visualizations Generated Successfully! ===")


if __name__ == "__main__":
    run_eda()
