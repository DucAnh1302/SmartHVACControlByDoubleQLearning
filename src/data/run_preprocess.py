"""
Script to execute full preprocessing on CU-BEMS dataset and save outputs.
"""

from pathlib import Path
import sys

# Ensure src is in sys.path
curr_dir = Path(__file__).resolve().parent
project_root = curr_dir.parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.loader import load_raw_floor_data, preprocess_cubems_data, save_processed_data
from src.data.preprocess import feature_engineering


def main():
    print("=== Starting CU-BEMS Data Preprocessing ===")
    
    # 1. Load Floor 2 2019
    df_raw = load_raw_floor_data(year=2019, floor=2)
    
    # 2. Process hourly dataset
    print("\n--- Processing Hourly Dataset ---")
    df_hourly = preprocess_cubems_data(df_raw, zone="z1", resample_rule="1h")
    df_hourly = feature_engineering(df_hourly)
    out_h = save_processed_data(df_hourly, "cubems_zone1_hourly.csv")

    # 3. Process 15-minute dataset
    print("\n--- Processing 15-Minute Dataset ---")
    df_15m = preprocess_cubems_data(df_raw, zone="z1", resample_rule="15min")
    df_15m = feature_engineering(df_15m)
    out_15 = save_processed_data(df_15m, "cubems_zone1_15min.csv")

    # 4. Save sample dataset to data/sample
    sample_dir = project_root / "data" / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_path = sample_dir / "sample_cubems.csv"
    df_hourly.head(500).to_csv(sample_path, index=False)
    print(f"\n[Sample] Saved 500-step sample to: {sample_path}")

    print("\n=== Data Preprocessing Completed Successfully! ===")


if __name__ == "__main__":
    main()
