import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import pandas as pd
from src.data.domain_loader import get_domain_dataset, get_processed_dir

pdir = get_processed_dir()
domains = ["2018_F2_Z1", "2018_F2_Z2", "2018_F3_Z1", "2019_F2_Z1", "2019_F3_Z1"]

print("=" * 70)
print("DOMAINS REGENERATION REPORT (RAW -> PROCESSED)")
print("=" * 70)

print("\n--- 1. BEFORE REGENERATION ---")
before_info = {}
for d in domains:
    p = pdir / f"{d}_1h.csv"
    if p.exists():
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        df = pd.read_csv(p)
        before_info[d] = {"rows": len(df), "sha256": h}
        print(f"{d:12s} | Rows: {len(df):5d} | SHA256: {h}")
    else:
        before_info[d] = {"rows": 0, "sha256": "MISSING"}
        print(f"{d:12s} | MISSING")

print("\n--- 2. REGENERATING FROM RAW VIA PREPROCESS_DOMAIN_TIMESERIES ---")
for d in domains:
    df_new = get_domain_dataset(d, force_recompute=True)
    print(f"[*] Processed {d:12s}: {len(df_new):5d} records generated.")

print("\n--- 3. AFTER REGENERATION ---")
for d in domains:
    p = pdir / f"{d}_1h.csv"
    with open(p, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    df = pd.read_csv(p)
    row_diff = len(df) - before_info[d]["rows"]
    is_same = (h == before_info[d]["sha256"])
    print(f"{d:12s} | Rows: {len(df):5d} (diff: {row_diff:+d}) | SHA256: {h} | Checksum match: {is_same}")
