"""
Re-dedup all four central-america sectors using the new per-class
thresholds from ontology.AssetClass.dedup_distance_m.
"""
from __future__ import annotations
import os, sys, time
REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

import pandas as pd
from deduplication import Deduplicator

PRE_DIR  = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
POST_DIR = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets")

CASES = [
    ("central-america_all_assets_substations.parquet",
     "central-america_deduped_assets_substations.parquet"),
    ("central-america_all_assets_water.parquet",
     "central-america_deduped_assets_water.parquet"),
    ("central-america_all_assets_transport.parquet",
     "central-america_deduped_assets_transport.parquet"),
    ("central-america_all_assets_telecom.parquet",
     "central-america_deduped_assets_telecom.parquet"),
]

# Snapshot prior counts (from the most recent flat-200m run)
PRIOR_KEPT = {
    "substations": 1815,
    "water":       1693,
    "transport":    562,
    "telecom":        1,
}

for src_name, dst_name in CASES:
    sector = src_name.replace("central-america_all_assets_", "").replace(".parquet", "")
    print(f"\n===== {sector} =====")
    src = os.path.join(PRE_DIR,  src_name)
    dst = os.path.join(POST_DIR, dst_name)
    df  = pd.read_parquet(src)
    print(f"  input: {len(df):,} rows")
    t = time.time()
    d = Deduplicator()  # no flat threshold; per-class drives everything
    clean, removed = d.run(df)
    elapsed = time.time() - t
    df.drop  # noop
    prior = PRIOR_KEPT.get(sector)
    delta = (len(clean) - prior) if prior is not None else None
    print(f"  wall: {elapsed:.1f}s")
    if prior is not None:
        print(f"  prior kept (flat 200m): {prior:,}   new kept: {len(clean):,}   "
              f"delta: {delta:+,}")
    clean.to_parquet(dst, index=False)
    print(f"  saved -> {dst}")
