"""
Dedup water + transport central-america parquets with the fixed
(BallTree+haversine) Deduplicator. Default 200m threshold for parity
with substation runs.

Notes:
  - 200m may or may not be the right threshold for every asset type.
    For storage tanks clustered in dense surveys it likely is fine.
    For train stations (subway transfer stations can be ~50m apart)
    the 200m threshold may suppress legitimate neighbors. Flagged.
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
os.makedirs(POST_DIR, exist_ok=True)

CASES = [
    ("central-america_all_assets_water.parquet",
     "central-america_deduped_assets_water.parquet"),
    ("central-america_all_assets_transport.parquet",
     "central-america_deduped_assets_transport.parquet"),
]

for src_name, dst_name in CASES:
    src = os.path.join(PRE_DIR,  src_name)
    dst = os.path.join(POST_DIR, dst_name)
    df = pd.read_parquet(src)
    print(f"\n===== {src_name} =====")
    print(f"  loaded {len(df):,} rows")
    print(f"  per-class:")
    for cls, n in df["asset_type"].value_counts().items():
        print(f"    {cls}: {n}")
    t = time.time()
    dedup = Deduplicator(distance_threshold_m=200)
    clean_df, removed_df = dedup.run(df)
    elapsed = time.time() - t
    print(f"  dedup wall: {elapsed:.1f}s")
    print(f"  kept: {len(clean_df):,}  removed: {len(removed_df):,}  "
          f"({100*len(removed_df)/max(len(df),1):.2f}%)")
    clean_df.to_parquet(dst, index=False)
    print(f"  saved -> {dst}")
    print(f"  post-dedup per-class:")
    for cls, n in clean_df["asset_type"].value_counts().items():
        print(f"    {cls}: {n}")
