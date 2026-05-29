"""Re-run dedup for europe only with the fixed code."""
from __future__ import annotations
import os, sys, time
REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

import pandas as pd
from deduplication import Deduplicator

PRE  = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets",
                    "europe_all_assets_substations.parquet")
POST = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets",
                    "europe_deduped_assets_substations.parquet")

print(f"Loading {PRE}...", flush=True)
pre_df = pd.read_parquet(PRE)
old_post = pd.read_parquet(POST, columns=["asset_id"])
print(f"  pre_n  = {len(pre_df):,}", flush=True)
print(f"  old_n  = {len(old_post):,}", flush=True)

t0 = time.time()
print(f"Running dedup (BallTree+haversine)...", flush=True)
dedup = Deduplicator(distance_threshold_m=200)
clean_df, removed_df = dedup.run(pre_df)
elapsed = time.time() - t0
print(f"Dedup wall time: {elapsed:.1f}s", flush=True)
print(f"  new_n  = {len(clean_df):,}", flush=True)
delta = len(clean_df) - len(old_post)
old_pct = 100 * (len(pre_df) - len(old_post)) / max(len(pre_df), 1)
new_pct = 100 * (len(pre_df) - len(clean_df)) / max(len(pre_df), 1)
print(f"  delta  = {delta:+,}", flush=True)
print(f"  old_pct= {old_pct:.2f}%", flush=True)
print(f"  new_pct= {new_pct:.2f}%", flush=True)

clean_df.to_parquet(POST, index=False)
print(f"Saved -> {POST}", flush=True)
