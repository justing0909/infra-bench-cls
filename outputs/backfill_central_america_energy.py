"""
Back-fill central-america with the full energy preset (all 7 classes,
not just the 4 substation subclasses). Uses the existing
central-america-260526.osm_power_only.osm.pbf — no pre-filter needed.
"""
from __future__ import annotations
import os, sys, time
REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

import pandas as pd
from sources import GeoFabrikSource
from deduplication import Deduplicator

POWER_ONLY = (
    r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm\data\pbf\power_only"
    r"\central-america-latest.osm_power_only.osm.pbf"
)
EXTRACT_OUT = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets",
                           "central-america_all_assets_energy.parquet")
DEDUP_OUT   = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets",
                           "central-america_deduped_assets_energy.parquet")

if not os.path.exists(POWER_ONLY):
    raise SystemExit(f"Missing: {POWER_ONLY}")

print(f"Extracting energy preset from {POWER_ONLY} ({os.path.getsize(POWER_ONLY)/1_048_576:.1f} MB)")
t0 = time.time()
src = GeoFabrikSource(POWER_ONLY, min_confidence="medium",
                      filter_preset="energy", pre_filter=False)
df = src.extract_all()
print(f"Extract done in {time.time()-t0:.1f}s -> {len(df):,} rows")
print()
print("Per-class counts:")
print(df["asset_type"].value_counts().to_string())
print()
df.drop(columns=["osm_tags"], errors="ignore").to_parquet(EXTRACT_OUT, index=False)
print(f"Saved -> {EXTRACT_OUT}")

print()
print("Dedup with per-class thresholds...")
t = time.time()
dedup = Deduplicator()
clean, removed = dedup.run(df)
print(f"Dedup done in {time.time()-t:.1f}s -> {len(clean):,} kept")
clean.drop(columns=["osm_tags"], errors="ignore").to_parquet(DEDUP_OUT, index=False)
print(f"Saved -> {DEDUP_OUT}")
