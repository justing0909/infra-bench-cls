"""
Re-do energy extract+dedup for central-america and australia-oceania
after the Pattern A solar/wind ontology switch.
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

CASES = [
    ("central-america",
     r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm\data\pbf\power_only"
     r"\central-america-latest.osm_power_only.osm.pbf"),
    ("australia-oceania",
     r"D:\australia-oceania-260526.osm_power_only.osm.pbf"),
]

for region, power_only in CASES:
    print(f"\n========== {region} ==========")
    print(f"Source: {power_only}  ({os.path.getsize(power_only)/1_048_576:.1f} MB)")
    extract = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets",
                           f"{region}_all_assets_energy.parquet")
    dedup_out = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets",
                             f"{region}_deduped_assets_energy.parquet")

    t = time.time()
    src = GeoFabrikSource(power_only, min_confidence="medium",
                          filter_preset="energy", pre_filter=False)
    df = src.extract_all()
    print(f"Extract done in {time.time()-t:.0f}s -> {len(df):,} rows")
    print(df["asset_type"].value_counts().to_string())
    print()
    df.drop(columns=["osm_tags"], errors="ignore").to_parquet(extract, index=False)
    print(f"Saved extract -> {extract}")

    t = time.time()
    dedup = Deduplicator()
    clean, _ = dedup.run(df)
    print(f"Dedup done in {time.time()-t:.0f}s -> {len(clean):,} kept")
    clean.drop(columns=["osm_tags"], errors="ignore").to_parquet(dedup_out, index=False)
    print(f"Saved dedup -> {dedup_out}")
