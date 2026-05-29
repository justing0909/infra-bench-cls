"""
Re-run dedup on all substation parquets with the fixed (BallTree+haversine)
implementation. Compare against the existing buggy-output parquet counts,
then overwrite with corrected versions.
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

REGIONS = [
    "maine",
    "central-america",
    "australia-oceania",
    "south-america",
    "africa",
    "asia",
    "north-america",
    "europe",
]

results = []
print(f"{'region':<20} {'input':>8} {'old_kept':>10} {'new_kept':>10} {'Δ kept':>8} "
      f"{'old_pct':>8} {'new_pct':>8} {'wall':>6}")
print("-" * 90)
for region in REGIONS:
    pre_path  = os.path.join(PRE_DIR,  f"{region}_all_assets_substations.parquet")
    post_path = os.path.join(POST_DIR, f"{region}_deduped_assets_substations.parquet")
    if not (os.path.exists(pre_path) and os.path.exists(post_path)):
        print(f"{region:<20}  missing parquet(s); skipped")
        continue

    pre_df  = pd.read_parquet(pre_path)
    old_post = pd.read_parquet(post_path, columns=["asset_id"])
    pre_n = len(pre_df)
    old_n = len(old_post)
    old_pct = 100.0 * (pre_n - old_n) / max(pre_n, 1)

    t0 = time.time()
    dedup = Deduplicator(distance_threshold_m=200)
    clean_df, removed_df = dedup.run(pre_df)
    elapsed = time.time() - t0
    new_n = len(clean_df)
    new_pct = 100.0 * (pre_n - new_n) / max(pre_n, 1)

    delta = new_n - old_n
    print(f"{region:<20} {pre_n:>8,} {old_n:>10,} {new_n:>10,} {delta:>+8,} "
          f"{old_pct:>7.2f}% {new_pct:>7.2f}% {elapsed:>5.1f}s")

    # Overwrite the deduped parquet with corrected output
    clean_df.to_parquet(post_path, index=False)

    results.append({
        "region": region, "input": pre_n, "old_kept": old_n, "new_kept": new_n,
        "delta_kept": delta, "old_pct": round(old_pct, 2),
        "new_pct": round(new_pct, 2),
    })

print()
df_res = pd.DataFrame(results)
print("Sorted by median |lat| (already in REGIONS order):")
print(df_res.to_string(index=False))
print()
print("INTERPRETATION")
print("  Δ kept < 0 = bug fix removed additional near-duplicates")
print("  Δ kept = 0 = bug had no measurable impact (typically low-lat regions)")
print("  larger |Δ| at higher latitude regions confirms the bug was lat-dependent")
