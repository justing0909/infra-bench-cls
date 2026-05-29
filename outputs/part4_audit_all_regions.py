"""
Audit existing substation parquets for the idx bug.

Strategy:
  - For each region, locate its existing power_only PBF and its existing
    parquet in data/PIPELINE/01-extracted-assets/.
  - For tractable files (< ~200 MB), re-extract substations with the FIXED
    code path and compare to the current parquet count.
  - For larger files, just flag them as suspect without re-running
    (in-memory locations would require many GB of RAM for the OLD buggy
    huge power_only files that retain every source node).
"""
from __future__ import annotations
import os, sys, time

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"

sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

import pandas as pd
from sources import InfraHandler, FILTER_PRESETS

POWER_ONLY_DIR = os.path.join(REPO, "data", "pbf", "power_only")
PARQUET_DIR    = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
SUBSTATION_CLASSES = FILTER_PRESETS["substation"]

# Region -> (power_only filename, parquet filename)
REGIONS = [
    ("maine",             "maine-latest.osm_power_only.osm.pbf",
                          "maine_all_assets_substations.parquet"),
    ("central-america",   "central-america-latest.osm_power_only.osm.pbf",
                          "central-america_all_assets_substations.parquet"),
    ("australia-oceania", "australia-oceania-260408.osm_power_only.osm.pbf",
                          "australia-oceania_all_assets_substations.parquet"),
    ("south-america",     "south-america-260410.osm_power_only.osm.pbf",
                          "south-america_all_assets_substations.parquet"),
    ("africa",            "africa-260408.osm_power_only.osm.pbf",
                          "africa_all_assets_substations.parquet"),
    ("asia",              "asia-260408.osm_power_only.osm.pbf",
                          "asia_all_assets_substations.parquet"),
    ("north-america",     "north-america-latest.osm_power_only.osm.pbf",
                          "north-america_all_assets_substations.parquet"),
    ("europe",            "europe-latest.osm_power_only.osm.pbf",
                          "europe_all_assets_substations.parquet"),
]

# Tractable cutoff: only attempt in-memory extract if the file is reasonably
# small. The OLD buggy power_only PBFs preserve every source node, so a
# 1+ GB file may need 10+ GB of RAM. Set the cutoff conservatively.
TRACTABLE_MB = 100

results = []
print(f"{'region':<22} {'pbf_size':>10} {'parq_size':>10} {'parq_rows':>10} "
      f"{'new_rows':>10} {'delta':>10} {'note'}")
print("-" * 100)
for region, pbf_name, parq_name in REGIONS:
    pbf_path  = os.path.join(POWER_ONLY_DIR, pbf_name)
    parq_path = os.path.join(PARQUET_DIR, parq_name)

    pbf_size_str  = "MISSING"
    parq_size_str = "MISSING"
    pbf_size_mb   = None

    if os.path.exists(pbf_path):
        pbf_size = os.path.getsize(pbf_path)
        pbf_size_mb = pbf_size / 1_048_576
        pbf_size_str = f"{pbf_size_mb:>7.0f} MB"
    if os.path.exists(parq_path):
        parq_size_str = f"{os.path.getsize(parq_path)/1024:>7.0f} KB"

    # Current parquet row count
    parq_rows = None
    if os.path.exists(parq_path):
        try:
            parq_rows = len(pd.read_parquet(parq_path, columns=["asset_id"]))
        except Exception:
            parq_rows = -1

    # Decide whether to re-run
    if pbf_size_mb is None or not os.path.exists(pbf_path):
        note = "no power_only PBF"
        new_rows = None
        delta = None
    elif pbf_size_mb > TRACTABLE_MB:
        note = f"skipped (>{TRACTABLE_MB}MB; old buggy PBF needs idx); suspect"
        new_rows = None
        delta = None
    else:
        # Re-extract with fixed code (no idx)
        h = InfraHandler(active_classes=SUBSTATION_CLASSES)
        t = time.time()
        try:
            h.apply_file(pbf_path, locations=True)
            new_rows = len(h.rows)
            note = f"extract {time.time()-t:.1f}s"
        except Exception as e:
            new_rows = None
            note = f"extract failed: {str(e)[:60]}"

        delta = (new_rows - parq_rows) if (new_rows is not None and parq_rows is not None) else None

    nr_str = f"{new_rows:>10,}" if new_rows is not None else f"{'n/a':>10}"
    pr_str = f"{parq_rows:>10,}" if parq_rows is not None else f"{'n/a':>10}"
    d_str  = f"{delta:>+10,}" if delta is not None else f"{'n/a':>10}"

    print(f"{region:<22} {pbf_size_str:>10} {parq_size_str:>10} {pr_str} {nr_str} {d_str}  {note}")
    results.append((region, pbf_size_mb, parq_rows, new_rows, delta, note))

print()
# Pass / suspect summary
print("Summary:")
tractable_with_data = [r for r in results if r[3] is not None and r[2] is not None]
mismatches = [r for r in tractable_with_data if r[4] != 0]
matches    = [r for r in tractable_with_data if r[4] == 0]
suspect    = [r for r in results if r[5].startswith("skipped")]
missing    = [r for r in results if r[5].startswith("no power_only")]
print(f"  Tractable regions audited: {len(tractable_with_data)}")
print(f"    matches parquet exactly : {len(matches)}  ({[r[0] for r in matches]})")
print(f"    mismatch (suspect bug)  : {len(mismatches)}  "
      f"({[(r[0], r[4]) for r in mismatches]})")
print(f"  Large regions (skipped, suspect): {len(suspect)}  "
      f"({[r[0] for r in suspect]})")
print(f"  Missing PBFs              : {len(missing)}  ({[r[0] for r in missing]})")
