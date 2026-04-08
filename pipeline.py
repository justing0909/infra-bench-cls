"""
pipeline.py
-----------
End-to-end infrastructure imagery curation pipeline.

Orchestrates the full sequence:
    1. Extract asset locations from GeoFabrik PBF (sources.py)
    2. Deduplicate spatially proximate assets (deduplication.py)
    3. Fetch imagery tiles — NAIP + Sentinel-2 (imagery.py)
    4. Basic quality control (qc.py)
    5. Confidence triage (triage.py)
    6. Assemble training dataset (dataset.py)

!! CONFIGURE THE SECTION BELOW BEFORE RUNNING !!

Designed to be run as a script for full pipeline runs, or imported
and called programmatically for notebook use.

Usage:
    python pipeline.py                    # full run with config below
    python pipeline.py --dry-run          # print plan, don't fetch imagery
"""

import os
import time
import argparse
import pandas as pd
from datetime import datetime

from sources import GeoFabrikSource
from deduplication import Deduplicator
from imagery import ImageryFetcher
from qc import QualityChecker
from triage import RuleBasedTriager
from dataset import DatasetAssembler


# ===========================================================================
# CONFIGURATION — change these before each run
# ===========================================================================

# --- Input ---
# !! CHANGE PBF_PATH to your downloaded GeoFabrik extract
# Download from https://download.geofabrik.de/
PBF_PATH = "data/pbf/maine-latest.osm.pbf"

# --- Output ---
# !! CHANGE OUTPUT_DIR for each new run to avoid overwriting
# Convention: data/dataset_<region>_<version>/
OUTPUT_DIR = "data/dataset_maine_v1"

# Intermediate CSV paths (auto-generated, but can be overridden)
ASSETS_CSV  = "data/maine_all_assets.csv"
DEDUPED_CSV = "data/maine_deduped_assets.csv"

# --- Asset filtering ---
MIN_CONFIDENCE   = "medium"  # "high", "medium", or "low"
MAX_ASSETS       = None      # set to an int to cap (useful for testing)
SAMPLE_PER_TYPE  = None      # set to an int to sample evenly per asset type

# --- Imagery ---
BUFFER_M = 150               # meters around each asset centroid
SOURCES  = ["sentinel2", "naip"]  # sentinel2 = global, naip = US only

# --- QC thresholds ---
MIN_VALID_RATIO = 0.80
MAX_BRIGHTNESS  = 220
MIN_BRIGHTNESS  = 15

# --- Deduplication ---
DISTANCE_THRESHOLD_M = 200

# --- Triage ---
CONTRADICTION_THRESHOLD = 3
LOW_THRESHOLD           = 1

# ===========================================================================
# Pipeline runner
# ===========================================================================

def run_pipeline(dry_run: bool = False) -> dict:
    """
    Runs the full curation pipeline end to end.

    Parameters
    ----------
    dry_run : bool
        If True, print the plan and asset counts but skip imagery fetching.

    Returns
    -------
    dict of pipeline results and counts
    """
    start_time = time.time()
    results    = {}

    print("=" * 60)
    print("INFRASTRUCTURE IMAGERY CURATION PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Extract assets
    # ------------------------------------------------------------------
    print("\n[1/6] Extracting assets from GeoFabrik PBF...")

    if os.path.exists(ASSETS_CSV):
        print(f"  Loading existing CSV: {ASSETS_CSV}")
        df = pd.read_csv(ASSETS_CSV)
        print(f"  Loaded {len(df)} assets")
    else:
        src = GeoFabrikSource(PBF_PATH, min_confidence=MIN_CONFIDENCE)
        df  = src.extract_all()
        os.makedirs("data", exist_ok=True)
        df.drop(columns=["osm_tags"], errors="ignore").to_csv(ASSETS_CSV, index=False)
        print(f"  Saved {len(df)} assets to {ASSETS_CSV}")

    results["n_extracted"] = len(df)
    print(f"  Asset counts:")
    for asset_type, count in df["asset_type"].value_counts().items():
        print(f"    {asset_type}: {count}")

    # ------------------------------------------------------------------
    # Step 2: Deduplicate
    # ------------------------------------------------------------------
    print("\n[2/6] Deduplicating assets...")

    if os.path.exists(DEDUPED_CSV):
        print(f"  Loading existing CSV: {DEDUPED_CSV}")
        df_clean = pd.read_csv(DEDUPED_CSV)
        print(f"  Loaded {len(df_clean)} deduplicated assets")
    else:
        dedup    = Deduplicator(distance_threshold_m=DISTANCE_THRESHOLD_M)
        df_clean, df_removed = dedup.run(df)
        df_clean.to_csv(DEDUPED_CSV, index=False)
        print(f"  Saved {len(df_clean)} assets to {DEDUPED_CSV}")

    results["n_after_dedup"] = len(df_clean)

    # Optional sampling for test runs
    if SAMPLE_PER_TYPE is not None:
        df_clean = (
            df_clean.groupby("asset_type", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), SAMPLE_PER_TYPE), random_state=42))
            .reset_index(drop=True)
        )
        print(f"  Sampled {len(df_clean)} assets ({SAMPLE_PER_TYPE} per type)")

    if MAX_ASSETS is not None:
        df_clean = df_clean.head(MAX_ASSETS)
        print(f"  Capped at {len(df_clean)} assets")

    if dry_run:
        print("\n[DRY RUN] Stopping before imagery fetch.")
        print(f"  Would fetch tiles for {len(df_clean)} assets")
        print(f"  Sources: {SOURCES}")
        print(f"  Estimated tiles: {len(df_clean) * len(SOURCES)}")
        return results

    # ------------------------------------------------------------------
    # Step 3: Fetch imagery
    # ------------------------------------------------------------------
    print(f"\n[3/6] Fetching imagery tiles ({len(df_clean)} assets × {len(SOURCES)} sources)...")

    fetcher = ImageryFetcher(buffer_m=BUFFER_M, sources=SOURCES)
    tiles   = fetcher.fetch_all(df_clean)

    n_ok    = sum(1 for t in tiles if t.status == "ok")
    n_fail  = sum(1 for t in tiles if t.status != "ok")
    results["n_tiles_fetched"] = n_ok
    results["n_tiles_failed"]  = n_fail
    print(f"  Fetched: {n_ok} ok, {n_fail} failed")

    # ------------------------------------------------------------------
    # Step 4: Quality control
    # ------------------------------------------------------------------
    print("\n[4/6] Running quality control...")

    checker    = QualityChecker(
        min_valid_ratio=MIN_VALID_RATIO,
        max_brightness=MAX_BRIGHTNESS,
        min_brightness=MIN_BRIGHTNESS,
    )
    qc_results = checker.check_all(tiles)
    clean      = checker.filter_ok(qc_results)

    n_qc_pass = len(clean)
    n_qc_fail = len(tiles) - n_qc_pass
    results["n_qc_passed"] = n_qc_pass
    results["n_qc_failed"] = n_qc_fail
    print(f"  QC passed: {n_qc_pass}, failed: {n_qc_fail}")

    # ------------------------------------------------------------------
    # Step 5: Triage
    # ------------------------------------------------------------------
    print("\n[5/6] Running confidence triage...")

    triager        = RuleBasedTriager(
        contradiction_threshold=CONTRADICTION_THRESHOLD,
        low_threshold=LOW_THRESHOLD,
    )
    triage_results = triager.triage_all(clean)
    accepted       = triager.filter_accepted(triage_results)
    flagged        = triager.filter_review(triage_results)

    results["n_accepted"]    = len(accepted)
    results["n_flagged"]     = len(flagged)
    results["n_rejected"]    = n_qc_pass - len(accepted) - len(flagged)
    print(f"  Accepted: {len(accepted)}, flagged: {len(flagged)}, "
          f"rejected: {results['n_rejected']}")

    # ------------------------------------------------------------------
    # Step 6: Assemble dataset
    # ------------------------------------------------------------------
    print(f"\n[6/6] Assembling dataset → {OUTPUT_DIR}...")

    assembler = DatasetAssembler(OUTPUT_DIR)
    summary   = assembler.assemble(accepted, triage_results)

    results["n_dataset_tiles"] = len(summary)
    results["output_dir"]      = OUTPUT_DIR
    results["elapsed_s"]       = round(time.time() - start_time, 1)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Extracted:       {results['n_extracted']:>6} assets")
    print(f"  After dedup:     {results['n_after_dedup']:>6} assets")
    print(f"  Tiles fetched:   {results['n_tiles_fetched']:>6}")
    print(f"  QC passed:       {results['n_qc_passed']:>6}")
    print(f"  Triage accepted: {results['n_accepted']:>6}")
    print(f"  Triage flagged:  {results['n_flagged']:>6}")
    print(f"  Dataset tiles:   {results['n_dataset_tiles']:>6}")
    print(f"  Elapsed:         {results['elapsed_s']}s")
    print(f"  Output:          {OUTPUT_DIR}")

    # Yield rate
    if results["n_tiles_fetched"] > 0:
        yield_rate = results["n_dataset_tiles"] / results["n_tiles_fetched"] * 100
        print(f"  Pipeline yield:  {yield_rate:.1f}% of fetched tiles")

    return results


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infrastructure curation pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without fetching imagery")
    args = parser.parse_args()

    run_pipeline(dry_run=args.dry_run)