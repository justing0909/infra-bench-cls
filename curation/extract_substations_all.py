"""
extract_substations_all.py
--------------------------
Batch extracts substation assets from power-only PBFs for all regions,
writing fresh _substations parquets to the pipeline directories.

Replaces run_solar_collapse_all.py — solar collapse is no longer needed
now that the pipeline scope is limited to substations.

Pre-filter=False is used throughout since the input PBFs are already
power-only, so the expensive two-pass pre-filter step can be skipped.

Output files:
    data/PIPELINE/01-extracted-assets/<region>_all_assets_substations.parquet
    data/PIPELINE/02-deduped-assets/<region>_deduped_assets_substations.parquet

After running, update JOB_PRESETS in pipeline.py to point to the new
_substations parquet paths, then do a dry-run to confirm counts.

Usage:
    python extract_substations_all.py
    python extract_substations_all.py --regions central-america africa
    python extract_substations_all.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sources import GeoFabrikSource
from deduplication import Deduplicator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POWER_ONLY_DIR = Path("../data/pbf/power_only")
ASSETS_DIR     = Path("../data/PIPELINE/01-extracted-assets")
DEDUPED_DIR    = Path("../data/PIPELINE/02-deduped-assets")

# ---------------------------------------------------------------------------
# Region -> power-only PBF filename
# ---------------------------------------------------------------------------

REGIONS = {
    "central-america":  "central-america-260408.osm_power_only.osm.pbf",
    "australia-oceania":"australia-oceania-260408.osm_power_only.osm.pbf",
    "south-america":    "south-america-260410.osm_power_only.osm.pbf",
    "africa":           "africa-260408.osm_power_only.osm.pbf",
    "asia":             "asia-260408.osm_power_only.osm.pbf",
    "north-america":    "north-america-latest.osm_power_only.osm.pbf",
    "europe":           "europe-latest.osm_power_only.osm.pbf",
    "maine":            "maine-latest.osm_power_only.osm.pbf",
}

# Europe is large — include it explicitly with --regions europe if ready
SKIP_REGIONS = {"europe"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assets_path(region: str) -> Path:
    return ASSETS_DIR / f"{region}_all_assets_substations.parquet"


def deduped_path(region: str) -> Path:
    return DEDUPED_DIR / f"{region}_deduped_assets_substations.parquet"


def process_region(region: str, pbf_path: Path, overwrite: bool) -> bool:
    """
    Extracts and deduplicates substation assets for one region.
    Returns True if successful, False if skipped or failed.
    """
    out_assets  = assets_path(region)
    out_deduped = deduped_path(region)

    if not pbf_path.exists():
        print(f"  [skip] PBF not found: {pbf_path}")
        return False

    if out_deduped.exists() and not overwrite:
        print(f"  [skip] already complete: {out_deduped}")
        return False

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DEDUPED_DIR.mkdir(parents=True, exist_ok=True)

    # Extract substations from power-only PBF
    # pre_filter=False — already power-only, skip the two-pass pre-filter
    src = GeoFabrikSource(
        str(pbf_path),
        min_confidence="medium",
        filter_preset="substation",
        pre_filter=False,
    )
    df = src.extract_all()

    if df.empty:
        print(f"  [warn] no substations found for {region}")
        return False

    # Save assets parquet
    df.drop(columns=["osm_tags"], errors="ignore").to_parquet(
        out_assets, index=False
    )
    print(f"  Assets:  {len(df):>6} -> {out_assets}")

    # Deduplicate
    dedup    = Deduplicator(distance_threshold_m=200)
    df_clean, df_removed = dedup.run(df)
    df_clean.to_parquet(out_deduped, index=False)
    print(f"  Deduped: {len(df_clean):>6} -> {out_deduped}")
    print(f"  Removed: {len(df_removed):>6} duplicates")

    print(f"  Asset type breakdown:")
    for asset_type, count in df_clean["asset_type"].value_counts().items():
        print(f"    {asset_type}: {count}")

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch extract substation assets from power-only PBFs"
    )
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=sorted(REGIONS.keys()),
        help="Regions to process (default: all except SKIP_REGIONS)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if output parquets already exist",
    )
    args = parser.parse_args()

    target_regions = args.regions or [
        r for r in REGIONS if r not in SKIP_REGIONS
    ]

    print(f"Extracting substations for: {target_regions}")
    print(f"Overwrite: {args.overwrite}\n")

    results = {}
    for region in target_regions:
        pbf_filename = REGIONS.get(region)
        if not pbf_filename:
            print(f"[skip] {region}: not in REGIONS map")
            continue

        pbf_path = POWER_ONLY_DIR / pbf_filename

        print(f"\n{'='*50}")
        print(f"Region: {region}")
        print(f"PBF:    {pbf_path}")
        print(f"{'='*50}")

        try:
            success = process_region(region, pbf_path, args.overwrite)
            results[region] = "ok" if success else "skipped"
        except Exception as e:
            print(f"  [error] {region}: {e}")
            results[region] = f"error: {e}"

    # Summary
    print(f"\n{'='*50}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*50}")
    for region, status in results.items():
        print(f"  {region}: {status}")

    errors = [r for r, s in results.items() if s.startswith("error")]
    if errors:
        print(f"\nFailed regions: {errors}")
        sys.exit(1)


if __name__ == "__main__":
    main()