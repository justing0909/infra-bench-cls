from __future__ import annotations

# Changes from original:
#   - DEDUPED_SUFFIX updated to "_deduped_assets_substations.parquet" to match
#     the actual filenames produced by extract_substations_all.py and Jack's runs.
#     The original "_deduped_assets.parquet" suffix was from an earlier naming
#     convention and no longer matches what's on disk.
#   - Europe removed from SKIP_REGIONS — it's now handled via a sampled parquet
#     (see REGION_DEDUPED_OVERRIDES below and sample_europe_substations.py).
#   - Added REGION_DEDUPED_OVERRIDES dict — allows per-region deduped parquet
#     overrides. Used for Europe to point at the sampled version (~127k assets)
#     rather than the full 505k. Apply the same pattern to any future region
#     that needs sampling (e.g. if North America or Asia also prove too large).
#   - Added REGION_ASSET_COUNTS for documentation — shows expected asset counts
#     per region so you can quickly sanity-check what's being processed.
#   - ASSETS_DIR lookup now falls back gracefully if no asset file is found for
#     a region, printing a clear message rather than silently skipping.

import json
import subprocess
from pathlib import Path

PYTHON_EXE      = "python"
PIPELINE_SCRIPT = Path("pipeline.py")

ASSETS_DIR  = Path("../data/PIPELINE/01-extracted-assets")
DEDUPED_DIR = Path("../data/PIPELINE/02-deduped-assets")
OUTPUT_ROOT = Path("../data/curated_datasets")

# Regions to skip entirely — add a region here if its parquet isn't ready yet.
# Europe is NO LONGER skipped — it uses the sampled parquet via the override below.
SKIP_REGIONS = set()

# Standard suffix for deduped substations parquets.
# Matches output of extract_substations_all.py and Jack's extraction runs.
DEDUPED_SUFFIX = "_deduped_assets_substations.parquet"

# ---------------------------------------------------------------------------
# Per-region deduped parquet overrides
# ---------------------------------------------------------------------------
# Use this to point specific regions at non-standard parquet files.
#
# WHY EUROPE IS OVERRIDDEN:
#   Europe has 505,951 deduped substations — far larger than any other region.
#   At ~0.5 tiles/second fetch rate, the full dataset would take ~88 hours.
#   We sample it down to ~127,244 assets (matching Asia, our next-largest region)
#   using sample_europe_substations.py, which preserves the real-world class
#   distribution (91.8% untyped, 6.2% DX, 2.0% TX).
#
#   See sample_europe_substations.py for full rationale and methodology.
#   The sampled parquet is saved as europe_deduped_assets_substations_sampled.parquet
#   and the original is preserved as europe_deduped_assets_substations.parquet.
#
# APPLYING TO OTHER REGIONS:
#   If North America or Asia also prove too large in future runs, add them here:
#   "north-america": DEDUPED_DIR / "north-america_deduped_assets_substations_sampled.parquet",
#
REGION_DEDUPED_OVERRIDES: dict[str, Path] = {
    "europe": DEDUPED_DIR / "europe_deduped_assets_substations_sampled.parquet",
}

# ---------------------------------------------------------------------------
# Expected asset counts per region (for reference / sanity checking)
# ---------------------------------------------------------------------------
# These are approximate post-deduplication counts.
# Update as new regions are added or re-extracted.
REGION_ASSET_COUNTS = {
    "central-america":   1_917,
    "australia-oceania": 5_583,
    "africa":           11_044,
    "south-america":    21_240,
    "maine":               372,
    "asia":            127_244,
    "europe":          127_244,   # sampled — full dataset is 505,951
    "north-america":      None,   # pending Jack's extraction
}

# ---------------------------------------------------------------------------
# Pipeline settings — edit these before each batch run
# ---------------------------------------------------------------------------

# Asset filter preset: "substation" (recommended) or "full"
FILTER_PRESET = "substation"

# Imagery modalities (comma-separated string passed to --modalities)
# sentinel2_ms + sentinel1 = 9 bands, consistent across all regions.
# Do NOT add landsat_thermal here — central america was re-fetched without it
# to maintain 9-band consistency across all regions.
MODALITIES = "sentinel2_ms,sentinel1"

# Set to True to fetch seasonal temporal stacks (slower, larger output)
TEMPORAL_STACK = False

# STAC (Planetary Computer) is the primary imagery path
USE_STAC = True
USE_GEE  = False

GEE_PROJECT   = "towards-an-infra-fm"
GEE_COMPOSITE = "median"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deduped_file_for_region(region: str) -> Path:
    """
    Returns the deduped parquet path for a region, respecting overrides.
    Europe uses the sampled version; all others use the standard naming.
    """
    if region in REGION_DEDUPED_OVERRIDES:
        return REGION_DEDUPED_OVERRIDES[region]
    return DEDUPED_DIR / f"{region}{DEDUPED_SUFFIX}"


def output_dir_for_region(region: str) -> Path:
    suffix = "stac_v1" if USE_STAC else "sentinel_v1"
    return OUTPUT_ROOT / f"dataset_{region}_{suffix}"


def is_done(region: str) -> bool:
    """
    Returns True if the region has a valid _SUCCESS file.
    Checks that the filter_preset matches current settings — if it changed,
    the region will be re-run even if _SUCCESS exists.
    """
    success_path = output_dir_for_region(region) / "_SUCCESS"
    if not success_path.exists():
        return False
    try:
        with open(success_path) as f:
            meta = json.load(f)
        if meta.get("filter_preset") != FILTER_PRESET:
            print(f"  [rerun] {region}: filter_preset changed "
                  f"({meta.get('filter_preset')} -> {FILTER_PRESET})")
            return False
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Discover all regions with deduped parquets available
    # Check both standard and override paths
    regions_to_run = []

    # Collect all regions from standard deduped files
    standard_files = sorted(DEDUPED_DIR.glob(f"*{DEDUPED_SUFFIX}"))
    for f in standard_files:
        # Skip sampled variants — those are accessed via overrides, not directly
        if "_sampled" in f.name:
            continue
        region = f.name.removesuffix(DEDUPED_SUFFIX)
        regions_to_run.append(region)

    # Add any override regions not already discovered
    for region in REGION_DEDUPED_OVERRIDES:
        if region not in regions_to_run:
            regions_to_run.append(region)

    # Sort for consistent ordering (smallest first is good practice)
    regions_to_run = sorted(set(regions_to_run))

    if not regions_to_run:
        print(f"No deduped parquets found in {DEDUPED_DIR}")
        print(f"Run extract_substations_all.py first.")
        return

    print(f"Found {len(regions_to_run)} region(s) to process:")
    for region in regions_to_run:
        expected = REGION_ASSET_COUNTS.get(region)
        count_str = f"~{expected:,}" if expected else "unknown"
        override = " (sampled)" if region in REGION_DEDUPED_OVERRIDES else ""
        print(f"  {region:25s} {count_str:>10} assets{override}")

    print(f"\nSettings: filter_preset={FILTER_PRESET}, modalities={MODALITIES}, "
          f"temporal_stack={TEMPORAL_STACK}, use_stac={USE_STAC}\n")

    for region in regions_to_run:
        if region in SKIP_REGIONS:
            print(f"[skip] {region} (in SKIP_REGIONS)")
            continue

        if is_done(region):
            print(f"[skip] {region} (already complete)")
            continue

        deduped = deduped_file_for_region(region)

        if not deduped.exists():
            print(f"[skip] {region} — deduped parquet not found: {deduped}")
            print(f"         Run sample_europe_substations.py first for europe,")
            print(f"         or extract_substations_all.py for other regions.")
            continue

        output_dir = output_dir_for_region(region)

        cmd = [
            PYTHON_EXE,
            str(PIPELINE_SCRIPT),
            "--deduped-table", str(deduped),
            "--output-dir",    str(output_dir),
            "--filter-preset", FILTER_PRESET,
            "--modalities",    MODALITIES,
        ]

        if TEMPORAL_STACK:
            cmd.append("--temporal-stack")
        else:
            cmd.append("--no-temporal-stack")

        if USE_STAC:
            cmd.append("--use-stac")
        elif USE_GEE:
            cmd += [
                "--no-use-stac",
                "--use-gee",
                "--gee-project",   GEE_PROJECT,
                "--gee-composite", GEE_COMPOSITE,
            ]

        print(f"\n[run] {region}")
        print(" ".join(cmd))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(
                f"Pipeline failed for {region} with exit code {result.returncode}"
            )

    print("\nAll regions processed.")


if __name__ == "__main__":
    main()