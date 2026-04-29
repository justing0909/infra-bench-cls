from __future__ import annotations

import json
import subprocess
from pathlib import Path

PYTHON_EXE     = "python"
PIPELINE_SCRIPT = Path("pipeline.py")

ASSETS_DIR  = Path("../data/PIPELINE/01-extracted-assets")
DEDUPED_DIR = Path("../data/PIPELINE/02-deduped-assets")
OUTPUT_ROOT = Path("../data/curated_datasets")

SKIP_REGIONS = {"europe"}   # add regions to skip here

ASSET_SUFFIX  = "_all_assets_substations.parquet"
DEDUPED_SUFFIX = "_deduped_assets_substations.parquet"

# ---------------------------------------------------------------------------
# Pipeline settings — edit these before a batch run
# ---------------------------------------------------------------------------

# Asset filter preset: "substation" (recommended) or "full"
FILTER_PRESET = "substation"

# Imagery modalities (comma-separated string passed to --modalities)
# Start with sentinel2_ms for a fast first run; add sentinel1 once working.
MODALITIES = "sentinel2_ms,sentinel1"

# Set to True to fetch seasonal temporal stacks (slower, larger output)
TEMPORAL_STACK = False

# GEE is disabled by default — STAC is the primary path
USE_STAC = True
USE_GEE  = False

GEE_PROJECT   = "towards-an-infra-fm"
GEE_COMPOSITE = "median"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def region_from_asset_file(path: Path) -> str:
    return path.name.removesuffix(ASSET_SUFFIX)


def output_dir_for_region(region: str) -> Path:
    suffix = "stac_v1" if USE_STAC else "sentinel_v1"
    return OUTPUT_ROOT / f"dataset_{region}_{suffix}"


def is_done(region: str) -> bool:
    """
    Returns True if the region has a valid _SUCCESS file.
    Reads the JSON metadata so you can inspect what settings were used.
    """
    success_path = output_dir_for_region(region) / "_SUCCESS"
    if not success_path.exists():
        return False
    try:
        with open(success_path) as f:
            meta = json.load(f)
        # Re-run if the filter preset changed since last run
        if meta.get("filter_preset") != FILTER_PRESET:
            print(f"  [rerun] {region}: filter_preset changed "
                  f"({meta.get('filter_preset')} -> {FILTER_PRESET})")
            return False
        return True
    except Exception:
        # Unreadable _SUCCESS — treat as not done
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    asset_files = sorted(ASSETS_DIR.glob(f"*{ASSET_SUFFIX}"))

    if not asset_files:
        print(f"No asset files found in {ASSETS_DIR} matching *{ASSET_SUFFIX}")
        return

    print(f"Found {len(asset_files)} region(s) to process")
    print(f"Settings: filter_preset={FILTER_PRESET}, modalities={MODALITIES}, "
          f"temporal_stack={TEMPORAL_STACK}, use_stac={USE_STAC}\n")

    for asset_file in asset_files:
        region = region_from_asset_file(asset_file)

        if region in SKIP_REGIONS:
            print(f"[skip] {region} (in SKIP_REGIONS)")
            continue

        if is_done(region):
            print(f"[skip] {region} (already complete)")
            continue

        deduped_file = DEDUPED_DIR / f"{region}{DEDUPED_SUFFIX}"
        output_dir   = output_dir_for_region(region)

        cmd = [
            PYTHON_EXE,
            str(PIPELINE_SCRIPT),
            "--assets-table",   str(asset_file),
            "--deduped-table",  str(deduped_file),
            "--output-dir",     str(output_dir),
            "--filter-preset",  FILTER_PRESET,
            "--modalities",     MODALITIES,
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