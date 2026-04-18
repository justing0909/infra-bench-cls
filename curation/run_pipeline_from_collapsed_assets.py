from __future__ import annotations

import subprocess
from pathlib import Path

PYTHON_EXE = "python"
PIPELINE_SCRIPT = Path("pipeline.py")

ASSETS_DIR = Path("data/PIPELINE/01-extracted-assets")
DEDUPED_DIR = Path("data/PIPELINE/02-deduped-assets")
OUTPUT_ROOT = Path("data/curated_datasets")

SKIP_REGIONS = {"europe"}

ASSET_SUFFIX = "_all_assets_collapsed.parquet"
DEDUPED_SUFFIX = "_deduped_assets.parquet"


def region_from_asset_file(path: Path) -> str:
    return path.name.removesuffix(ASSET_SUFFIX)


def output_dir_for_region(region: str) -> Path:
    return OUTPUT_ROOT / f"dataset_{region}_sentinel_v1"


def is_done(region: str) -> bool:
    out_dir = output_dir_for_region(region)
    return (out_dir / "_SUCCESS").exists()


def main() -> None:
    asset_files = sorted(ASSETS_DIR.glob(f"*{ASSET_SUFFIX}"))

    for asset_file in asset_files:
        region = region_from_asset_file(asset_file)

        if region in SKIP_REGIONS:
            print(f"[skip] {region} (manual skip)")
            continue

        if is_done(region):
            print(f"[skip] {region} (already complete)")
            continue

        deduped_file = DEDUPED_DIR / f"{region}{DEDUPED_SUFFIX}"
        output_dir = output_dir_for_region(region)

        cmd = [
            PYTHON_EXE,
            str(PIPELINE_SCRIPT),
            "--assets-table", str(asset_file),
            "--deduped-table", str(deduped_file),
            "--output-dir", str(output_dir),
            "--sources", "sentinel2",
            "--use-gee",
            "--gee-project", "towards-an-infra-fm",
            "--gee-composite", "median",
        ]

        print(f"\n[run] {region}")
        print(" ".join(cmd))

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(f"Pipeline failed for {region} with exit code {result.returncode}")


if __name__ == "__main__":
    main()