"""
fetch_minimal_example.py
------------------------
Fetches a minimal demonstration imagery dataset for v1: N images per
(country, asset_type) combination per v1 sampled parquet cell.

Purpose: produce a small, representative imagery subset showing the full
file structure working across all sectors and regions. The output is what
lab members will manually validate (per Ed's "1 per asset per country"
spec) and what proves the pipeline end-to-end.

This is a thin wrapper around the existing stac_imagery.STACImageryFetcher
and dataset.DatasetAssembler. Output tiles, manifest, and summary match
the existing curated_datasets convention exactly.

Usage
-----
# Fetch 1 image per (country, asset_type) for all v1 sample parquets:
python fetch_minimal_example.py \\
    --samples-dir data/PIPELINE/03-v1-samples \\
    --output-root data/curated_datasets/minimal_example \\
    --per-country-per-type 1

# Fetch just one cell:
python fetch_minimal_example.py \\
    --samples-dir data/PIPELINE/03-v1-samples \\
    --output-root data/curated_datasets/minimal_example \\
    --per-country-per-type 1 \\
    --only-parquet central-america_energy_v1_sample.parquet

# Per-region instead of per-country (simpler, no geocoding):
python fetch_minimal_example.py \\
    --samples-dir data/PIPELINE/03-v1-samples \\
    --output-root data/curated_datasets/minimal_example \\
    --per-country-per-type 1 \\
    --geography region

Requirements
------------
  pip install reverse_geocoder      # offline lat/lon -> country lookup
  # (everything else from your existing stac pipeline)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import re
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Local imports — assumes this script lives at the repo root or that
# curation/ is on PYTHONPATH.
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, "curation"))

from curation.stac_imagery import STACImageryFetcher
from curation.dataset      import DatasetAssembler


# ---------------------------------------------------------------------------
# Region inference from filename
# ---------------------------------------------------------------------------

# Parquet naming: <region>_<sector>_v1_sample.parquet
# e.g. central-america_energy_v1_sample.parquet
SAMPLE_FILENAME_RE = re.compile(
    r"^(?P<region>[a-z-]+?)_(?P<sector>energy|water|transport|telecom)_v1_sample\.parquet$"
)


def parse_sample_filename(filename: str) -> Optional[dict]:
    """Returns {'region': ..., 'sector': ...} or None if filename doesn't match."""
    m = SAMPLE_FILENAME_RE.match(Path(filename).name)
    if not m:
        return None
    return {"region": m.group("region"), "sector": m.group("sector")}


# ---------------------------------------------------------------------------
# Country lookup (offline reverse geocoding)
# ---------------------------------------------------------------------------

_country_lookup_cache = None


def _get_country_lookup():
    """Lazy-init the reverse_geocoder. ~10MB, loads in ~2s on first call."""
    global _country_lookup_cache
    if _country_lookup_cache is None:
        import reverse_geocoder as rg
        # Loading the package warm-starts its internal data.
        _country_lookup_cache = rg
    return _country_lookup_cache


def add_country_column(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'country' column (ISO-2 code) to df via offline reverse geocoding.

    Vectorized — does a single batch call for the entire df.
    """
    rg = _get_country_lookup()
    coords = list(zip(df["lat"].tolist(), df["lon"].tolist()))
    # mode=2 = fast (single-thread), no progress prints; mode=1 = parallel.
    results = rg.search(coords, mode=2)
    df = df.copy()
    df["country"] = [r["cc"] for r in results]
    return df


def add_region_column(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Just attaches the region name as a geography column (no geocoding)."""
    df = df.copy()
    df["country"] = region   # reuse 'country' as the geography column for downstream grouping
    return df


# ---------------------------------------------------------------------------
# Subsampling per (geography × asset_type)
# ---------------------------------------------------------------------------

def subsample_for_minimal_example(
    df              : pd.DataFrame,
    per_geo_per_type: int,
    seed            : int = 42,
) -> pd.DataFrame:
    """Take up to N rows per (country, asset_type) combination.

    If a (country, asset_type) cell has fewer than N rows, takes all of them.
    Sampling is random within each cell, deterministic by seed.
    """
    if "country" not in df.columns:
        raise ValueError("df must have a 'country' column. Call add_country_column or add_region_column first.")

    sampled = pd.concat([
        g.sample(n=min(per_geo_per_type, len(g)), random_state=seed)
        for _, g in df.groupby(["country", "asset_type"], group_keys=False)
    ], ignore_index=True)
    return sampled


# ---------------------------------------------------------------------------
# Fetch one cell (one parquet)
# ---------------------------------------------------------------------------

def fetch_one_cell(
    parquet_path     : Path,
    output_root      : Path,
    per_geo_per_type : int,
    geography        : str,             # "country" or "region"
    modalities       : list,
    buffer_m         : float,
    workers          : int,
    adaptive         : bool,
) -> dict:
    """Process one v1 sample parquet end-to-end. Returns a per-cell summary dict."""
    parsed = parse_sample_filename(parquet_path.name)
    if not parsed:
        print(f"  [SKIP] {parquet_path.name}: doesn't match v1 naming pattern")
        return {"parquet": parquet_path.name, "status": "skipped_bad_name"}

    region, sector = parsed["region"], parsed["sector"]
    print(f"\n=== {region} / {sector} ===")

    df = pd.read_parquet(parquet_path)
    print(f"  Population in v1 sample: {len(df)}")

    # Attach geography column
    if geography == "country":
        df = add_country_column(df)
    elif geography == "region":
        df = add_region_column(df, region=region)
    else:
        raise ValueError(f"geography must be 'country' or 'region', got {geography!r}")

    # Subsample
    sub = subsample_for_minimal_example(df, per_geo_per_type=per_geo_per_type)
    print(f"  Subsampled to: {len(sub)} assets across {sub['country'].nunique()} {geography}(ies) "
          f"and {sub['asset_type'].nunique()} asset types")

    if len(sub) == 0:
        return {
            "parquet": parquet_path.name,
            "region": region,
            "sector": sector,
            "n_targeted": 0,
            "n_succeeded": 0,
            "status": "empty",
        }

    # Set up the per-cell output directory matching existing convention
    cell_output = output_root / f"dataset_{region}_{sector}_minimal_v1"
    cell_output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cell_output / "stac_checkpoint.pkl"

    fetcher = STACImageryFetcher(
        buffer_m             = buffer_m,
        modalities           = modalities,
        temporal_stack       = False,        # single composite, not temporal stack
        checkpoint_path      = str(checkpoint_path),
        checkpoint_every     = 50,           # smaller batch since the total is small
        adaptive_concurrency = adaptive,
        start_workers        = workers,
        max_workers          = workers * 2,
    )

    t0 = time.time()
    results = fetcher.fetch_all(sub)
    fetch_secs = time.time() - t0

    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = len(results) - n_ok
    print(f"  Fetched: {n_ok} ok, {n_fail} failed in {fetch_secs:.0f}s")

    # Assemble manifest + tile files via the canonical DatasetAssembler
    accepted = [r for r in results if r.status == "ok" and r.image is not None]
    if accepted:
        assembler = DatasetAssembler(output_dir=str(cell_output))
        assembler.assemble(accepted_tiles=accepted, triage_results=None)
    else:
        print(f"  [WARN] No accepted tiles to assemble for {region}/{sector}")

    return {
        "parquet"     : parquet_path.name,
        "region"      : region,
        "sector"      : sector,
        "n_targeted"  : len(sub),
        "n_succeeded" : n_ok,
        "n_failed"    : n_fail,
        "fetch_secs"  : round(fetch_secs, 1),
        "output_dir"  : str(cell_output),
        "status"      : "ok" if n_ok > 0 else "all_failed",
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fetch minimal-example imagery (N per country per asset_type) for v1 demo."
    )
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Directory of v1 sample parquets (data/PIPELINE/03-v1-samples)")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Output root for minimal-example datasets")
    parser.add_argument("--per-country-per-type", type=int, default=1,
                        help="N images per (country, asset_type) combination (default: 1)")
    parser.add_argument("--geography", choices=["country", "region"], default="country",
                        help="Geographic unit for stratification (default: country)")
    parser.add_argument("--only-parquet", type=str, default=None,
                        help="Only process this one parquet (filename, not path)")
    parser.add_argument("--modalities", nargs="+",
                        default=["sentinel2_ms", "sentinel1"],
                        help="STAC modalities (default: sentinel2_ms sentinel1)")
    parser.add_argument("--buffer-m", type=float, default=300,
                        help="Tile buffer in meters (default: 300, => 600x600m tile)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent STAC workers (default: 8)")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive concurrency (default: off for small runs)")
    args = parser.parse_args()

    args.samples_dir = args.samples_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Samples dir: {args.samples_dir}")
    print(f"Output root: {args.output_root}")
    print(f"Per ({args.geography}, asset_type): up to {args.per_country_per_type} images")
    print(f"Modalities:  {args.modalities}")
    print()

    # Discover parquets
    if args.only_parquet:
        parquets = [args.samples_dir / args.only_parquet]
        if not parquets[0].exists():
            print(f"ERROR: {parquets[0]} does not exist", file=sys.stderr)
            sys.exit(1)
    else:
        parquets = sorted(args.samples_dir.glob("*_v1_sample.parquet"))
        if not parquets:
            print(f"ERROR: no *_v1_sample.parquet found in {args.samples_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Processing {len(parquets)} parquet(s).")

    summaries = []
    for p in parquets:
        try:
            s = fetch_one_cell(
                parquet_path     = p,
                output_root      = args.output_root,
                per_geo_per_type = args.per_country_per_type,
                geography        = args.geography,
                modalities       = args.modalities,
                buffer_m         = args.buffer_m,
                workers          = args.workers,
                adaptive         = args.adaptive,
            )
        except Exception as exc:
            print(f"  [ERROR] {p.name}: {exc}")
            traceback.print_exc()
            s = {"parquet": p.name, "status": "error", "error": str(exc)}
        summaries.append(s)

    # Write run summary
    summary_path = args.output_root / "_minimal_example_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"\nSummary written: {summary_path}")

    # Headline counts
    total_target = sum(s.get("n_targeted", 0)  for s in summaries)
    total_ok     = sum(s.get("n_succeeded", 0) for s in summaries)
    print(f"\nTotal targeted: {total_target}")
    print(f"Total succeeded: {total_ok}")
    print(f"Yield: {100*total_ok/total_target:.1f}%" if total_target else "Yield: n/a")


if __name__ == "__main__":
    main()
