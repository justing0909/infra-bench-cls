"""
sample_v1.py
------------
Produce v1 Infra-Bench samples: random class-stratified up to 10k assets
per (region, sector) cell, drawn from the deduped parquets in
data/PIPELINE/02-deduped-assets/.

Class stratification is PROPORTIONAL: each class contributes a share of
the sample that matches its proportion of the source parquet. This
preserves the real-world OSM class distribution per the PDF's spec
("class proportions reflecting real-world distributions as recorded in
OSM"). Classes smaller than their proportional allocation are taken in
full; the resulting sample may be slightly smaller than 10k.

Output: data/PIPELINE/03-v1-samples/<region>_<sector>_v1_sample.parquet

Usage:
    python -m curation.sample_v1                 # sample everything that exists
    python -m curation.sample_v1 --target 10000  # change cell size
    python -m curation.sample_v1 --seed 42       # rng seed
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import math

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRE_DIR  = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets")
POST_DIR = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets")
OUT_DIR  = os.path.join(REPO, "data", "PIPELINE", "03-v1-samples")

# 7 continents per Geofabrik. Maine is intentionally excluded — it's a
# gold-validation subset, not one of the v1 continents.
V1_REGIONS = [
    "central-america",
    "australia-oceania",
    "south-america",
    "africa",
    "asia",
    "north-america",
    "europe",
]

# Sectors per the v1 spec. For energy, we prefer the _energy.parquet
# (which covers all 7 energy classes) but fall back to _substations.parquet
# (4 substation subclasses only) if energy isn't available. The fallback
# is flagged in the summary.
SECTORS = ["energy", "water", "transport", "telecom"]
ENERGY_FALLBACK = "substations"


def find_input_parquet(region: str, sector: str) -> tuple[str, bool]:
    """
    Returns (path, used_fallback) for the deduped parquet to sample.
    Raises FileNotFoundError if neither the canonical nor the fallback exists.
    """
    canonical = os.path.join(POST_DIR, f"{region}_deduped_assets_{sector}.parquet")
    if os.path.exists(canonical):
        return canonical, False
    if sector == "energy":
        fallback = os.path.join(POST_DIR, f"{region}_deduped_assets_{ENERGY_FALLBACK}.parquet")
        if os.path.exists(fallback):
            return fallback, True
    raise FileNotFoundError(
        f"No deduped parquet for region={region} sector={sector} "
        f"(checked: {canonical})"
    )


def stratified_sample(
    df: pd.DataFrame,
    target_total: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Proportional class-stratified random sample.

    For each asset_type class:
      - allocation = round(class_size / total_size * target_total)
      - sample min(allocation, class_size) rows uniformly at random
      - (a class smaller than its allocation contributes in full)

    Returns the sampled DataFrame and a list of per-class summary dicts.
    """
    total_n = len(df)
    summary: list[dict] = []
    parts: list[pd.DataFrame] = []

    if total_n == 0:
        return df.iloc[0:0].copy(), summary

    # If the cell is below the target, just return everything — no sampling.
    if total_n <= target_total:
        for cls_name, cls_df in df.groupby("asset_type", sort=False):
            summary.append({
                "asset_type":    cls_name,
                "in_population": len(cls_df),
                "in_sample":     len(cls_df),
                "proportion":    len(cls_df) / total_n,
                "note":          "all (population <= target)",
            })
        return df.copy(), summary

    for cls_name, cls_df in df.groupby("asset_type", sort=False):
        cls_n = len(cls_df)
        # Round to nearest int; small classes get at least 0 (and may end
        # up with 0 if their proportional share rounds down to nothing).
        # That's a known limitation of proportional stratification.
        allocation = int(round(cls_n / total_n * target_total))
        take = min(allocation, cls_n)
        if take <= 0:
            summary.append({
                "asset_type":    cls_name,
                "in_population": cls_n,
                "in_sample":     0,
                "proportion":    cls_n / total_n,
                "note":          "rounded to zero",
            })
            continue
        sampled = cls_df.sample(n=take, random_state=seed)
        parts.append(sampled)
        summary.append({
            "asset_type":    cls_name,
            "in_population": cls_n,
            "in_sample":     take,
            "proportion":    cls_n / total_n,
            "note":          ("all" if take == cls_n else "sampled"),
        })

    if parts:
        return pd.concat(parts, ignore_index=True), summary
    return df.iloc[0:0].copy(), summary


def sample_cell(region: str, sector: str, target: int, seed: int) -> dict:
    try:
        src_path, used_fallback = find_input_parquet(region, sector)
    except FileNotFoundError:
        return {
            "region":         region,
            "sector":         sector,
            "status":         "MISSING_INPUT",
            "input_path":     None,
            "input_rows":     0,
            "sample_rows":    0,
            "fallback":       False,
            "per_class":      [],
            "output_path":    None,
        }

    df = pd.read_parquet(src_path)
    sample_df, per_class = stratified_sample(df, target_total=target, seed=seed)
    out_path = os.path.join(OUT_DIR, f"{region}_{sector}_v1_sample.parquet")
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_df.to_parquet(out_path, index=False)

    return {
        "region":         region,
        "sector":         sector,
        "status":         "ok",
        "input_path":     src_path,
        "input_rows":     len(df),
        "sample_rows":    len(sample_df),
        "fallback":       used_fallback,
        "per_class":      per_class,
        "output_path":    out_path,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=10_000,
                   help="Target sample size per cell (default 10000)")
    p.add_argument("--seed",   type=int, default=42,
                   help="RNG seed for reproducibility")
    p.add_argument("--region", action="append",
                   help="Restrict to specific region(s); default = all 7")
    p.add_argument("--sector", action="append",
                   help="Restrict to specific sector(s); default = all 4")
    args = p.parse_args()

    regions = args.region or V1_REGIONS
    sectors = args.sector or SECTORS

    results = []
    print(f"V1 sampling: target={args.target:,} seed={args.seed}")
    print(f"Regions: {regions}")
    print(f"Sectors: {sectors}")
    print()
    print(f"{'region':<20} {'sector':<11} {'input':>10} {'sample':>8}  notes")
    print("-" * 80)
    for region in regions:
        for sector in sectors:
            r = sample_cell(region, sector, args.target, args.seed)
            results.append(r)
            if r["status"] == "MISSING_INPUT":
                print(f"{region:<20} {sector:<11} {'--':>10} {'--':>8}  no input parquet")
            else:
                tag = " (fallback: substations)" if r["fallback"] else ""
                print(f"{region:<20} {sector:<11} {r['input_rows']:>10,} "
                      f"{r['sample_rows']:>8,}{tag}")

    # Write a summary JSON alongside the parquets.
    summary_path = os.path.join(OUT_DIR, "_v1_sample_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_per_cell": args.target,
            "seed":            args.seed,
            "regions":         regions,
            "sectors":         sectors,
            "results":         results,
        }, f, indent=2, default=str)
    print()
    print(f"Summary -> {summary_path}")

    # Cell coverage summary
    n_total = len(regions) * len(sectors)
    n_ok    = sum(1 for r in results if r["status"] == "ok")
    n_fb    = sum(1 for r in results if r.get("fallback"))
    print(f"\nCells produced: {n_ok}/{n_total}   "
          f"({n_fb} via substations-fallback for energy)")
    missing = [(r["region"], r["sector"]) for r in results if r["status"] != "ok"]
    if missing:
        print(f"Missing cells:")
        for region, sector in missing:
            print(f"  {region}/{sector}")


if __name__ == "__main__":
    main()
