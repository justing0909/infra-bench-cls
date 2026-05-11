"""
sample_north_america_substations.py
------------------------------------
Samples the North America substation parquet down to 25,000 assets for
STAC imagery fetch, preserving the real-world class distribution.

WHY 25,000 (NOT 127,244)
-------------------------
This is a deliberate change from the Europe sampling script. Europe was
originally sampled to 127,244 to "match Asia," but the pipeline's fetch
cap is independently 25,000 — meaning Europe's 127k pre-sample was then
truncated to 25k by `pipeline.py`'s `max_assets` logic. The 127k step
was effectively wasted.

For North America (and going forward for any region exceeding 25k), we
sample directly to 25,000. This:
  - Matches what is actually fetched and stored in the final dataset
  - Avoids the two-stage 127k → 25k truncation that drops tail classes
  - Gives a consistent, single-step methodology that's easy to describe
    in the manuscript

REGIONS AFTER THIS RUN
-----------------------
This script is the second region (after Europe) to use stratified-25k.
Asia was fetched without sampling at 127k deduped → 18,441 surviving,
which represents a different methodology. For the final paper pass,
plan to re-run Asia (and possibly Europe, if reviewer concerns arise)
with the same stratified-25k approach for full consistency.

For this trial pass, the inconsistency is acceptable — the goal is to
shake out the pipeline before the final substation+water+telecom run.

WITHIN-SAMPLE CLASS PRESERVATION
----------------------------------
We preserve the real-world class proportions from the full North America
deduped dataset. North America's typed/untyped split will likely differ
from Europe's (different OSM contributor culture in US/Canada/Mexico),
which is expected and correct — preserving each region's own
distribution is more honest than imposing a global target.

REPRODUCIBILITY
---------------
Fixed seed (42), same as Europe.

Usage:
    cd curation
    python sample_north_america_substations.py

    # Dry run:
    python sample_north_america_substations.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NA_PARQUET = Path(
    "../data/PIPELINE/02-deduped-assets/"
    "north-america_deduped_assets_substations.parquet"
)
NA_SAMPLED = Path(
    "../data/PIPELINE/02-deduped-assets/"
    "north-america_deduped_assets_substations_sampled.parquet"
)

# Target = the pipeline's fetch cap. Sampling to this directly avoids
# the wasted 127k → 25k truncation step that Europe went through.
TARGET_COUNT = 25_000
SEED         = 42


# ---------------------------------------------------------------------------
# Sampling function (identical to Europe — same logic, different region)
# ---------------------------------------------------------------------------

def sample_proportional(
    df: pd.DataFrame,
    target_n: int,
    class_col: str = "asset_type",
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Stratified sampling without replacement, preserving class proportions.
    Each class gets a count proportional to its share in the full dataset.
    """
    rng = np.random.RandomState(seed)

    class_counts      = df[class_col].value_counts()
    class_proportions = class_counts / len(df)

    print(f"\nFull dataset class distribution:")
    for cls, count in class_counts.items():
        pct = class_proportions[cls] * 100
        print(f"  {cls:45s} {count:>8,}  ({pct:.1f}%)")

    target_per_class = {
        cls: int(np.floor(class_proportions[cls] * target_n))
        for cls in class_counts.index
    }
    # Distribute floor-rounding remainder to the largest classes
    allocated = sum(target_per_class.values())
    remainder = target_n - allocated
    for cls in class_counts.index[:remainder]:
        target_per_class[cls] += 1

    print(f"\nTarget sample distribution (target_n={target_n:,}):")
    for cls, count in target_per_class.items():
        pct = count / target_n * 100
        print(f"  {cls:45s} {count:>8,}  ({pct:.1f}%)")

    sampled_parts = []
    for cls, n in target_per_class.items():
        class_df = df[df[class_col] == cls]
        actual_n = min(n, len(class_df))
        if actual_n < n:
            print(f"  Warning: {cls} has only {len(class_df):,} assets, "
                  f"requested {n:,} — using all available")
        sampled = class_df.sample(n=actual_n, random_state=rng, replace=False)
        sampled_parts.append(sampled)

    result = pd.concat(sampled_parts, ignore_index=True)
    # Shuffle so classes aren't grouped together
    result = result.sample(frac=1, random_state=rng).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample North America substation parquet for STAC fetch"
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_COUNT,
        help=f"Target sample size (default: {TARGET_COUNT:,})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sampling plan without saving"
    )
    parser.add_argument(
        "--input", default=str(NA_PARQUET),
        help="Input parquet path"
    )
    parser.add_argument(
        "--output", default=str(NA_SAMPLED),
        help="Output parquet path"
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"North America parquet not found: {input_path}"
        )

    print(f"Loading {input_path.name}...")
    df = pd.read_parquet(input_path)
    print(f"Full dataset: {len(df):,} assets")

    if len(df) <= args.target:
        print(f"\nDataset ({len(df):,}) is already <= target ({args.target:,}). "
              "No sampling needed.")
        if not args.dry_run:
            df.to_parquet(output_path, index=False)
            print(f"Saved as-is to {output_path}")
        return

    reduction_pct = (1 - args.target / len(df)) * 100
    print(f"\nSampling {len(df):,} -> {args.target:,} assets "
          f"({reduction_pct:.1f}% reduction)")
    print(f"Seed: {SEED} (fixed for reproducibility)")

    sampled_df = sample_proportional(df, target_n=args.target, seed=SEED)

    print(f"\nSampled dataset: {len(sampled_df):,} assets")
    print(f"\nFinal class distribution:")
    for cls, count in sampled_df["asset_type"].value_counts().items():
        pct = count / len(sampled_df) * 100
        print(f"  {cls:45s} {count:>8,}  ({pct:.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_parquet(output_path, index=False)
    print(f"\nSaved sampled parquet to: {output_path}")
    print(f"Original preserved at:    {input_path}")
    print(f"\nNext step: ensure pipeline.py's north-america job points to:")
    print(f"  {output_path.name}")


if __name__ == "__main__":
    main()
