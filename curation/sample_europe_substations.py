"""
sample_europe_substations.py
-----------------------------
Samples the Europe substation parquet down to a manageable size for
STAC imagery fetch, while preserving the real-world class distribution.

WHY WE SAMPLE EUROPE
---------------------
Europe's OSM substation coverage is exceptionally dense — 505,951 assets
after deduplication, compared to:
  - Asia:           127,244
  - South America:   21,240 (full fetch, ~10h)
  - Africa:          11,044 (full fetch, ~8h)
  - Australia-Oceania: 5,583 (full fetch, ~4h)
  - Central America:  1,917 (full fetch, ~3h)

At our observed fetch rate of ~0.5 tiles/second with adaptive concurrency,
505k Europe assets would take ~88 hours of continuous fetching across
~8 Colab sessions. This is impractical for the paper timeline.

SAMPLING STRATEGY: MATCH ASIA
------------------------------
We cap Europe at ~127,000 assets to match Asia — our largest other region.
The rationale is defensible and consistent:

  "We capped each continental extract at a comparable scale to prevent
   any single region from dominating the pretraining corpus. Europe and
   Asia, as the two largest extracts, are both capped at ~127k assets."

This is NOT arbitrary — it's a deliberate choice to ensure geographic
balance in the pretraining dataset. A model trained on 500k European
substations and 20k from everywhere else would learn European infrastructure
patterns, not global ones.

WITHIN-SAMPLE CLASS PRESERVATION
----------------------------------
We preserve the real-world class proportions from the full Europe dataset:
  - energy.distribution.substation_untyped: ~91.8% of assets
  - energy.distribution.substation:          ~6.2%
  - energy.transmission.substation:          ~2.0%

This is important because these proportions reflect actual OSM tagging
behavior in Europe. Oversampling typed substations would make the dataset
look cleaner than reality — the untyped majority is a real finding about
OSM data quality that the model should be exposed to.

REPRODUCIBILITY
---------------
A fixed random seed (42) ensures the same sample is drawn every time.
The sampled parquet is saved alongside the original so the full dataset
is never lost.

FUTURE CONSISTENCY
------------------
For subsequent runs (other sectors, other asset types), apply the same
logic:
  1. Identify the "reference region" — the largest non-Europe region
     (currently Asia at ~127k)
  2. If a region exceeds 2x the reference, sample down to match
  3. Always preserve within-sample class proportions
  4. Always use seed=42 for reproducibility
  5. Document the sampling decision in the parquet filename

This ensures that as we expand to water/telecom/transport assets, the
sampling approach remains consistent and defensible across the paper.

Usage:
    cd curation
    python sample_europe_substations.py

    # Or with custom target count:
    python sample_europe_substations.py --target 50000

    # Dry run to see what would be sampled:
    python sample_europe_substations.py --dry-run
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Input: full Europe deduped parquet from Jack
EUROPE_PARQUET = Path("../data/PIPELINE/02-deduped-assets/europe_deduped_assets_substations.parquet")

# Output: sampled parquet — saved alongside original, never overwrites it
EUROPE_SAMPLED = Path("../data/PIPELINE/02-deduped-assets/europe_deduped_assets_substations_sampled.parquet")

# Reference region for target count
# Asia is our largest other region at 127,244 assets
# We match Europe to this so no single continent dominates pretraining
REFERENCE_REGION = "asia"
REFERENCE_COUNT  = 127_244

# Random seed for reproducibility
SEED = 42


# ---------------------------------------------------------------------------
# Sampling function
# ---------------------------------------------------------------------------

def sample_proportional(
    df: pd.DataFrame,
    target_n: int,
    class_col: str = "asset_type",
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Samples a DataFrame down to target_n rows while preserving the
    original class proportions as closely as possible.

    This is stratified sampling without replacement. Each class gets
    a number of samples proportional to its share in the full dataset.

    For example, if untyped substations are 91.8% of the full dataset,
    they will be ~91.8% of the sampled dataset. This preserves the
    real-world distribution rather than artificially balancing classes.

    Parameters
    ----------
    df       : full DataFrame to sample from
    target_n : desired sample size
    class_col: column containing class labels
    seed     : random seed for reproducibility

    Returns
    -------
    Sampled DataFrame with target_n rows (or fewer if a class has
    fewer assets than its proportional allocation).
    """
    rng = np.random.RandomState(seed)

    # Compute per-class proportions from the full dataset
    class_counts = df[class_col].value_counts()
    class_proportions = class_counts / len(df)

    print(f"\nFull dataset class distribution:")
    for cls, count in class_counts.items():
        pct = class_proportions[cls] * 100
        print(f"  {cls:45s} {count:>8,}  ({pct:.1f}%)")

    # Compute target count per class
    # Use floor to avoid going over target_n due to rounding
    target_per_class = {}
    for cls in class_counts.index:
        target_per_class[cls] = int(np.floor(class_proportions[cls] * target_n))

    # Distribute any remaining slots (due to floor rounding) to largest classes
    allocated = sum(target_per_class.values())
    remainder = target_n - allocated
    for cls in class_counts.index[:remainder]:
        target_per_class[cls] += 1

    print(f"\nTarget sample distribution (target_n={target_n:,}):")
    for cls, count in target_per_class.items():
        pct = count / target_n * 100
        print(f"  {cls:45s} {count:>8,}  ({pct:.1f}%)")

    # Sample each class
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
        description="Sample Europe substation parquet for STAC fetch"
    )
    parser.add_argument(
        "--target", type=int, default=REFERENCE_COUNT,
        help=f"Target sample size (default: {REFERENCE_COUNT:,} to match {REFERENCE_REGION})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sampling plan without saving"
    )
    parser.add_argument(
        "--input", default=str(EUROPE_PARQUET),
        help="Input parquet path"
    )
    parser.add_argument(
        "--output", default=str(EUROPE_SAMPLED),
        help="Output parquet path"
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Europe parquet not found: {input_path}\n"
            "Run extract_substations_all.py or get the file from Jack."
        )

    # Load full dataset
    print(f"Loading {input_path.name}...")
    df = pd.read_parquet(input_path)
    print(f"Full dataset: {len(df):,} assets")

    # Decide whether to sample
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
    print(f"Reference region: {REFERENCE_REGION} ({REFERENCE_COUNT:,} assets)")
    print(f"Seed: {SEED} (fixed for reproducibility)")

    # Sample
    sampled_df = sample_proportional(df, target_n=args.target, seed=SEED)

    print(f"\nSampled dataset: {len(sampled_df):,} assets")
    print(f"\nFinal class distribution:")
    for cls, count in sampled_df["asset_type"].value_counts().items():
        pct = count / len(sampled_df) * 100
        print(f"  {cls:45s} {count:>8,}  ({pct:.1f}%)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_parquet(output_path, index=False)
    print(f"\nSaved sampled parquet to: {output_path}")
    print(f"Original preserved at:    {input_path}")
    print(f"\nNext step: update run_pipeline_from_collapsed_assets.py to use")
    print(f"  {output_path.name}")
    print(f"instead of the full europe parquet.")


if __name__ == "__main__":
    main()