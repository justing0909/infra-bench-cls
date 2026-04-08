"""
deduplication.py
----------------
Removes near-duplicate tiles from spatially proximate assets.

When two assets of the same type are very close together, their imagery
tiles will overlap substantially — feeding the model nearly identical
images twice can cause overfitting to that visual pattern.

Strategy: geographic distance between asset centroids.
If two same-type assets are within DISTANCE_THRESHOLD_M meters of each
other, only one is kept (the one with the higher OSM ID, for determinism).
The surviving record notes which asset's tile it represents.

A more sophisticated approach (embedding similarity) is possible later
but requires a trained encoder — geographic distance is the right
pragmatic choice at this stage.

Usage:
    from deduplication import Deduplicator
    dedup = Deduplicator(distance_threshold_m=200)
    clean_df, removed_df = dedup.run(df)   # df from sources.py
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Two assets within this distance (meters) are considered duplicates.
# 200m means their 150m-buffer tiles would overlap by ~50%.
# Increase for larger buffer sizes.
DEFAULT_DISTANCE_THRESHOLD_M = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """
    Returns the great-circle distance in meters between two lat/lon points.
    Uses the Haversine formula — accurate enough for small distances.
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Deduplicator:
    """
    Removes spatially near-duplicate assets from a sources.py DataFrame.

    Parameters
    ----------
    distance_threshold_m : float
        Assets of the same type within this distance are considered
        duplicates. Default 200m works well for a 150m buffer.
        Scale proportionally if you change buffer_m in imagery.py.
    """

    def __init__(self, distance_threshold_m: float = DEFAULT_DISTANCE_THRESHOLD_M):
        self.threshold = distance_threshold_m

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Deduplicates the asset DataFrame.

        Deduplication is applied per asset_type — a substation and a
        solar farm at the same location are NOT considered duplicates.

        Parameters
        ----------
        df : pd.DataFrame — output of GeoFabrikSource.extract_all()

        Returns
        -------
        clean_df   : pd.DataFrame — deduplicated assets (keep these)
        removed_df : pd.DataFrame — removed duplicates with reference to
                     the asset that replaced them
        """
        keep_rows    = []
        removed_rows = []

        for asset_type, group in df.groupby("asset_type"):
            group = group.reset_index(drop=True)
            kept_indices = self._deduplicate_group(group)
            removed_indices = set(range(len(group))) - set(kept_indices)

            keep_rows.append(group.iloc[list(kept_indices)])

            # Record removed assets with reference to nearest kept asset
            if removed_indices:
                removed = group.iloc[list(removed_indices)].copy()
                removed["deduplicated_by"] = removed.apply(
                    lambda row: self._nearest_kept(
                        row, group.iloc[list(kept_indices)]
                    ),
                    axis=1,
                )
                removed_rows.append(removed)

        clean_df = pd.concat(keep_rows, ignore_index=True) if keep_rows else pd.DataFrame()
        removed_df = pd.concat(removed_rows, ignore_index=True) if removed_rows else pd.DataFrame()

        n_removed = len(removed_df)
        n_kept    = len(clean_df)
        print(f"Deduplication complete:")
        print(f"  Kept:    {n_kept}")
        print(f"  Removed: {n_removed} ({n_removed / max(len(df), 1) * 100:.1f}%)")
        if n_removed > 0:
            print(f"\n  Removed by asset type:")
            if not removed_df.empty:
                print(removed_df["asset_type"].value_counts().to_string())

        return clean_df, removed_df

    def _deduplicate_group(self, group: pd.DataFrame) -> list:
        """
        Returns indices of assets to keep within a single asset_type group.
        Uses a greedy approach: iterate through assets, mark any asset
        within threshold distance of an already-kept asset as duplicate.
        """
        lats = group["lat"].values
        lons = group["lon"].values
        n    = len(group)

        kept = []
        suppressed = set()

        for i in range(n):
            if i in suppressed:
                continue
            kept.append(i)
            # suppress everything within threshold of this asset
            for j in range(i + 1, n):
                if j in suppressed:
                    continue
                dist = _haversine_m(lats[i], lons[i], lats[j], lons[j])
                if dist <= self.threshold:
                    suppressed.add(j)

        return kept

    def _nearest_kept(self, removed_row: pd.Series,
                      kept_df: pd.DataFrame) -> str:
        """Returns the asset_id of the nearest kept asset to a removed one."""
        if kept_df.empty:
            return ""
        dists = kept_df.apply(
            lambda r: _haversine_m(
                removed_row["lat"], removed_row["lon"],
                r["lat"], r["lon"]
            ),
            axis=1,
        )
        return kept_df.iloc[dists.argmin()]["asset_id"]

    def summarize(self, clean_df: pd.DataFrame,
                  removed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a summary DataFrame showing kept/removed counts per asset type.
        """
        kept_counts    = clean_df["asset_type"].value_counts().rename("kept")
        removed_counts = removed_df["asset_type"].value_counts().rename("removed") \
                         if not removed_df.empty else pd.Series(dtype=int)

        summary = pd.concat([kept_counts, removed_counts], axis=1).fillna(0).astype(int)
        summary["total"]       = summary["kept"] + summary["removed"]
        summary["removed_pct"] = (summary["removed"] / summary["total"] * 100).round(1)
        return summary.sort_values("total", ascending=False)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    INPUT_CSV  = "data/maine_all_assets.csv"
    OUTPUT_CSV = "data/maine_deduped_assets.csv"

    if not os.path.exists(INPUT_CSV):
        print(f"No CSV found at {INPUT_CSV} — run sources.py first.")
    else:
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} assets from {INPUT_CSV}\n")

        dedup = Deduplicator(distance_threshold_m=200)
        clean_df, removed_df = dedup.run(df)

        print("\nSummary by asset type:")
        print(dedup.summarize(clean_df, removed_df).to_string())

        clean_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(clean_df)} deduplicated assets to {OUTPUT_CSV}")