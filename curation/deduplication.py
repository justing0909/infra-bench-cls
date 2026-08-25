"""
deduplication.py
----------------
removes near-duplicate tiles from spatially proximate assets.

when two assets of the same type are very close together, their imagery
tiles will overlap substantially — feeding the model nearly identical
images twice can cause overfitting to that visual pattern.

strategy: geographic distance between asset centroids.
if two same-type assets are within DISTANCE_THRESHOLD_M meters of each
other, only one is kept (the one with the higher OSM ID, for determinism).
the surviving record notes which asset's tile it represents.

a more sophisticated approach (embedding similarity) is possible later
but requires a trained encoder — geographic distance is the right
pragmatic choice at this stage.

Usage:
    from curation.deduplication import Deduplicator
    dedup = Deduplicator(distance_threshold_m=200)
    clean_df, removed_df = dedup.run(df)   # df from sources.py
"""

import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# two assets within this distance (meters) are considered duplicates.
# 200m means their 150m-buffer tiles would overlap by ~50%.
# increase for larger buffer sizes.
DEFAULT_DISTANCE_THRESHOLD_M = 200


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float,
                 lat2: float, lon2: float) -> float:
    """
    returns the great-circle distance in meters between two lat/lon points.
    uses the Haversine formula — accurate enough for small distances.
    """
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi  = np.radians(lat2 - lat1)
    dlam  = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class Deduplicator:
    """
    removes spatially near-duplicate assets from a sources.py DataFrame.

    the threshold used for each asset type is determined as follows:
      1. if the asset type has a matching `AssetClass` in `curation.ontology`
         with `dedup_distance_m` set, that value is used.
      2. otherwise, the constructor-supplied `distance_threshold_m` is used
         (which itself defaults to 200m).

    this lets the ontology express per-class scale: subway transfer
    stations want ~50m, solar farms want ~1000m, etc.

    Parameters
    ----------
    distance_threshold_m : float
        fallback threshold for asset types not present in the ontology
        (e.g. legacy types). default 200m matches a 150m buffer.
    """

    def __init__(self, distance_threshold_m: float = DEFAULT_DISTANCE_THRESHOLD_M):
        self.threshold = distance_threshold_m

    @staticmethod
    def _threshold_for(asset_type: str, fallback: float) -> float:
        """
        look up the per-class dedup threshold from the ontology.
        returns the fallback if the type is unknown or the class did
        not declare `dedup_distance_m`.

        import is lazy + tolerant so this module remains usable even if
        the ontology import path can't resolve (e.g. unusual sys.path).
        """
        try:
            from .ontology import get_class_by_name
        except Exception:
            return fallback
        try:
            cls = get_class_by_name(asset_type)
        except KeyError:
            return fallback
        return cls.dedup_distance_m if cls.dedup_distance_m is not None else fallback

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        deduplicates the asset DataFrame.

        deduplication is applied per asset_type — a substation and a
        solar farm at the same location are NOT considered duplicates.
        each asset type uses its own per-class threshold from the
        ontology (falling back to `self.threshold` for unknown types).

        Parameters
        ----------
        df : pd.DataFrame — output of GeoFabrikSource.extract_all()

        Returns
        -------
        clean_df   : pd.DataFrame — deduplicated assets (keep these)
        removed_df : pd.DataFrame — removed duplicates with reference to
                     the asset that replaced them
        """
        keep_rows         = []
        removed_rows      = []
        per_class_summary = []  # (asset_type, n_in, n_kept, threshold_m)

        for asset_type, group in df.groupby("asset_type"):
            group = group.reset_index(drop=True)
            threshold_m = self._threshold_for(asset_type, self.threshold)

            kept_indices = self._deduplicate_group(group, threshold_m)
            removed_indices = set(range(len(group))) - set(kept_indices)

            keep_rows.append(group.iloc[list(kept_indices)])

            # record removed assets with reference to nearest kept asset
            if removed_indices:
                removed = group.iloc[list(removed_indices)].copy()
                kept_subset = group.iloc[list(kept_indices)]
                removed["deduplicated_by"] = removed.apply(
                    lambda row: self._nearest_kept(row, kept_subset),
                    axis=1,
                )
                removed_rows.append(removed)

            per_class_summary.append(
                (asset_type, len(group), len(kept_indices), threshold_m)
            )

        clean_df   = pd.concat(keep_rows,    ignore_index=True) if keep_rows    else pd.DataFrame()
        removed_df = pd.concat(removed_rows, ignore_index=True) if removed_rows else pd.DataFrame()

        n_removed = len(removed_df)
        n_kept    = len(clean_df)
        print(f"Deduplication complete:")
        print(f"  Kept:    {n_kept}")
        print(f"  Removed: {n_removed} "
              f"({n_removed / max(len(df), 1) * 100:.1f}%)")
        print(f"  Per-class breakdown (threshold_m | in -> kept):")
        for asset_type, n_in, n_out, thr in sorted(per_class_summary):
            removed_n = n_in - n_out
            print(f"    {asset_type:42s}  {int(thr):>5d}m | "
                  f"{n_in:>7,} -> {n_out:>7,}  ({removed_n} removed)")
        return clean_df, removed_df

    def _deduplicate_group(self, group: pd.DataFrame, threshold_m: float) -> list:
        """
        returns indices of assets to keep within a single asset_type group.

        uses sklearn's BallTree with the `haversine` metric — this gives
        true great-circle distances, so the meter-valued `threshold_m`
        is applied uniformly regardless of latitude.

        history note:
          a previous implementation here built a `scipy.spatial.KDTree`
          over raw (lat, lon) degree pairs and queried with
          `threshold / 111_320`. that was wrong on two counts:
            (1) longitude degrees shrink by cos(lat), so the conversion
                only held at the equator. at lat 60° the E–W radius
                collapsed to ~50% of intended; at lat 45° to ~70%.
            (2) KDTree's Euclidean metric in degree space treats N–S and
                e–W as commensurate, which is anisotropic away from the
                equator regardless of the threshold conversion.
          both are fixed by switching to BallTree+haversine on radians.
        """
        lats = group["lat"].values
        lons = group["lon"].values
        n    = len(group)

        if n == 1:
            return [0]

        try:
            from sklearn.neighbors import BallTree
            EARTH_RADIUS_M = 6_371_000

            # convert to radians; BallTree haversine requires that.
            coords_rad = np.column_stack([
                np.radians(lats),
                np.radians(lons),
            ])
            tree = BallTree(coords_rad, metric="haversine")

            # threshold in radians (the haversine metric returns distances
            # in radians; multiply by Earth radius to get meters).
            threshold_rad = threshold_m / EARTH_RADIUS_M

            # batch query — for each point, indices of points within radius.
            neighbors = tree.query_radius(coords_rad, r=threshold_rad)

            suppressed = set()
            kept       = []
            for i in range(n):
                if i in suppressed:
                    continue
                kept.append(i)
                for j in neighbors[i]:
                    if j > i:  # only suppress points after current
                        suppressed.add(j)
            return kept

        except ImportError:
            # sklearn not available — fall back to O(n²) with a warning.
            # the fallback uses _haversine_m so it is geographically
            # correct, just slow on large groups.
            print("  Warning: scikit-learn not installed, using slow O(n²) "
                  "dedup. Install with: pip install scikit-learn")
            kept       = []
            suppressed = set()
            for i in range(n):
                if i in suppressed:
                    continue
                kept.append(i)
                for j in range(i + 1, n):
                    if j in suppressed:
                        continue
                    dist = _haversine_m(lats[i], lons[i], lats[j], lons[j])
                    if dist <= threshold_m:
                        suppressed.add(j)
            return kept

    def _nearest_kept(self, removed_row: pd.Series,
                      kept_df: pd.DataFrame) -> str:
        """
        returns the asset_id of the nearest kept asset to a removed one.

        uses BallTree+haversine so the nearest neighbor reported is the
        true geographic nearest, not whatever a Euclidean-on-degrees
        KDTree happens to surface (which can be off by tens of meters
        at mid-to-high latitudes).
        """
        if kept_df.empty:
            return ""
        try:
            from sklearn.neighbors import BallTree
            coords_rad = np.column_stack([
                np.radians(kept_df["lat"].values),
                np.radians(kept_df["lon"].values),
            ])
            tree = BallTree(coords_rad, metric="haversine")
            removed_coord = np.radians([[
                removed_row["lat"], removed_row["lon"],
            ]])
            _, idx = tree.query(removed_coord, k=1)
            return kept_df.iloc[idx[0][0]]["asset_id"]

        except ImportError:
            dists = kept_df.apply(
                lambda r: _haversine_m(
                    removed_row["lat"], removed_row["lon"],
                    r["lat"], r["lon"],
                ),
                axis=1,
            )
            return kept_df.iloc[dists.argmin()]["asset_id"]

    def summarize(self, clean_df: pd.DataFrame,
                  removed_df: pd.DataFrame) -> pd.DataFrame:
        """
        returns a summary DataFrame showing kept/removed counts per asset type.
        """
        kept_counts    = clean_df["asset_type"].value_counts().rename("kept")
        removed_counts = removed_df["asset_type"].value_counts().rename("removed") \
                         if not removed_df.empty else pd.Series(dtype=int)

        summary = pd.concat([kept_counts, removed_counts], axis=1).fillna(0).astype(int)
        summary["total"]       = summary["kept"] + summary["removed"]
        summary["removed_pct"] = (summary["removed"] / summary["total"] * 100).round(1)
        return summary.sort_values("total", ascending=False)


# ---------------------------------------------------------------------------
# quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    INPUT_CSV  = "data/us-northeast_all_assets.csv"
    OUTPUT_CSV = "data/us-northeast_deduped_assets.csv"

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