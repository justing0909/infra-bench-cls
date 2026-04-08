"""
gee_imagery.py
--------------
Google Earth Engine imagery fetcher for infrastructure asset tiles.

Replaces the Planetary Computer / rasterio approach in imagery.py with
server-side GEE processing. All clipping and compositing happens on
Google's infrastructure — only the final small tile arrays are transferred
to your machine. This is the correct approach for global scale.

Why GEE over Planetary Computer for global runs:
  - No per-request network overhead — computation is server-side
  - Built-in cloud masking via QA60 band
  - Handles temporal compositing (median of cloud-free pixels) natively
  - Free for non-commercial research use
  - No rate limit issues at the scale we need

Setup (one time):
  1. Register at https://code.earthengine.google.com
  2. Create a Google Cloud project at https://console.cloud.google.com
  3. Enable the Earth Engine API for your project
  4. Run: python -c "import ee; ee.Authenticate()"
  5. Note your project ID (e.g. "ee-yourname")

Usage:
    from gee_imagery import GEEImageryFetcher, GEETileResult
    fetcher  = GEEImageryFetcher(project="ee-yourname")
    results  = fetcher.fetch_all(df)
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict
from tile_types import TileResult, _centroid_to_bbox


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Harmonized Sentinel-2 L2A — atmospherically corrected, DN-shift handled
S2_COLLECTION   = "COPERNICUS/S2_SR_HARMONIZED"

# RGB bands at 10m native resolution
S2_RGB_BANDS    = ["B4", "B3", "B2"]

# Date range for scene search
S2_DATE_START   = "2021-01-01"
S2_DATE_END     = "2024-12-31"

# Cloud cover filter — scenes above this % are excluded from compositing
S2_MAX_CLOUD    = 20

# Output resolution in meters
S2_SCALE        = 10

# Output CRS
S2_CRS          = "EPSG:4326"

# Max concurrent GEE requests — stay conservative to avoid quota issues
# Increase carefully; GEE free tier has limits on concurrent operations
MAX_CONCURRENT  = 10


# ---------------------------------------------------------------------------
# GEE tile result — extends TileResult with GEE-specific metadata
# ---------------------------------------------------------------------------

@dataclass
class GEETileResult(TileResult):
    """
    TileResult subclass with GEE-specific fields.
    Fully compatible with qc.py, triage.py, dataset.py.
    """
    n_scenes_composited: int = 0   # how many scenes went into the composite
    cloud_masked:        bool = True


# ---------------------------------------------------------------------------
# Cloud masking function
# ---------------------------------------------------------------------------

def _get_masked_collection(ee, lat: float, lon: float,
                            buffer_m: float) -> object:
    """
    Returns a cloud-masked, normalized Sentinel-2 ImageCollection
    filtered to the asset's location and date range.

    Applies two layers of cloud filtering:
      1. CLOUDY_PIXEL_PERCENTAGE metadata filter (scene-level)
      2. QA60 bitmask (pixel-level cloud + cirrus mask)
    """
    def mask_s2_clouds(image):
        qa          = image.select("QA60")
        cloud_mask  = 1 << 10
        cirrus_mask = 1 << 11
        mask = (qa.bitwiseAnd(cloud_mask).eq(0)
                  .And(qa.bitwiseAnd(cirrus_mask).eq(0)))
        # divide by 10000 to normalize reflectance to 0-1
        return image.updateMask(mask).divide(10000)

    point = ee.Geometry.Point([lon, lat])
    bbox  = ee.Geometry.BBox(
        lon - buffer_m / 111320,
        lat - buffer_m / 111320,
        lon + buffer_m / 111320,
        lat + buffer_m / 111320,
    )

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterDate(S2_DATE_START, S2_DATE_END)
        .filterBounds(bbox)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", S2_MAX_CLOUD))
        .map(mask_s2_clouds)
    )

    return collection, bbox


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------

class GEEImageryFetcher:
    """
    Fetches Sentinel-2 imagery tiles using Google Earth Engine.

    Parameters
    ----------
    project    : str   — your GEE/Google Cloud project ID
                         e.g. "ee-yourname" or "my-research-project"
    buffer_m   : float — meters around each asset centroid (default 150m)
                         consider 300m for GEE since compute is free
    composite  : str   — how to handle multiple cloud-free scenes:
                         "median"  — median composite (most cloud-robust)
                         "mosaic"  — most recent valid pixel
                         "best"    — single least-cloudy scene
    scale      : int   — output resolution in meters (default 10 = native S2)
    checkpoint_path : str — path to checkpoint file for resumable runs
    checkpoint_every: int — save checkpoint every N assets
    """

    def __init__(
        self,
        project: str,
        buffer_m: float = 150,
        composite: str = "median",
        scale: int = S2_SCALE,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 100,
    ):
        self.project          = project
        self.buffer_m         = buffer_m
        self.composite        = composite
        self.scale            = scale
        self.checkpoint_path  = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self._ee              = None

    def _init_ee(self):
        """Initializes Earth Engine (lazy — only on first use)."""
        if self._ee is not None:
            return self._ee
        try:
            import ee
            ee.Initialize(project=self.project)
            self._ee = ee
            print(f"  GEE initialized (project={self.project})")
            return ee
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize Google Earth Engine: {e}\n"
                f"Make sure you have run: python -c \"import ee; ee.Authenticate()\"\n"
                f"And that your project ID '{self.project}' is correct."
            )

    def fetch_tile(self, asset_row: pd.Series) -> GEETileResult:
        """
        Fetches a Sentinel-2 tile for a single asset using GEE.

        The composite strategy controls how multiple cloud-free scenes
        are merged:
          - "median"  recommended for training data — robust to outliers
          - "mosaic"  most recent valid pixel — preserves temporal recency
          - "best"    single least-cloudy scene — simplest, fastest
        """
        ee = self._init_ee()

        lat        = float(asset_row["lat"])
        lon        = float(asset_row["lon"])
        asset_id   = str(asset_row["asset_id"])
        asset_type = str(asset_row["asset_type"])
        bbox_tuple = _centroid_to_bbox(lat, lon, self.buffer_m)

        result = GEETileResult(
            asset_id=asset_id, asset_type=asset_type,
            lat=lat, lon=lon, bbox=bbox_tuple,
            source="sentinel2_gee",
        )

        try:
            collection, ee_bbox = _get_masked_collection(
                ee, lat, lon, self.buffer_m
            )

            # Check how many scenes are available
            n_scenes = collection.size().getInfo()
            if n_scenes == 0:
                result.status    = "no_scene"
                result.error_msg = "No cloud-free Sentinel-2 scenes found"
                return result

            result.n_scenes_composited = n_scenes

            # Build composite image
            if self.composite == "median":
                image = collection.select(S2_RGB_BANDS).median()
            elif self.composite == "mosaic":
                image = (collection
                         .sort("CLOUDY_PIXEL_PERCENTAGE")
                         .select(S2_RGB_BANDS)
                         .mosaic())
            else:  # "best"
                image = (collection
                         .sort("CLOUDY_PIXEL_PERCENTAGE")
                         .first()
                         .select(S2_RGB_BANDS))

            # Clip to asset bbox
            image = image.clip(ee_bbox)

            # Download as numpy array
            # getDownloadURL is fine for small tiles at our scale
            arr = image.getDownloadArray(
                scale=self.scale,
                crs=S2_CRS,
                region=ee_bbox,
            )

            # arr shape: (rows, cols, bands) → convert to (bands, rows, cols)
            arr = np.array(arr)
            if arr.ndim == 3:
                arr = np.moveaxis(arr, -1, 0)

            # Normalize to uint8 (values are 0-1 after divide(10000))
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

            if arr.shape[1] == 0 or arr.shape[2] == 0:
                result.status    = "empty_crop"
                result.error_msg = "GEE returned empty array"
                return result

            result.image      = arr
            result.image_date = S2_DATE_END  # composite has no single date
            result.status     = "ok"

        except Exception as e:
            result.status    = "error"
            result.error_msg = str(e)

        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            import pickle
            with open(self.checkpoint_path, "rb") as f:
                data = pickle.load(f)
            print(f"  Resuming from checkpoint: "
                  f"{len(data['results'])} tiles already fetched")
            return data
        return {"results": [], "completed_ids": set()}

    def _save_checkpoint(self, results: list, completed_ids: set) -> None:
        if not self.checkpoint_path:
            return
        import pickle
        os.makedirs(
            os.path.dirname(self.checkpoint_path)
            if os.path.dirname(self.checkpoint_path) else ".",
            exist_ok=True,
        )
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(
                {"results": results, "completed_ids": completed_ids}, f
            )

    def _clear_checkpoint(self) -> None:
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Batch fetch with checkpointing
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        df: pd.DataFrame,
        max_assets: Optional[int] = None,
    ) -> List[GEETileResult]:
        """
        Fetches GEE tiles for all assets with checkpointing.

        For large global runs, set checkpoint_path so crashes
        can be resumed without starting over.

        Parameters
        ----------
        df          : DataFrame from GeoFabrikSource
        max_assets  : cap for testing
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        if max_assets is not None:
            df = df.head(max_assets)

        checkpoint    = self._load_checkpoint()
        all_results   = checkpoint["results"]
        completed_ids = checkpoint["completed_ids"]

        pending = df[~df["asset_id"].isin(completed_ids)].copy()
        total   = len(df)

        print(f"  GEE fetch: {len(pending)} assets pending "
              f"({len(completed_ids)} from checkpoint)")

        rows = [row for _, row in pending.iterrows()]
        lock = threading.Lock()

        def fetch_one(row):
            return self.fetch_tile(row)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {executor.submit(fetch_one, row): row["asset_id"]
                       for row in rows}

            for future in as_completed(futures):
                result = future.result()
                with lock:
                    all_results.append(result)
                    completed_ids.add(result.asset_id)

                    n_done = len(completed_ids)
                    if n_done % self.checkpoint_every == 0:
                        self._save_checkpoint(all_results, completed_ids)
                        ok  = sum(1 for r in all_results if r.status == "ok")
                        pct = n_done / total * 100
                        print(f"  [{n_done}/{total} ({pct:.0f}%)] "
                              f"ok={ok} | checkpoint saved")

        self._clear_checkpoint()
        ok_total = sum(1 for r in all_results if r.status == "ok")
        print(f"\n  GEE fetch complete: {ok_total} ok / "
              f"{len(all_results) - ok_total} failed")
        return all_results

    def summarize(self, results: List[GEETileResult]) -> pd.DataFrame:
        """Same interface as ImageryFetcher.summarize()."""
        rows = []
        for r in results:
            rows.append({
                "asset_id":            r.asset_id,
                "asset_type":          r.asset_type,
                "source":              r.source,
                "lat":                 r.lat,
                "lon":                 r.lon,
                "image_date":          r.image_date,
                "status":              r.status,
                "has_image":           r.image is not None,
                "image_shape":         str(r.image.shape)
                                       if r.image is not None else "",
                "n_scenes_composited": r.n_scenes_composited,
                "error_msg":           r.error_msg,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # !! SET YOUR GEE PROJECT ID HERE
    GEE_PROJECT = "ee-yourproject"   # e.g. "ee-justinguthrie"

    INPUT_CSV = "data/maine_deduped_assets.csv"

    if not os.path.exists(INPUT_CSV):
        print(f"No CSV at {INPUT_CSV} — run sources.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} assets")

    # Small sample to verify setup
    df_sample = (
        df.groupby("asset_type", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), 2), random_state=42))
          .reset_index(drop=True)
    )
    print(f"Testing with {len(df_sample)} sampled assets...")

    fetcher = GEEImageryFetcher(
        project=GEE_PROJECT,
        buffer_m=150,
        composite="median",        # recommended for training data
        checkpoint_path="data/checkpoints/gee_test.pkl",
    )

    results = fetcher.fetch_all(df_sample)

    print("\nSummary:")
    summary = fetcher.summarize(results)
    print(summary[["asset_type", "source", "status",
                   "n_scenes_composited", "image_shape"]].to_string(index=False))