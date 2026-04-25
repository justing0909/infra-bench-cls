"""
gee_imagery.py
--------------
Google Earth Engine imagery fetcher for infrastructure asset tiles.

Replaces the Planetary Computer / rasterio approach in imagery.py with
server-side GEE processing. All clipping and compositing happens on
Google's infrastructure — only the final small tile arrays are transferred
to your machine.

This version is hardened for large runs:
  - conservative concurrency
  - exponential backoff + jitter
  - high-volume EE endpoint
  - atomic checkpoint writes
  - corrupted checkpoint fallback
  - only successful assets counted as completed
"""

import os
import io
import time
import random
import pickle
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from helpers.tile_types import TileResult, _centroid_to_bbox


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Harmonized Sentinel-2 L2A — atmospherically corrected, DN-shift handled
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

# RGB bands at 10m native resolution
S2_RGB_BANDS = ["B4", "B3", "B2"]

# Date range for scene search
S2_DATE_START = "2021-01-01"
S2_DATE_END = "2024-12-31"

# Cloud cover filter — scenes above this % are excluded from compositing
S2_MAX_CLOUD = 20

# Output resolution in meters
S2_SCALE = 10

# Output CRS
S2_CRS = "EPSG:4326"

# Conservative settings for large global runs
MAX_CONCURRENT = 6

# Retry/backoff settings for transient EE / HTTP throttling
MAX_RETRIES = 10
BASE_BACKOFF_S = 3.0
MAX_BACKOFF_S = 60.0

# Small random stagger before each worker starts to reduce burstiness
REQUEST_STAGGER_S = 1.5

# Whether to query EE for collection.size() on every asset.
# This is useful for diagnostics, but it adds an extra EE RPC per asset
# and slows large runs substantially.
TRACK_SCENE_COUNT = False

# ---------------------------------------------------------------------------
# GEE tile result — extends TileResult with GEE-specific metadata
# ---------------------------------------------------------------------------

@dataclass
class GEETileResult(TileResult):
    """
    TileResult subclass with GEE-specific fields.
    Fully compatible with qc.py, triage.py, dataset.py.
    """
    n_scenes_composited: int = 0
    cloud_masked: bool = True


# ---------------------------------------------------------------------------
# Cloud masking function
# ---------------------------------------------------------------------------

def _get_masked_collection(ee, lat: float, lon: float, buffer_m: float):
    """
    Returns a cloud-masked, normalized Sentinel-2 ImageCollection
    filtered to the asset's location and date range.

    Applies two layers of cloud filtering:
      1. CLOUDY_PIXEL_PERCENTAGE metadata filter (scene-level)
      2. QA60 bitmask (pixel-level cloud + cirrus mask)
    """
    def mask_s2_clouds(image):
        qa = image.select("QA60")
        cloud_mask = 1 << 10
        cirrus_mask = 1 << 11
        mask = (
            qa.bitwiseAnd(cloud_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_mask).eq(0))
        )
        return image.updateMask(mask).divide(10000)

    bbox = ee.Geometry.BBox(
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
    buffer_m   : float — meters around each asset centroid
    composite  : str   — "median", "mosaic", or "best"
    scale      : int   — output resolution in meters
    checkpoint_path : str — path to checkpoint file for resumable runs
    checkpoint_every: int — save checkpoint every N successful assets
    """

    def __init__(
        self,
        project: str,
        buffer_m: float = 150,
        composite: str = "median",
        scale: int = S2_SCALE,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
    ):
        self.project = project
        self.buffer_m = buffer_m
        self.composite = composite
        self.scale = scale
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self._ee = None
        self._session = requests.Session()

    def _init_ee(self):
        """Initializes Earth Engine (lazy — only on first use)."""
        if self._ee is not None:
            return self._ee

        try:
            import ee
            ee.Initialize(
                project=self.project,
                opt_url="https://earthengine-highvolume.googleapis.com",
            )
            self._ee = ee
            print(f"  GEE initialized (project={self.project}, endpoint=highvolume)")
            return ee
        except Exception as e:
            raise RuntimeError(
                f"Could not initialize Google Earth Engine: {e}\n"
                f"Make sure you have run: python -c \"import ee; ee.Authenticate()\"\n"
                f"And that your project ID '{self.project}' is correct."
            )

    def _sleep_with_jitter(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        sleep_s = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** attempt))
        sleep_s = sleep_s * (0.7 + 0.6 * random.random())
        time.sleep(sleep_s)
        return sleep_s

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Returns True for transient throttling/server errors."""
        msg = str(exc).lower()

        if isinstance(exc, requests.HTTPError):
            status = getattr(exc.response, "status_code", None)
            if status in (429, 500, 502, 503, 504):
                return True

        retry_strings = [
            "too many requests",
            "429",
            "rate limit",
            "quota exceeded",
            "internal error",
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
            "service unavailable",
            "deadline exceeded",
            "resource exhausted",
        ]
        return any(s in msg for s in retry_strings)

    def _download_image_with_retry(self, image, ee_bbox, asset_id: str):
        """Builds the download URL and fetches the GeoTIFF with retry/backoff."""
        import rasterio as rio

        # Step 1: get URL with retry
        last_exc = None
        url = None
        for attempt in range(MAX_RETRIES):
            try:
                url = image.getDownloadURL({
                    "bands": S2_RGB_BANDS,
                    "region": ee_bbox,
                    "scale": self.scale,
                    "crs": S2_CRS,
                    "format": "GEO_TIFF",
                })
                break
            except Exception as e:
                last_exc = e
                if not self._is_retryable_error(e) or attempt == MAX_RETRIES - 1:
                    raise
                sleep_s = self._sleep_with_jitter(attempt)
                # print(
                    # f"  RETRY [{asset_id}] getDownloadURL error: {e} "
                    # f"(attempt {attempt + 1}/{MAX_RETRIES}, sleep={sleep_s:.1f}s)"
                # )

        if url is None:
            raise last_exc

        # Step 2: download content with retry
        for attempt in range(MAX_RETRIES):
            try:
                response = self._session.get(url, timeout=120)
                response.raise_for_status()

                with rio.open(io.BytesIO(response.content)) as src:
                    arr = src.read()

                return arr

            except Exception as e:
                last_exc = e
                if not self._is_retryable_error(e) or attempt == MAX_RETRIES - 1:
                    raise
                sleep_s = self._sleep_with_jitter(attempt)
                # print(
                #     f"  RETRY [{asset_id}] download error: {e} "
                #     f"(attempt {attempt + 1}/{MAX_RETRIES}, sleep={sleep_s:.1f}s)"
                # )

        raise last_exc

    def _get_info_with_retry(self, ee_obj, asset_id: str):
        """Retries ee_object.getInfo() for transient EE errors."""
        last_exc = None

        for attempt in range(MAX_RETRIES):
            try:
                return ee_obj.getInfo()
            except Exception as e:
                last_exc = e
                if not self._is_retryable_error(e) or attempt == MAX_RETRIES - 1:
                    raise

                sleep_s = self._sleep_with_jitter(attempt)
                # print(
                #     f"  RETRY [{asset_id}] getInfo error: {e} "
                #     f"(attempt {attempt + 1}/{MAX_RETRIES}, sleep={sleep_s:.1f}s)"
                # )

        raise last_exc

    def fetch_tile(self, asset_row: pd.Series) -> GEETileResult:
        """
        Fetches a Sentinel-2 tile for a single asset using GEE.
        """
        ee = self._init_ee()

        lat = float(asset_row["lat"])
        lon = float(asset_row["lon"])
        asset_id = str(asset_row["asset_id"])
        asset_type = str(asset_row["asset_type"])
        bbox_tuple = _centroid_to_bbox(lat, lon, self.buffer_m)

        result = GEETileResult(
            asset_id=asset_id,
            asset_type=asset_type,
            lat=lat,
            lon=lon,
            bbox=bbox_tuple,
            source="sentinel2_gee",
        )

        try:
            collection, ee_bbox = _get_masked_collection(
                ee, lat, lon, self.buffer_m
            )

            # if TRACK_SCENE_COUNT:
                # n_scenes = self._get_info_with_retry(collection.size(), asset_id)
                # if n_scenes == 0:
                #     result.status = "no_scene"
                #     result.error_msg = "No cloud-free Sentinel-2 scenes found"
                #     return result
                
                # result.n_scenes_composited = n_scenes

            if self.composite == "median":
                image = collection.select(S2_RGB_BANDS).median()
            elif self.composite == "mosaic":
                image = (
                    collection
                    .sort("CLOUDY_PIXEL_PERCENTAGE")
                    .select(S2_RGB_BANDS)
                    .mosaic()
                )
            else:  # "best"
                image = (
                    collection
                    .sort("CLOUDY_PIXEL_PERCENTAGE")
                    .first()
                    .select(S2_RGB_BANDS)
                )

            image = image.clip(ee_bbox)

            arr = self._download_image_with_retry(image, ee_bbox, asset_id)

            if arr is None or arr.size == 0 or arr.shape[1] == 0 or arr.shape[2] == 0:
                result.status = "empty_crop"
                result.error_msg = "GEE returned empty array"
                return result

            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

            result.image = arr
            result.image_date = S2_DATE_END
            result.status = "ok"

        except Exception as e:
            msg = str(e).lower()
            if any(s in msg for s in ["no images", "imagecollection", "empty"]):
                result.status = "no_scene"
                result.error_msg = str(e)
            else:
                result.status = "error"
                result.error_msg = str(e)
                print(f"  ERROR [{asset_id}]: {e}")

        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "rb") as f:
                    data = pickle.load(f)

                if not isinstance(data, dict):
                    raise ValueError("Checkpoint is not a dict")

                data.setdefault("results", [])
                data.setdefault("completed_ids", set())

                if not isinstance(data["completed_ids"], set):
                    data["completed_ids"] = set(data["completed_ids"])

                print(
                    f"  Resuming from checkpoint: "
                    f"{len(data['results'])} tiles already fetched"
                )
                return data

            except (EOFError, pickle.UnpicklingError, ValueError, KeyError, TypeError) as e:
                broken_path = f"{self.checkpoint_path}.corrupt"
                try:
                    os.replace(self.checkpoint_path, broken_path)
                    print(
                        f"  Warning: checkpoint unreadable ({e}). "
                        f"Moved it to: {broken_path}"
                    )
                except OSError:
                    print(
                        f"  Warning: checkpoint unreadable ({e}). "
                        f"Ignoring and starting fresh."
                    )

        return {"results": [], "completed_ids": set()}

    def _save_checkpoint(self, results: list, completed_ids: set) -> None:
        if not self.checkpoint_path:
            return

        os.makedirs(
            os.path.dirname(self.checkpoint_path)
            if os.path.dirname(self.checkpoint_path) else ".",
            exist_ok=True,
        )

        tmp_path = f"{self.checkpoint_path}.tmp"
        payload = {
            "results": results,
            "completed_ids": list(completed_ids),
        }

        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, self.checkpoint_path)

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
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        if max_assets is not None:
            df = df.head(max_assets)

        checkpoint = self._load_checkpoint()
        all_results = checkpoint["results"]
        completed_ids = checkpoint["completed_ids"]

        pending = df[~df["asset_id"].isin(completed_ids)].copy()
        total = len(df)

        print(
            f"  GEE fetch: {len(pending)} assets pending "
            f"({len(completed_ids)} successful assets already checkpointed)"
        )

        self._init_ee()
        print("  EE ready; starting worker pool...")

        rows = [row for _, row in pending.iterrows()]
        lock = threading.Lock()

        def fetch_one(row):
            time.sleep(random.random() * REQUEST_STAGGER_S)
            result = self.fetch_tile(row)
            time.sleep(0.3)
            return result

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(fetch_one, row): row["asset_id"]
                for row in rows
            }

            for future in as_completed(futures):
                result = future.result()

                with lock:
                    all_results.append(result)

                    if result.status == "ok":
                        completed_ids.add(result.asset_id)

                    n_done = len(completed_ids)
                    n_fail = sum(1 for r in all_results if r.status != "ok")

                    n_seen = len(all_results)
                    pct = n_seen / total * 100 if total else 0.0

                    # if n_seen <= 10 or n_seen % 25 == 0:
                        # print(
                        #     f"  progress: seen={n_seen}/{total} ({pct:.0f}%) "
                        #     f"ok={n_done} fail={n_fail} last_status={result.status}"
                        # )

                    if n_done > 0 and n_done % self.checkpoint_every == 0:
                        self._save_checkpoint(all_results, completed_ids)
                        print(
                            f"  [{n_done}/{total}] ok={n_done} fail={n_fail} | checkpoint saved"
                        )

        ok_total = sum(1 for r in all_results if r.status == "ok")
        fail_total = len(all_results) - ok_total

        if fail_total == 0:
            self._clear_checkpoint()
        else:
            self._save_checkpoint(all_results, completed_ids)
            print("  Checkpoint preserved because some assets failed.")

        print(f"\n  GEE fetch complete: {ok_total} ok / {fail_total} failed")
        return all_results

    def summarize(self, results: List[GEETileResult]) -> pd.DataFrame:
        """Same interface as ImageryFetcher.summarize()."""
        rows = []
        for r in results:
            rows.append({
                "asset_id": r.asset_id,
                "asset_type": r.asset_type,
                "source": r.source,
                "lat": r.lat,
                "lon": r.lon,
                "image_date": r.image_date,
                "status": r.status,
                "has_image": r.image is not None,
                "image_shape": str(r.image.shape) if r.image is not None else "",
                "n_scenes_composited": r.n_scenes_composited,
                "error_msg": r.error_msg,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    GEE_PROJECT = "towards-an-infra-fm"
    INPUT_CSV = "data/maine_deduped_assets.csv"

    if not os.path.exists(INPUT_CSV):
        print(f"No CSV at {INPUT_CSV} — run sources.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} assets")

    df_sample = (
        df.groupby("asset_type", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), 2), random_state=42))
        .reset_index(drop=True)
    )
    print(f"Testing with {len(df_sample)} sampled assets...")

    fetcher = GEEImageryFetcher(
        project=GEE_PROJECT,
        buffer_m=150,
        composite="median",
        checkpoint_path="data/checkpoints/gee_test.pkl",
    )

    results = fetcher.fetch_all(df_sample)

    print("\nSummary:")
    summary = fetcher.summarize(results)
    print(
        summary[
            ["asset_type", "source", "status", "n_scenes_composited", "image_shape", "error_msg"]
        ].to_string(index=False)
    )