"""
stac_imagery.py
---------------
Planetary Computer STAC imagery fetcher for infrastructure asset tiles.

Replaces gee_imagery.py as the primary imagery source.
All imagery is free via Microsoft Planetary Computer's public STAC endpoint.

Supported modalities
--------------------
  sentinel2_ms      Sentinel-2 L2A multispectral (RGB + NIR + RedEdge + SWIR)
                    10/20m resolution, 5-day revisit, global
  sentinel1         Sentinel-1 GRD SAR (VV + VH polarizations)
                    10m resolution, cloud-independent, global
  landsat_thermal   Landsat Collection 2 L2 thermal infrared (Band 10)
                    30m resolution, global, back to 1982
  naip              NAIP aerial imagery (R + G + B + NIR)
                    1m resolution, CONUS only

Temporal stacks
---------------
When temporal_stack=True, the fetcher collects N seasonal composites
(one per quarter by default) and returns a (T, C, H, W) image_stack
in addition to the single best composite in TileResult.image.

Output tile arrays
------------------
  Single-date:  image       shape (C, H, W)
  Temporal:     image_stack shape (T, C, H, W)
                image       shape (C, H, W)  — best composite, for QC

C = total bands across active modalities (see MODALITY_REGISTRY in tile_types.py)

Interface
---------
Identical to GEEImageryFetcher: fetch_all(df) -> List[TileResult]
Drop-in replacement in pipeline.py via USE_STAC=True.

Dataset version suffix
----------------------
Datasets produced by this fetcher are versioned as:
  dataset_<region>_stac_v1/
(vs. dataset_<region>_sentinel_v1/ from the GEE path)

Usage:
    from stac_imagery import STACImageryFetcher
    fetcher = STACImageryFetcher(
        buffer_m=300,
        modalities=["sentinel2_ms", "sentinel1"],
        temporal_stack=True,
        checkpoint_path="data/checkpoints/stac_central-america.pkl",
    )
    results = fetcher.fetch_all(df)
"""

import os
import io
import time
import random
import pickle
import threading
from googlesearch import search
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

from helpers.tile_types import (
    TileResult,
    MODALITY_REGISTRY,
    _centroid_to_bbox,
    total_bands,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Default date range for scene search
DATE_START = "2021-01-01"
DATE_END   = "2024-12-31"

# Cloud cover filter for optical modalities
MAX_CLOUD_PCT = 20

# Output resolution for resampling (meters) — all modalities resampled to this
# when stacking. Set to None to keep each modality at native resolution.
TARGET_RESOLUTION_M = 10

# Conservative concurrency — Planetary Computer is generous but not unlimited
MAX_CONCURRENT = 32

# Retry / backoff
MAX_RETRIES     = 4
BASE_BACKOFF_S  = 2.0
MAX_BACKOFF_S   = 60.0
REQUEST_STAGGER_S = 0.05

# Temporal stack: seasons per year
SEASON_WINDOWS = [
    ("01-01", "03-31"),   # Q1 — winter
    ("04-01", "06-30"),   # Q2 — spring
    ("07-01", "09-30"),   # Q3 — summer
    ("10-01", "12-31"),   # Q4 — fall
]

# Collection IDs on Planetary Computer
COLLECTION_MAP = {
    "sentinel2_ms":    "sentinel-2-l2a",
    "sentinel2_rgb":   "sentinel-2-l2a",
    "sentinel1":       "sentinel-1-grd",
    "landsat_thermal": "landsat-c2-l2",
    "naip":            "naip",
}

# Band names per modality as used in PC STAC assets
BAND_ASSET_KEYS = {
    "sentinel2_ms":    ["B04", "B03", "B02", "B08", "B8A", "B11", "B12"],
    "sentinel2_rgb":   ["B04", "B03", "B02"],
    "sentinel1":       ["VV", "VH"],
    "landsat_thermal": ["ST_B10"],
    "naip":            ["image"],   # NAIP is delivered as a single 4-band asset
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry_backoff(attempt: int) -> float:
    sleep_s = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** attempt))
    sleep_s *= (0.7 + 0.6 * random.random())
    time.sleep(sleep_s)
    return sleep_s


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_strings = [
        "429", "too many requests", "rate limit", "quota",
        "500", "502", "503", "504", "timed out", "timeout",
        "connection reset", "service unavailable", "temporarily unavailable",
    ]
    return any(s in msg for s in retry_strings)


def _normalize_array(arr: np.ndarray, modality: str) -> np.ndarray:
    """
    Normalizes a raw fetched array to the expected dtype/range for its modality.

    Optical (sentinel2_ms, sentinel2_rgb, naip): -> uint8 [0, 255]
    SAR (sentinel1):  -> float32 dB scale, clipped to [-30, 10]
    Thermal (landsat_thermal): -> float32 Kelvin, clipped to [200, 350]
    """
    info = MODALITY_REGISTRY[modality]
    vmin, vmax = info["value_range"]

    if info["dtype"] == "uint8":
        # Assume input is reflectance [0, 1] or DN [0, 10000]
        if arr.max() > 1.0:
            arr = arr / 10000.0
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    elif modality == "sentinel1":
        # Convert linear power to dB
        arr = arr.astype(np.float32)
        arr = np.where(arr > 0, 10 * np.log10(arr + 1e-10), vmin)
        arr = np.clip(arr, vmin, vmax).astype(np.float32)
    elif modality == "landsat_thermal":
        # Landsat ST_B10 scale factor: multiply by 0.00341802, add 149.0 -> Kelvin
        arr = arr.astype(np.float32) * 0.00341802 + 149.0
        arr = np.clip(arr, vmin, vmax).astype(np.float32)
    else:
        arr = arr.astype(np.float32)

    return arr


def _resample_to_target(arr: np.ndarray,
                         target_h: int,
                         target_w: int) -> np.ndarray:
    """
    Resamples a (C, H, W) array to (C, target_h, target_w) using nearest neighbor.
    Used to harmonize multi-resolution modalities into a single spatial grid.
    """
    if arr.shape[1] == target_h and arr.shape[2] == target_w:
        return arr

    import cv2
    out = np.zeros((arr.shape[0], target_h, target_w), dtype=arr.dtype)
    for c in range(arr.shape[0]):
        out[c] = cv2.resize(arr[c], (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST)
    return out


# ---------------------------------------------------------------------------
# Per-modality fetch functions
# ---------------------------------------------------------------------------

def _fetch_sentinel2(catalog, bbox: tuple, date_start: str, date_end: str,
                     modality: str = "sentinel2_ms") -> Optional[np.ndarray]:
    import planetary_computer
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds
    import cv2

    band_keys = BAND_ASSET_KEYS[modality]

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        limit=50,
    )
    items = list(search.items())
    if not items:
        return None

    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    clean = [i for i in items if i.properties.get("eo:cloud_cover", 100) < MAX_CLOUD_PCT]
    item = planetary_computer.sign(clean[0] if clean else items[0])

    arrays = []
    target_h, target_w = None, None

    for band in band_keys:
        if band not in item.assets:
            return None
        href = item.assets[band].href
        with rasterio.open(href) as src:
            bbox_native = transform_bounds(
                CRS.from_epsg(4326), src.crs,
                bbox[0], bbox[1], bbox[2], bbox[3],
            )
            window = rasterio.windows.from_bounds(
                *bbox_native, transform=src.transform
            )
            data = src.read(1, window=window)

        # Set reference shape from first band
        if target_h is None:
            target_h, target_w = data.shape
        elif data.shape != (target_h, target_w):
            data = cv2.resize(
                data.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )

        arrays.append(data)

    if not arrays:
        return None

    arr = np.stack(arrays, axis=0).astype(np.float32)
    return _normalize_array(arr, modality)


def _fetch_sentinel1(catalog, bbox: tuple, date_start: str,
                     date_end: str) -> Optional[np.ndarray]:
    import planetary_computer
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        limit=10,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )
    items = list(search.items())
    if not items:
        return None

    item = planetary_computer.sign(items[0])
    arrays = []
    for band in ["VV", "VH"]:
        if band not in item.assets:
            return None
        href = item.assets[band].href
        with rasterio.open(href) as src:
            bbox_native = transform_bounds(
                CRS.from_epsg(4326), src.crs,
                bbox[0], bbox[1], bbox[2], bbox[3],
            )
            window = rasterio.windows.from_bounds(
                *bbox_native, transform=src.transform
            )
            data = src.read(1, window=window)
            arrays.append(data)

    if len(arrays) < 2:
        return None

    arr = np.stack(arrays, axis=0).astype(np.float32)
    return _normalize_array(arr, "sentinel1")


def _fetch_landsat_thermal(catalog, bbox: tuple, date_start: str,
                            date_end: str) -> Optional[np.ndarray]:
    import planetary_computer
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
        limit=50,
    )
    items = list(search.items())
    if not items:
        return None

    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    item = planetary_computer.sign(items[0])

    if "ST_B10" not in item.assets:
        return None

    href = item.assets["ST_B10"].href
    with rasterio.open(href) as src:
        bbox_native = transform_bounds(
            CRS.from_epsg(4326), src.crs,
            bbox[0], bbox[1], bbox[2], bbox[3],
        )
        window = rasterio.windows.from_bounds(
            *bbox_native, transform=src.transform
        )
        data = src.read(1, window=window)

    arr = data[np.newaxis, :, :].astype(np.float32)
    return _normalize_array(arr, "landsat_thermal")


def _fetch_naip(catalog, bbox: tuple, date_start: str,
                date_end: str) -> Optional[np.ndarray]:
    import planetary_computer
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    search = catalog.search(
        collections=["naip"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        limit=5,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )
    items = list(search.items())
    if not items:
        return None

    item = planetary_computer.sign(items[0])

    if "image" not in item.assets:
        return None

    href = item.assets["image"].href
    with rasterio.open(href) as src:
        bbox_native = transform_bounds(
            CRS.from_epsg(4326), src.crs,
            bbox[0], bbox[1], bbox[2], bbox[3],
        )
        window = rasterio.windows.from_bounds(
            *bbox_native, transform=src.transform
        )
        data = src.read(window=window)

    arr = data.astype(np.float32)
    return _normalize_array(arr, "naip")


# Dispatch table: modality -> fetch function
FETCH_DISPATCH = {
    "sentinel2_ms":    lambda cat, bbox, ds, de: _fetch_sentinel2(cat, bbox, ds, de, "sentinel2_ms"),
    "sentinel2_rgb":   lambda cat, bbox, ds, de: _fetch_sentinel2(cat, bbox, ds, de, "sentinel2_rgb"),
    "sentinel1":       _fetch_sentinel1,
    "landsat_thermal": _fetch_landsat_thermal,
    "naip":            _fetch_naip,
}


# ---------------------------------------------------------------------------
# Main fetcher class
# ---------------------------------------------------------------------------

class STACImageryFetcher:
    """
    Fetches multimodal imagery tiles via Microsoft Planetary Computer STAC.

    Parameters
    ----------
    buffer_m          : float  — buffer around each asset centroid in meters
    modalities        : list   — ordered list of modality keys to fetch
                                 default: ["sentinel2_ms"]
                                 options: sentinel2_ms, sentinel2_rgb,
                                          sentinel1, landsat_thermal, naip
    temporal_stack    : bool   — if True, fetch N seasonal composites and
                                 return a (T, C, H, W) image_stack
    n_years           : int    — number of years to collect seasonal composites
                                 (each year has 4 quarters = T = n_years * 4)
    date_start        : str    — ISO date string for scene search start
    date_end          : str    — ISO date string for scene search end
    checkpoint_path   : str    — path to checkpoint pickle for resumable runs
    checkpoint_every  : int    — save checkpoint every N successful assets

    Notes
    -----
    - All modalities are resampled to TARGET_RESOLUTION_M (10m default)
      for stacking when multiple modalities are requested.
    - SAR and thermal arrays are float32; optical arrays are uint8.
      qc.py handles per-modality thresholds automatically.
    - NAIP coverage is CONUS only; tiles outside CONUS return status="no_scene"
      for the naip modality but the other modalities still succeed.
    """

    def __init__(
        self,
        buffer_m         : float       = 300,
        modalities       : List[str]   = None,
        temporal_stack   : bool        = False,
        n_years          : int         = 2,
        date_start       : str         = DATE_START,
        date_end         : str         = DATE_END,
        checkpoint_path  : Optional[str] = None,
        checkpoint_every : int         = 200,
    ):
        self.buffer_m         = buffer_m
        self.modalities       = modalities or ["sentinel2_ms"]
        self.temporal_stack   = temporal_stack
        self.n_years          = n_years
        self.date_start       = date_start
        self.date_end         = date_end
        self.checkpoint_path  = checkpoint_path
        self.checkpoint_every = checkpoint_every

        # Validate modalities
        for m in self.modalities:
            if m not in MODALITY_REGISTRY:
                raise ValueError(
                    f"Unknown modality '{m}'. "
                    f"Available: {list(MODALITY_REGISTRY.keys())}"
                )

        self._n_bands = total_bands(self.modalities)
        print(f"STACImageryFetcher: modalities={self.modalities}, "
              f"n_bands={self._n_bands}, "
              f"temporal_stack={self.temporal_stack}, "
              f"buffer_m={self.buffer_m}")

    def _get_catalog(self):
        """Returns a signed pystac_client catalog (lazy init per thread)."""
        import pystac_client
        import planetary_computer
        return pystac_client.Client.open(
            STAC_ENDPOINT,
            modifier=planetary_computer.sign_inplace,
        )

    def _build_temporal_windows(self) -> List[Tuple[str, str]]:
        """
        Builds (date_start, date_end) pairs for each seasonal window
        across n_years ending at date_end.
        Returns list of (start, end) string pairs.
        """
        end_year  = int(self.date_end[:4])
        windows   = []
        for year_offset in range(self.n_years):
            year = end_year - year_offset
            for q_start, q_end in SEASON_WINDOWS:
                windows.append((f"{year}-{q_start}", f"{year}-{q_end}"))
        return list(reversed(windows))  # chronological order

    def _fetch_single_date(self, catalog, lat: float, lon: float,
                           date_start: str, date_end: str,
                           modalities: List[str]) -> Optional[np.ndarray]:
        """
        Fetches and stacks all requested modalities for one date window.
        Returns (C, H, W) array with all bands stacked, or None if any
        primary modality fails.
        """
        bbox = _centroid_to_bbox(lat, lon, self.buffer_m)
        arrays   = []
        ref_h, ref_w = None, None

        for modality in modalities:
            fetch_fn = FETCH_DISPATCH.get(modality)
            if fetch_fn is None:
                continue

            for attempt in range(MAX_RETRIES):
                try:
                    arr = fetch_fn(catalog, bbox, date_start, date_end)
                    break
                except Exception as exc:
                    print(f"  DEBUG fetch_fn exception: {type(exc).__name__}: {exc}")
                    import traceback
                    traceback.print_exc()
                    if _is_retryable(exc) and attempt < MAX_RETRIES - 1:
                        _retry_backoff(attempt)
                    else:
                        arr = None
                        break

            if arr is None:
                # NAIP and sentinel1 missing are non-fatal
                if modality in ("naip", "sentinel1"):
                    info = MODALITY_REGISTRY[modality]
                    dummy = np.zeros(
                        (info["n_bands"], ref_h or 32, ref_w or 32),
                        dtype=np.float32,
                    )
                    arrays.append(dummy)
                    continue
                return None

            # Establish reference spatial dimensions from first modality
            if ref_h is None:
                ref_h, ref_w = arr.shape[1], arr.shape[2]
            else:
                arr = _resample_to_target(arr, ref_h, ref_w)

            arrays.append(arr)

        if not arrays:
            return None

        # Stack along channel dimension: (C_total, H, W)
        # Mixed dtypes: cast all to float32 for stacking, keep per-band semantics
        stacked = np.concatenate(
            [a.astype(np.float32) for a in arrays], axis=0
        )
        return stacked

    def fetch_tile(self, row: pd.Series) -> TileResult:
        """
        Fetches all requested modalities for a single asset row.
        Returns a TileResult with image (and image_stack if temporal).
        """
        asset_id   = row["asset_id"]
        asset_type = row["asset_type"]
        lat        = float(row["lat"])
        lon        = float(row["lon"])
        bbox       = _centroid_to_bbox(lat, lon, self.buffer_m)

        source_tag = "stac_" + "+".join(self.modalities)

        result = TileResult(
            asset_id   = asset_id,
            asset_type = asset_type,
            lat        = lat,
            lon        = lon,
            bbox       = bbox,
            source     = source_tag,
            modalities = list(self.modalities),
            n_bands    = self._n_bands,
        )

        try:
            catalog = self._get_catalog()

            if not self.temporal_stack:
                # Single best composite
                arr = self._fetch_single_date(
                    catalog, lat, lon,
                    self.date_start, self.date_end,
                    self.modalities,
                )
                if arr is None or arr.size == 0:
                    result.status    = "no_scene"
                    result.error_msg = "no usable scenes in date range"
                    return result

                result.image      = arr
                result.image_date = self.date_end
                result.status     = "ok"

            else:
                # Temporal stack: one composite per seasonal window
                windows = self._build_temporal_windows()
                stack   = []
                dates   = []

                for (ds, de) in windows:
                    arr = self._fetch_single_date(
                        catalog, lat, lon, ds, de, self.modalities
                    )
                    if arr is not None and arr.size > 0:
                        stack.append(arr)
                        dates.append(de)

                if not stack:
                    result.status    = "no_scene"
                    result.error_msg = "no usable scenes in any temporal window"
                    return result

                # Pad missing windows with zeros to keep uniform (T, C, H, W)
                n_windows = len(windows)
                c, h, w   = stack[0].shape
                full_stack = np.zeros((n_windows, c, h, w), dtype=np.float32)
                full_dates = [""] * n_windows

                j = 0
                for i in range(n_windows):
                    if j < len(stack):
                        full_stack[i] = stack[j]
                        full_dates[i] = dates[j]
                        j += 1

                result.image_stack  = full_stack          # (T, C, H, W)
                result.image        = full_stack[-1]       # best = most recent
                result.image_date   = full_dates[-1]
                result.image_dates  = full_dates
                result.n_timesteps  = n_windows
                result.status       = "ok"

        except Exception as e:
            result.status    = "error"
            result.error_msg = str(e)
            print(f"  DEBUG exception: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

        return result

    # ------------------------------------------------------------------
    # Checkpointing (identical interface to GEEImageryFetcher)
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
                print(f"  Resuming from checkpoint: "
                      f"{len(data['results'])} tiles already fetched")
                return data
            except Exception as e:
                broken = f"{self.checkpoint_path}.corrupt"
                try:
                    os.replace(self.checkpoint_path, broken)
                    print(f"  Warning: checkpoint unreadable ({e}). "
                          f"Moved to {broken}")
                except OSError:
                    print(f"  Warning: checkpoint unreadable ({e}). Starting fresh.")
        return {"results": [], "completed_ids": set()}

    def _save_checkpoint(self, results: list, completed_ids: set) -> None:
        if not self.checkpoint_path:
            return
        os.makedirs(
            os.path.dirname(self.checkpoint_path)
            if os.path.dirname(self.checkpoint_path) else ".",
            exist_ok=True,
        )
        tmp = f"{self.checkpoint_path}.tmp"
        payload = {"results": results, "completed_ids": list(completed_ids)}
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.checkpoint_path)

    def _clear_checkpoint(self) -> None:
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)

    # ------------------------------------------------------------------
    # Batch fetch
    # ------------------------------------------------------------------

    def fetch_all(
        self,
        df: pd.DataFrame,
        max_assets: Optional[int] = None,
    ) -> List[TileResult]:
        """
        Fetches tiles for all assets in df with checkpointing and concurrency.
        Returns list of TileResult in the same order as df.
        """
        if max_assets is not None:
            df = df.head(max_assets)

        checkpoint    = self._load_checkpoint()
        all_results   = checkpoint["results"]
        completed_ids = checkpoint["completed_ids"]

        pending = df[~df["asset_id"].isin(completed_ids)].copy()
        total   = len(df)

        print(f"  STAC fetch: {len(pending)} assets pending "
              f"({len(completed_ids)} already checkpointed)")

        rows = [row for _, row in pending.iterrows()]
        lock = threading.Lock()

        def fetch_one(row):
            time.sleep(random.random() * REQUEST_STAGGER_S)
            return self.fetch_tile(row)

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {executor.submit(fetch_one, row): row["asset_id"]
                       for row in rows}

            for future in as_completed(futures):
                result = future.result()

                with lock:
                    all_results.append(result)

                    if result.status == "ok":
                        completed_ids.add(result.asset_id)

                    n_done = len(completed_ids)
                    n_fail = sum(1 for r in all_results if r.status != "ok")
                    n_seen = len(all_results)
                    pct    = n_seen / total * 100 if total else 0.0

                    if n_seen % 10 == 0 or n_seen == total:
                        print(f"  [{n_seen}/{total}] ({pct:.0f}%) "
                            f"ok={n_done} fail={n_fail}",
                            flush=True)

                    if n_done > 0 and n_done % self.checkpoint_every == 0:
                        self._save_checkpoint(all_results, completed_ids)
                        print(f"  checkpoint saved at {n_done} ok tiles", flush=True)

        ok_total   = sum(1 for r in all_results if r.status == "ok")
        fail_total = len(all_results) - ok_total

        if fail_total == 0:
            self._clear_checkpoint()
        else:
            self._save_checkpoint(all_results, completed_ids)
            print("  Checkpoint preserved (some assets failed).")

        print(f"\n  STAC fetch complete: {ok_total} ok / {fail_total} failed")
        return all_results

    def summarize(self, results: List[TileResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append({
                "asset_id":    r.asset_id,
                "asset_type":  r.asset_type,
                "source":      r.source,
                "lat":         r.lat,
                "lon":         r.lon,
                "modalities":  "+".join(r.modalities),
                "n_bands":     r.n_bands,
                "n_timesteps": r.n_timesteps,
                "image_date":  r.image_date,
                "status":      r.status,
                "has_image":   r.image is not None,
                "has_stack":   r.image_stack is not None,
                "image_shape": str(r.image_shape) if r.image is not None else "",
                "stack_shape": str(r.stack_shape) if r.image_stack is not None else "",
                "error_msg":   r.error_msg,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    INPUT_CSV = "data/maine_deduped_assets.csv"

    if not os.path.exists(INPUT_CSV):
        print(f"No CSV at {INPUT_CSV} — run sources.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    df_sample = (
        df.groupby("asset_type", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), 2), random_state=42))
          .reset_index(drop=True)
    )
    print(f"Testing with {len(df_sample)} sampled assets...")

    fetcher = STACImageryFetcher(
        buffer_m       = 300,
        modalities     = ["sentinel2_ms", "sentinel1"],
        temporal_stack = False,
        checkpoint_path= "data/checkpoints/stac_test.pkl",
    )
    results = fetcher.fetch_all(df_sample)

    print("\nSummary:")
    print(fetcher.summarize(results).to_string(index=False))