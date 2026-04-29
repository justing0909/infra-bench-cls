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
  sentinel1         Sentinel-1 RTC SAR (VV + VH polarizations)
                    10m resolution, cloud-independent, global
                    Uses sentinel-1-rtc collection (terrain corrected)
  landsat_thermal   Landsat Collection 2 L2 thermal infrared (Band 10)
                    30m resolution, global, back to 1982
  naip              NAIP aerial imagery (R + G + B + NIR)
                    1m resolution, CONUS only

Non-fatal modalities
--------------------
sentinel1, landsat_thermal, and naip are non-fatal — if unavailable for
a tile, zeros are substituted rather than failing the whole tile.
This ensures sentinel2_ms always drives tile success/failure.

Adaptive concurrency
--------------------
Set adaptive_concurrency=True in STACImageryFetcher to automatically
tune MAX_CONCURRENT during a run based on observed throughput and
fail rate. Starts at start_workers and hill-climbs toward the optimum.

Usage:
    from stac_imagery import STACImageryFetcher
    fetcher = STACImageryFetcher(
        buffer_m=300,
        modalities=["sentinel2_ms", "sentinel1", "landsat_thermal"],
        temporal_stack=False,
        adaptive_concurrency=True,
        checkpoint_path="data/checkpoints/stac_central-america.pkl",
    )
    results = fetcher.fetch_all(df)
"""

import os
import time
import random
import pickle
import threading
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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

DATE_START = "2021-01-01"
DATE_END   = "2024-12-31"

MAX_CLOUD_PCT       = 20
TARGET_RESOLUTION_M = 10

# Default concurrency — adaptive tuning starts here and adjusts up/down
MAX_CONCURRENT    = 16

MAX_RETRIES       = 4
BASE_BACKOFF_S    = 2.0
MAX_BACKOFF_S     = 60.0
REQUEST_STAGGER_S = 0.05

SEASON_WINDOWS = [
    ("01-01", "03-31"),
    ("04-01", "06-30"),
    ("07-01", "09-30"),
    ("10-01", "12-31"),
]

COLLECTION_MAP = {
    "sentinel2_ms":    "sentinel-2-l2a",
    "sentinel2_rgb":   "sentinel-2-l2a",
    "sentinel1":       "sentinel-1-rtc",   # RTC — terrain corrected, no auth required
    "landsat_thermal": "landsat-c2-l2",
    "naip":            "naip",
}

BAND_ASSET_KEYS = {
    "sentinel2_ms":    ["B04", "B03", "B02", "B08", "B8A", "B11", "B12"],
    "sentinel2_rgb":   ["B04", "B03", "B02"],
    "sentinel1":       ["vv", "vh"],
    "landsat_thermal": ["ST_B10", "lwir", "lwir11"],
    "naip":            ["image"],
}

# Non-fatal modalities — missing fills with zeros instead of failing tile
NON_FATAL_MODALITIES = {"naip", "sentinel1", "landsat_thermal"}

# ---------------------------------------------------------------------------
# Adaptive concurrency controller
# ---------------------------------------------------------------------------

class AdaptiveConcurrencyController:
    """
    Hill-climbing adaptive concurrency tuner.

    Evaluates throughput and fail rate every `window` completions and
    adjusts worker count up or down accordingly. Tracks the best observed
    (concurrency, throughput) pair and converges toward it.

    Parameters
    ----------
    min_workers   : int   — floor on worker count
    max_workers   : int   — ceiling on worker count
    start_workers : int   — initial worker count
    window        : int   — evaluate every N tile completions
    step          : int   — workers to add or remove per adjustment
    max_fail_rate : float — fail rate above which we back off
    """

    def __init__(self,
                 min_workers   : int   = 8,
                 max_workers   : int   = 128,
                 start_workers : int   = 16,
                 window        : int   = 20,
                 step          : int   = 8,
                 max_fail_rate : float = 0.10):
        self.min_workers   = min_workers
        self.max_workers   = max_workers
        self.current       = start_workers
        self.window        = window
        self.step          = step
        self.max_fail_rate = max_fail_rate

        self._window_start  = time.time()
        self._window_ok     = 0
        self._window_fail   = 0
        self._best_workers  = start_workers
        self._best_tp       = 0.0
        self._last_tp       = 0.0
        self._history       = []

    def record(self, success: bool) -> bool:
        """Record one tile completion. Returns True when window is full."""
        if success:
            self._window_ok += 1
        else:
            self._window_fail += 1
        return (self._window_ok + self._window_fail) >= self.window

    def evaluate_and_adjust(self) -> int:
        """Evaluate window and return new worker count."""
        total     = self._window_ok + self._window_fail
        elapsed   = time.time() - self._window_start
        fail_rate = self._window_fail / max(total, 1)
        tp        = self._window_ok / max(elapsed, 0.01)

        self._history.append((self.current, round(tp, 2), round(fail_rate, 3)))

        if fail_rate > self.max_fail_rate:
            self.current = max(self.min_workers, self.current - self.step)
            action = "↓ back off (high fail rate)"
        elif tp > self._best_tp:
            self._best_tp      = tp
            self._best_workers = self.current
            self.current = min(self.max_workers, self.current + self.step)
            action = "↑ increase (throughput improving)"
        elif tp < self._last_tp * 0.9:
            self.current = max(self.min_workers,
                               min(self._best_workers, self.current - self.step))
            action = "↓ decrease (throughput degrading)"
        else:
            self.current = min(self.max_workers, self.current + self.step // 2)
            action = "→ nudge up"

        self._last_tp       = tp
        self._window_start  = time.time()
        self._window_ok     = 0
        self._window_fail   = 0

        print(f"  [concurrency] workers={self.current} | "
              f"throughput={tp:.1f} tiles/s | "
              f"fail_rate={fail_rate:.1%} | {action}",
              flush=True)

        return self.current

    def summary(self) -> None:
        if not self._history:
            return
        print(f"\n  Concurrency tuning summary:")
        print(f"  {'workers':>8} {'tiles/s':>10} {'fail_rate':>10}")
        for w, tp, fr in self._history:
            marker = " ← best" if w == self._best_workers else ""
            print(f"  {w:>8} {tp:>10.2f} {fr:>10.1%}{marker}")
        print(f"  Best: {self._best_workers} workers @ {self._best_tp:.2f} tiles/s")


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
    info = MODALITY_REGISTRY[modality]
    vmin, vmax = info["value_range"]

    if info["dtype"] == "uint8":
        if arr.max() > 1.0:
            arr = arr / 10000.0
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    elif modality == "sentinel1":
        arr = arr.astype(np.float32)
        arr = np.where(arr > 0, 10 * np.log10(arr + 1e-10), vmin)
        arr = np.clip(arr, vmin, vmax).astype(np.float32)
    elif modality == "landsat_thermal":
        arr = arr.astype(np.float32) * 0.00341802 + 149.0
        arr = np.clip(arr, vmin, vmax).astype(np.float32)
    else:
        arr = arr.astype(np.float32)

    return arr


def _resample_to_target(arr: np.ndarray,
                         target_h: int,
                         target_w: int) -> np.ndarray:
    if arr.shape[1] == target_h and arr.shape[2] == target_w:
        return arr
    import cv2
    out = np.zeros((arr.shape[0], target_h, target_w), dtype=arr.dtype)
    for c in range(arr.shape[0]):
        out[c] = cv2.resize(arr[c], (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST)
    return out


def _windowed_read(href: str, bbox: tuple) -> Optional[np.ndarray]:
    """
    Opens a raster href, reprojects bbox to its native CRS, and returns
    a windowed read. Falls back to full-tile read if CRS is missing.
    Returns (H, W) float32 array or None on empty read.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    with rasterio.open(href) as src:
        if src.crs is not None:
            bbox_native = tuple(float(x) for x in transform_bounds(
                CRS.from_epsg(4326), src.crs,
                bbox[0], bbox[1], bbox[2], bbox[3],
            ))
            window = rasterio.windows.from_bounds(
                *bbox_native, transform=src.transform
            )
            data = src.read(1, window=window)
        else:
            data = src.read(1)

    return data if data.size > 0 else None


def _windowed_read_multiband(href: str, bbox: tuple) -> Optional[np.ndarray]:
    """
    Same as _windowed_read but reads all bands. Returns (C, H, W) array.
    Used for NAIP which delivers all bands in a single asset.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    with rasterio.open(href) as src:
        if src.crs is not None:
            bbox_native = tuple(float(x) for x in transform_bounds(
                CRS.from_epsg(4326), src.crs,
                bbox[0], bbox[1], bbox[2], bbox[3],
            ))
            window = rasterio.windows.from_bounds(
                *bbox_native, transform=src.transform
            )
            data = src.read(window=window)
        else:
            data = src.read()

    return data if data.size > 0 else None


# ---------------------------------------------------------------------------
# Per-modality fetch functions
# ---------------------------------------------------------------------------

def _fetch_sentinel2(catalog, bbox: tuple, date_start: str, date_end: str,
                     modality: str = "sentinel2_ms") -> Optional[np.ndarray]:
    """
    Fetches Sentinel-2 L2A composite. Sorts client-side by cloud cover.
    Returns (C, H, W) uint8 array or None.
    """
    import planetary_computer
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
    item  = planetary_computer.sign(clean[0] if clean else items[0])

    for band in band_keys:
        if band not in item.assets:
            return None

    arrays   = []
    target_h = None
    target_w = None

    for band in band_keys:
        data = _windowed_read(item.assets[band].href, bbox)
        if data is None:
            return None

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
    """
    Fetches Sentinel-1 RTC (terrain corrected) VV+VH bands.
    Uses sentinel-1-rtc collection — no subscription key required.
    Returns (2, H, W) float32 dB array or None.
    """
    import planetary_computer
    import cv2

    search = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        limit=10,
    )
    items = list(search.items())
    if not items:
        return None

    item     = planetary_computer.sign(items[0])
    arrays   = []
    target_h = None
    target_w = None

    for band in ["vv", "vh"]:
        if band not in item.assets:
            return None

        data = _windowed_read(item.assets[band].href, bbox)
        if data is None:
            return None

        if target_h is None:
            target_h, target_w = data.shape
        elif data.shape != (target_h, target_w):
            data = cv2.resize(
                data.astype(np.float32),
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR,
            )
        arrays.append(data)

    if len(arrays) < 2:
        return None

    arr = np.stack(arrays, axis=0).astype(np.float32)
    return _normalize_array(arr, "sentinel1")


def _fetch_landsat_thermal(catalog, bbox: tuple, date_start: str,
                            date_end: str) -> Optional[np.ndarray]:
    """
    Fetches Landsat Collection 2 L2 thermal band.
    Filters to Landsat 8/9 only — consistent ST_B10/lwir11/lwir asset key,
    no Landsat 7 scan line corrector issues.
    Returns (1, H, W) float32 Kelvin array or None.
    """
    import planetary_computer

    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"platform": {"in": ["landsat-8", "landsat-9"]}},
        limit=50,
    )
    items = list(search.items())
    if not items:
        return None

    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    item = planetary_computer.sign(items[0])

    thermal_key = next((k for k in ["lwir11", "lwir", "ST_B10"] if k in item.assets), None)
    if thermal_key is None:
        return None

    data = _windowed_read(item.assets[thermal_key].href, bbox)
    if data is None:
        return None

    arr = data[np.newaxis, :, :].astype(np.float32)
    return _normalize_array(arr, "landsat_thermal")


def _fetch_naip(catalog, bbox: tuple, date_start: str,
                date_end: str) -> Optional[np.ndarray]:
    """
    Fetches NAIP 4-band (R, G, B, NIR). CONUS only.
    Returns (4, H, W) uint8 array or None.
    """
    import planetary_computer

    search = catalog.search(
        collections=["naip"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        limit=5,
    )
    items = list(search.items())
    if not items:
        return None

    item = planetary_computer.sign(items[0])

    if "image" not in item.assets:
        return None

    data = _windowed_read_multiband(item.assets["image"].href, bbox)
    if data is None:
        return None

    arr = data.astype(np.float32)
    return _normalize_array(arr, "naip")


# Dispatch table
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
    buffer_m             : float — buffer around each asset centroid in meters
    modalities           : list  — modality keys to fetch
    temporal_stack       : bool  — fetch seasonal stacks (T, C, H, W)
    n_years              : int   — years of seasonal composites
    date_start           : str   — ISO date string for scene search start
    date_end             : str   — ISO date string for scene search end
    checkpoint_path      : str   — path to checkpoint pickle
    checkpoint_every     : int   — save checkpoint every N successful tiles
    adaptive_concurrency : bool  — auto-tune worker count during run
    start_workers        : int   — initial worker count (used when adaptive=True)
    max_workers          : int   — ceiling for adaptive tuning
    """

    def __init__(
        self,
        buffer_m             : float       = 300,
        modalities           : List[str]   = None,
        temporal_stack       : bool        = False,
        n_years              : int         = 2,
        date_start           : str         = DATE_START,
        date_end             : str         = DATE_END,
        checkpoint_path      : Optional[str] = None,
        checkpoint_every     : int         = 200,
        adaptive_concurrency : bool        = False,
        start_workers        : int         = MAX_CONCURRENT,
        max_workers          : int         = 128,
    ):
        self.buffer_m             = buffer_m
        self.modalities           = modalities or ["sentinel2_ms"]
        self.temporal_stack       = temporal_stack
        self.n_years              = n_years
        self.date_start           = date_start
        self.date_end             = date_end
        self.checkpoint_path      = checkpoint_path
        self.checkpoint_every     = checkpoint_every
        self.adaptive_concurrency = adaptive_concurrency
        self.start_workers        = start_workers
        self.max_workers          = max_workers

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
              f"buffer_m={self.buffer_m}, "
              f"workers={self.start_workers}"
              f"{' (adaptive)' if self.adaptive_concurrency else ''}")

    def _get_catalog(self):
        import pystac_client
        import planetary_computer
        return pystac_client.Client.open(
            STAC_ENDPOINT,
            modifier=planetary_computer.sign_inplace,
        )

    def _build_temporal_windows(self) -> List[Tuple[str, str]]:
        end_year = int(self.date_end[:4])
        windows  = []
        for year_offset in range(self.n_years):
            year = end_year - year_offset
            for q_start, q_end in SEASON_WINDOWS:
                windows.append((f"{year}-{q_start}", f"{year}-{q_end}"))
        return list(reversed(windows))

    def _fetch_single_date(self, catalog, lat: float, lon: float,
                           date_start: str, date_end: str,
                           modalities: List[str]) -> Optional[np.ndarray]:
        bbox         = _centroid_to_bbox(lat, lon, self.buffer_m)
        arrays       = []
        ref_h, ref_w = None, None

        for modality in modalities:
            fetch_fn = FETCH_DISPATCH.get(modality)
            if fetch_fn is None:
                continue

            arr = None
            for attempt in range(MAX_RETRIES):
                try:
                    arr = fetch_fn(catalog, bbox, date_start, date_end)
                    break
                except Exception as exc:
                    if _is_retryable(exc) and attempt < MAX_RETRIES - 1:
                        _retry_backoff(attempt)
                    else:
                        arr = None
                        break

            if arr is None:
                if modality in NON_FATAL_MODALITIES:
                    info  = MODALITY_REGISTRY[modality]
                    dummy = np.zeros(
                        (info["n_bands"], ref_h or 32, ref_w or 32),
                        dtype=np.float32,
                    )
                    arrays.append(dummy)
                    continue
                return None

            if ref_h is None:
                ref_h, ref_w = arr.shape[1], arr.shape[2]
            else:
                arr = _resample_to_target(arr, ref_h, ref_w)

            arrays.append(arr)

        if not arrays:
            return None

        return np.concatenate([a.astype(np.float32) for a in arrays], axis=0)

    def fetch_tile(self, row: pd.Series) -> TileResult:
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

                n_windows  = len(windows)
                c, h, w    = stack[0].shape
                full_stack = np.zeros((n_windows, c, h, w), dtype=np.float32)
                full_dates = [""] * n_windows

                j = 0
                for i in range(n_windows):
                    if j < len(stack):
                        full_stack[i] = stack[j]
                        full_dates[i] = dates[j]
                        j += 1

                result.image_stack = full_stack
                result.image       = full_stack[-1]
                result.image_date  = full_dates[-1]
                result.image_dates = full_dates
                result.n_timesteps = n_windows
                result.status      = "ok"

        except Exception as e:
            result.status    = "error"
            result.error_msg = str(e)

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
        df          : pd.DataFrame,
        max_assets  : Optional[int] = None,
    ) -> List[TileResult]:
        """
        Fetches tiles for all assets in df with checkpointing and concurrency.
        If adaptive_concurrency=True, tunes worker count during the run.
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

        # Set up adaptive controller or fixed concurrency
        if self.adaptive_concurrency:
            controller = AdaptiveConcurrencyController(
                start_workers = self.start_workers,
                max_workers   = self.max_workers,
            )
            semaphore = threading.Semaphore(controller.current)
        else:
            controller = None
            semaphore  = threading.Semaphore(self.start_workers)

        def fetch_one(row):
            with semaphore:
                time.sleep(random.random() * REQUEST_STAGGER_S)
                return self.fetch_tile(row)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_one, row): row["asset_id"]
                       for row in rows}

            for future in as_completed(futures):
                result = future.result()

                with lock:
                    all_results.append(result)
                    success = result.status == "ok"

                    if success:
                        completed_ids.add(result.asset_id)

                    n_done = len(completed_ids)
                    n_fail = sum(1 for r in all_results if r.status != "ok")
                    n_seen = len(all_results)
                    pct    = n_seen / total * 100 if total else 0.0

                    if n_seen % 10 == 0 or n_seen == total:
                        current_workers = controller.current if controller else self.start_workers
                        print(f"  [{n_seen}/{total}] ({pct:.0f}%) "
                              f"ok={n_done} fail={n_fail} "
                              f"workers={current_workers}",
                              flush=True)

                    # Adaptive: adjust semaphore when window is full
                    if controller and controller.record(success):
                        new_count = controller.evaluate_and_adjust()
                        diff = new_count - semaphore._value
                        if diff > 0:
                            for _ in range(diff):
                                semaphore.release()
                        # Decreasing is handled naturally as threads finish

                    if n_done > 0 and n_done % self.checkpoint_every == 0:
                        self._save_checkpoint(all_results, completed_ids)
                        print(f"  checkpoint saved at {n_done} ok tiles",
                              flush=True)

        ok_total   = sum(1 for r in all_results if r.status == "ok")
        fail_total = len(all_results) - ok_total

        if controller:
            controller.summary()

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
        buffer_m             = 300,
        modalities           = ["sentinel2_ms", "sentinel1"],
        temporal_stack       = False,
        adaptive_concurrency = True,
        start_workers        = 16,
        max_workers          = 64,
        checkpoint_path      = "data/checkpoints/stac_test.pkl",
    )
    results = fetcher.fetch_all(df_sample)

    print("\nSummary:")
    print(fetcher.summarize(results).to_string(index=False))