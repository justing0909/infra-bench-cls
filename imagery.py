"""
imagery.py
----------
Fetches imagery tiles for infrastructure assets from multiple sources:
  - NAIP       (60cm, USA only)   via Microsoft Planetary Computer
  - Sentinel-2 (10m,  global)     via Microsoft Planetary Computer

For each asset, all available sources are fetched, producing up to
one tile per source. This multi-source approach supports the goal of
three images per asset (Sentinel-2, NAIP, Maxar) for pretraining.

Dependencies:
    pystac_client, planetary_computer, rasterio, numpy, pandas

Usage:
    from imagery import ImageryFetcher
    fetcher = ImageryFetcher(buffer_m=150)
    results = fetcher.fetch_all(df)   # df from sources.py
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List

from pystac_client import Client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Source configs — name, collection, date range, RGB band names, search limit
SOURCE_CONFIGS = {
    "naip": {
        "collection":   "naip",
        "date_range":   "2021-01-01/2024-12-31",
        "rgb_bands":    ["image"],        # NAIP stores all bands in one asset
        "naip_mode":    True,             # special handling flag
        "search_limit": 12,
        "cloud_filter": None,             # NAIP has no cloud metadata
        "description":  "NAIP 60cm (USA only)",
    },
    "sentinel2": {
        "collection":   "sentinel-2-l2a",
        "date_range":   "2021-01-01/2024-12-31",
        "rgb_bands":    ["B04", "B03", "B02"],   # red, green, blue
        "naip_mode":    False,
        "search_limit": 20,
        "cloud_filter": 20,               # max cloud cover % to accept
        "description":  "Sentinel-2 L2A 10m (global)",
    },
}

DEFAULT_BUFFER_M = 150


# ---------------------------------------------------------------------------
# Data class for a single-source tile result
# ---------------------------------------------------------------------------

@dataclass
class TileResult:
    """
    Holds the result of fetching one imagery tile from one source.

    Attributes
    ----------
    asset_id    : str
    asset_type  : str
    lat / lon   : float  — centroid
    bbox        : tuple  — (min_lon, min_lat, max_lon, max_lat)
    source      : str    — "naip" or "sentinel2"
    image       : np.ndarray or None — (3, rows, cols) uint8 RGB
    image_date  : str or None
    status      : str    — "ok", "no_scene", "empty_crop", "error"
    error_msg   : str
    """
    asset_id   : str
    asset_type : str
    lat        : float
    lon        : float
    bbox       : tuple
    source     : str
    image      : Optional[np.ndarray] = None
    image_date : Optional[str]        = None
    status     : str                  = "ok"
    error_msg  : str                  = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meters_to_degrees(meters: float, lat: float):
    delta_lat = meters / 111_320
    delta_lon = meters / (111_320 * np.cos(np.radians(lat)))
    return delta_lat, delta_lon


def _centroid_to_bbox(lat: float, lon: float, buffer_m: float) -> tuple:
    delta_lat, delta_lon = _meters_to_degrees(buffer_m, lat)
    return (lon - delta_lon, lat - delta_lat,
            lon + delta_lon, lat + delta_lat)


def _bbox_overlap(a: tuple, b: tuple) -> float:
    minx = max(a[0], b[0]); miny = max(a[1], b[1])
    maxx = min(a[2], b[2]); maxy = min(a[3], b[3])
    if maxx <= minx or maxy <= miny:
        return 0.0
    return (maxx - minx) * (maxy - miny)


# ---------------------------------------------------------------------------
# Agentic scene selector (optional — improves scene quality at scale)
# ---------------------------------------------------------------------------

class AgentSceneSelector:
    """
    Uses an LLM to select the best Sentinel-2 scene from candidates
    rather than blindly taking the most recent one.

    The agent considers cloud cover, season, data completeness, and
    asset type context to make a smarter pick. At scale this reduces
    wasted fetches on poor-quality scenes.

    Same backend pattern as AgentTriager — works with Ollama locally
    or Anthropic/OpenAI in production.

    Parameters
    ----------
    backend  : str  — "ollama", "anthropic", or "openai"
    api_key  : str  — not needed for ollama
    model    : str  — defaults per backend
    """

    BACKEND_DEFAULTS = {
        "ollama":    {"base_url": "http://localhost:11434/v1",
                      "api_key": "ollama", "model": "llama3.1"},
        "anthropic": {"base_url": "https://api.anthropic.com/v1",
                      "api_key": None, "model": "claude-sonnet-4-20250514"},
        "openai":    {"base_url": "https://api.openai.com/v1",
                      "api_key": None, "model": "gpt-4o-mini"},
    }

    def __init__(self, backend: str = "ollama",
                 api_key: Optional[str] = None,
                 model: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("AgentSceneSelector requires: pip install openai")

        defaults       = self.BACKEND_DEFAULTS[backend]
        self.model     = model   or defaults["model"]
        self._client   = OpenAI(
            base_url=defaults["base_url"],
            api_key=api_key or defaults["api_key"],
        )

    def select(self, items: list, asset_type: str, bbox: tuple) -> object:
        """
        Selects the best scene from a list of STAC items for a given asset.

        Falls back to the heuristic (most recent, least cloudy) if the
        agent fails or returns an invalid index.

        Parameters
        ----------
        items      : list of pystac Items
        asset_type : str — e.g. "energy.transmission.substation"
        bbox       : tuple — (min_lon, min_lat, max_lon, max_lat)

        Returns
        -------
        The selected pystac Item
        """
        import json as _json

        if len(items) == 1:
            return items[0]

        # Build candidate summary for the agent
        candidates = []
        for i, item in enumerate(items):
            candidates.append({
                "index":       i,
                "date":        str(item.datetime.date()),
                "cloud_cover": round(item.properties.get("eo:cloud_cover", 99), 1),
                "coverage":    round(_bbox_overlap(bbox, item.bbox) /
                               max(((bbox[2]-bbox[0])*(bbox[3]-bbox[1])), 1e-9), 3),
            })

        prompt = (
            f"You are selecting the best satellite imagery scene for a "
            f"'{asset_type}' asset.\n\n"
            f"Candidate scenes:\n{_json.dumps(candidates, indent=2)}\n\n"
            f"Choose the scene index that best balances: low cloud cover, "
            f"good spatial coverage of the bounding box, and recent date. "
            f"Reply with only the integer index of the best scene."
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            idx = int(response.choices[0].message.content.strip())
            if 0 <= idx < len(items):
                return items[idx]
        except Exception:
            pass  # fall back to heuristic

        # Heuristic fallback
        return sorted(items, key=lambda i: (
            -i.datetime.timestamp(),
            i.properties.get("eo:cloud_cover", 100),
        ))[0]


# ---------------------------------------------------------------------------
# Per-source fetch logic
# ---------------------------------------------------------------------------

def _fetch_naip(catalog, bbox: tuple, cfg: dict) -> tuple:
    """
    Fetches the best NAIP tile for a bbox.
    Returns (image_array, date_str) or raises on failure.
    """
    items = list(catalog.search(
        collections=[cfg["collection"]],
        bbox=bbox,
        datetime=cfg["date_range"],
        limit=cfg["search_limit"],
        method="POST",
    ).items())

    if not items:
        raise LookupError("no_scene")

    latest = max(i.datetime for i in items if i.datetime).date()
    same_day = [i for i in items if i.datetime and i.datetime.date() == latest]
    item = max(same_day, key=lambda i: _bbox_overlap(bbox, i.bbox))

    href = item.assets["image"].href
    with rasterio.open(href) as src:
        minx, miny, maxx, maxy = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        rgb = src.read([1, 2, 3], window=window)

    if rgb.shape[1] == 0 or rgb.shape[2] == 0:
        raise ValueError("empty_crop")

    return rgb, str(item.datetime.date())


def _fetch_sentinel2(catalog, bbox: tuple, cfg: dict,
                     scene_selector=None) -> tuple:
    """
    Fetches the best Sentinel-2 tile for a bbox.
    If scene_selector is provided, uses it to pick the best scene.
    Otherwise falls back to heuristic (most recent, least cloudy).
    """
    search_params = dict(
        collections=[cfg["collection"]],
        bbox=bbox,
        datetime=cfg["date_range"],
        limit=cfg["search_limit"],
    )
    if cfg["cloud_filter"] is not None:
        search_params["query"] = {
            "eo:cloud_cover": {"lt": cfg["cloud_filter"]}
        }

    items = list(catalog.search(**search_params).items())

    if not items:
        # Relax cloud filter and try again
        items = list(catalog.search(
            collections=[cfg["collection"]],
            bbox=bbox,
            datetime=cfg["date_range"],
            limit=cfg["search_limit"],
        ).items())

    if not items:
        raise LookupError("no_scene")

    # Pick best scene — agent if available, otherwise heuristic
    if scene_selector is not None:
        item = scene_selector.select(items, cfg.get("asset_type", ""), bbox)
    else:
        item = sorted(items, key=lambda i: (
            -i.datetime.timestamp(),
            i.properties.get("eo:cloud_cover", 100),
        ))[0]

    # Read each RGB band separately and stack
    bands = []
    for band_name in cfg["rgb_bands"]:
        asset = item.assets.get(band_name)
        if asset is None:
            raise KeyError(f"Band {band_name} not found in item assets")
        with rasterio.open(asset.href) as src:
            minx, miny, maxx, maxy = transform_bounds(
                "EPSG:4326", src.crs, *bbox
            )
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            band = src.read(1, window=window)
            bands.append(band)

    rgb = np.stack(bands, axis=0)  # (3, rows, cols)

    if rgb.shape[1] == 0 or rgb.shape[2] == 0:
        raise ValueError("empty_crop")

    # Sentinel-2 values are uint16 (0-10000 reflectance scaled)
    # Normalize to uint8 for consistency with NAIP
    rgb = np.clip(rgb / 10000.0 * 255, 0, 255).astype(np.uint8)

    return rgb, str(item.datetime.date())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImageryFetcher:
    """
    Fetches imagery tiles for infrastructure assets from multiple sources.

    Parameters
    ----------
    buffer_m : float
        Buffer in meters around each asset centroid.
        150m gives ~300m x 300m crop at NAIP resolution.
        Consider larger values (300-500m) for Sentinel-2 at 10m resolution.
    sources : list of str
        Which imagery sources to fetch. Default: ["naip", "sentinel2"].
        Available: "naip", "sentinel2"
    """

    def __init__(
        self,
        buffer_m: float = DEFAULT_BUFFER_M,
        sources: Optional[List[str]] = None,
        scene_selector=None,
    ):
        self.buffer_m       = buffer_m
        self.sources        = sources or ["naip", "sentinel2"]
        self.scene_selector = scene_selector   # optional AgentSceneSelector
        self._catalog       = None

        # Validate source names
        for s in self.sources:
            if s not in SOURCE_CONFIGS:
                raise ValueError(
                    f"Unknown source '{s}'. "
                    f"Available: {list(SOURCE_CONFIGS.keys())}"
                )

    def _get_catalog(self):
        if self._catalog is None:
            self._catalog = Client.open(
                PLANETARY_COMPUTER_URL,
                modifier=pc.sign_inplace,
            )
        return self._catalog

    def fetch_tile(self, asset_row: pd.Series) -> List[TileResult]:
        """
        Fetches tiles for one asset from all configured sources.

        Returns a list of TileResult — one per source.
        """
        lat        = float(asset_row["lat"])
        lon        = float(asset_row["lon"])
        asset_id   = str(asset_row["asset_id"])
        asset_type = str(asset_row["asset_type"])
        bbox       = _centroid_to_bbox(lat, lon, self.buffer_m)
        catalog    = self._get_catalog()

        results = []
        for source_name in self.sources:
            cfg = SOURCE_CONFIGS[source_name]
            result = TileResult(
                asset_id=asset_id, asset_type=asset_type,
                lat=lat, lon=lon, bbox=bbox, source=source_name,
            )
            try:
                if cfg["naip_mode"]:
                    image, date = _fetch_naip(catalog, bbox, cfg)
                else:
                    image, date = _fetch_sentinel2(
                        catalog, bbox, cfg,
                        scene_selector=self.scene_selector
                    )

                result.image      = image
                result.image_date = date
                result.status     = "ok"

            except LookupError:
                result.status    = "no_scene"
                result.error_msg = f"No {source_name} scene for this location"
            except ValueError as e:
                result.status    = "empty_crop"
                result.error_msg = str(e)
            except Exception as e:
                result.status    = "error"
                result.error_msg = str(e)

            results.append(result)

        return results

    def fetch_all(
        self,
        df: pd.DataFrame,
        max_assets: Optional[int] = None,
        max_workers: int = 8,
    ) -> List[TileResult]:
        """
        Fetches tiles for all assets using parallel threads.

        Parameters
        ----------
        df          : DataFrame from GeoFabrikSource.extract_all()
        max_assets  : cap for testing — set None for full run
        max_workers : number of parallel threads (default 8)
                      increase for faster fetching, decrease if hitting
                      API rate limits. 8 is a safe starting point for
                      Planetary Computer.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        if max_assets is not None:
            df = df.head(max_assets)

        rows  = [row for _, row in df.iterrows()]
        total = len(rows)

        all_results = []
        completed   = 0
        lock        = threading.Lock()

        def fetch_one(row):
            return self.fetch_tile(row)

        print(f"  Fetching {total} assets with {max_workers} parallel workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, row): i
                       for i, row in enumerate(rows)}

            for future in as_completed(futures):
                tile_results = future.result()
                with lock:
                    all_results.extend(tile_results)
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        ok    = sum(1 for r in all_results if r.status == "ok")
                        fails = sum(1 for r in all_results if r.status != "ok")
                        pct   = completed / total * 100
                        print(f"  [{completed}/{total} ({pct:.0f}%)] "
                              f"ok={ok}  failed={fails}")

        return all_results

    def summarize(self, results: List[TileResult]) -> pd.DataFrame:
        """Converts results to a summary DataFrame for inspection."""
        rows = []
        for r in results:
            rows.append({
                "asset_id":    r.asset_id,
                "asset_type":  r.asset_type,
                "source":      r.source,
                "lat":         r.lat,
                "lon":         r.lon,
                "image_date":  r.image_date,
                "status":      r.status,
                "has_image":   r.image is not None,
                "image_shape": str(r.image.shape) if r.image is not None else "",
                "error_msg":   r.error_msg,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batched imagery fetcher — groups assets by Sentinel-2 MGRS tile
# ---------------------------------------------------------------------------

class BatchedImageryFetcher:
    """
    Fetches Sentinel-2 tiles by grouping assets into MGRS scene batches.

    Instead of one API call per asset, this fetcher:
      1. Maps each asset lat/lon to its Sentinel-2 MGRS tile ID
      2. Groups assets by MGRS tile
      3. Downloads the Sentinel-2 scene ONCE per MGRS tile
      4. Clips all assets within that scene from the cached raster

    This dramatically reduces API calls and download volume when assets
    cluster in the same geographic area — common for infrastructure which
    tends to be regionally distributed.

    NAIP tiles are still fetched individually (NAIP uses a different
    tiling scheme) via a fallback to ImageryFetcher.

    Checkpointing: saves progress to disk every `checkpoint_every` assets
    so a crashed run can be resumed without starting over.

    Parameters
    ----------
    buffer_m          : float — meters around each asset centroid
    checkpoint_path   : str   — path to checkpoint file (.pkl)
                                set to None to disable checkpointing
    checkpoint_every  : int   — save checkpoint every N assets
    naip_fallback     : bool  — also fetch NAIP via ImageryFetcher
    """

    def __init__(
        self,
        buffer_m: float = DEFAULT_BUFFER_M,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
        naip_fallback: bool = True,
    ):
        self.buffer_m         = buffer_m
        self.checkpoint_path  = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.naip_fallback    = naip_fallback
        self._catalog         = None
        self._scene_cache: Dict[str, object] = {}  # mgrs_id → rasterio dataset

    def _get_catalog(self):
        if self._catalog is None:
            self._catalog = Client.open(
                PLANETARY_COMPUTER_URL,
                modifier=pc.sign_inplace,
            )
        return self._catalog

    def _lat_lon_to_mgrs_id(self, lat: float, lon: float,
                             precision: int = 1) -> Optional[str]:
        """
        Converts lat/lon to a Sentinel-2 MGRS tile ID at the given precision.
        precision=1 gives 100km tile IDs like '19T' — the right granularity
        for grouping assets that share a Sentinel-2 scene.

        Returns None if the mgrs library is not installed.
        """
        try:
            import mgrs as mgrs_lib
            m   = mgrs_lib.MGRS()
            raw = m.toMGRS(lat, lon, MGRSPrecision=precision)
            # Return just the zone+band+square portion (first 5 chars at precision=1)
            return raw[:5]
        except ImportError:
            return None
        except Exception:
            return None

    def _get_best_sentinel2_scene(self, bbox: tuple):
        """
        Finds and returns the best Sentinel-2 STAC item for a bbox.
        Returns None if no suitable scene found.
        """
        cfg = SOURCE_CONFIGS["sentinel2"]
        try:
            items = list(self._get_catalog().search(
                collections=[cfg["collection"]],
                bbox=bbox,
                datetime=cfg["date_range"],
                limit=cfg["search_limit"],
                query={"eo:cloud_cover": {"lt": cfg["cloud_filter"]}}
                if cfg["cloud_filter"] else {},
            ).items())

            if not items:
                items = list(self._get_catalog().search(
                    collections=[cfg["collection"]],
                    bbox=bbox,
                    datetime=cfg["date_range"],
                    limit=cfg["search_limit"],
                ).items())

            if not items:
                return None

            return sorted(items, key=lambda i: (
                -i.datetime.timestamp(),
                i.properties.get("eo:cloud_cover", 100),
            ))[0]
        except Exception:
            return None

    def _clip_from_item(self, item, bbox: tuple) -> Optional[np.ndarray]:
        """Clips RGB bands from a STAC item to a bbox. Returns uint8 array."""
        cfg   = SOURCE_CONFIGS["sentinel2"]
        bands = []
        try:
            for band_name in cfg["rgb_bands"]:
                asset = item.assets.get(band_name)
                if asset is None:
                    return None
                with rasterio.open(asset.href) as src:
                    minx, miny, maxx, maxy = transform_bounds(
                        "EPSG:4326", src.crs, *bbox
                    )
                    window = from_bounds(minx, miny, maxx, maxy, src.transform)
                    band   = src.read(1, window=window)
                    bands.append(band)

            rgb = np.stack(bands, axis=0)
            if rgb.shape[1] == 0 or rgb.shape[2] == 0:
                return None
            return np.clip(rgb / 10000.0 * 255, 0, 255).astype(np.uint8)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> dict:
        """Loads checkpoint from disk. Returns empty dict if none exists."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            import pickle
            with open(self.checkpoint_path, "rb") as f:
                data = pickle.load(f)
            print(f"  Resuming from checkpoint: "
                  f"{len(data['results'])} tiles already fetched")
            return data
        return {"results": [], "completed_ids": set()}

    def _save_checkpoint(self, results: list, completed_ids: set) -> None:
        """Saves progress to checkpoint file."""
        if not self.checkpoint_path:
            return
        import pickle
        os.makedirs(os.path.dirname(self.checkpoint_path)
                    if os.path.dirname(self.checkpoint_path) else ".",
                    exist_ok=True)
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump({"results": results, "completed_ids": completed_ids}, f)

    def _clear_checkpoint(self) -> None:
        """Removes checkpoint file after successful completion."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)
            print(f"  Checkpoint cleared: {self.checkpoint_path}")

    # ------------------------------------------------------------------
    # Main fetch logic
    # ------------------------------------------------------------------

    def fetch_all(self, df: pd.DataFrame,
                  max_assets: Optional[int] = None) -> List[TileResult]:
        """
        Fetches Sentinel-2 tiles for all assets using MGRS scene batching.

        Parameters
        ----------
        df          : DataFrame from GeoFabrikSource.extract_all()
        max_assets  : cap for testing
        """
        if max_assets is not None:
            df = df.head(max_assets)

        # Load checkpoint — skip already-completed assets
        checkpoint     = self._load_checkpoint()
        all_results    = checkpoint["results"]
        completed_ids  = checkpoint["completed_ids"]

        # Filter to pending assets
        pending = df[~df["asset_id"].isin(completed_ids)].copy()
        total   = len(df)
        print(f"  {len(pending)} assets pending "
              f"({len(completed_ids)} already done from checkpoint)")

        # Map each asset to its MGRS tile ID
        mgrs_available = True
        try:
            import mgrs as _mgrs_test
        except ImportError:
            mgrs_available = False
            print("  Warning: mgrs library not installed — "
                  "falling back to per-asset fetching. "
                  "Install with: pip install mgrs")

        if not mgrs_available:
            # Fall back to sequential fetch
            fetcher = ImageryFetcher(buffer_m=self.buffer_m,
                                     sources=["sentinel2"])
            for _, row in pending.iterrows():
                results = fetcher.fetch_tile(row)
                all_results.extend(results)
                completed_ids.add(row["asset_id"])
                if len(completed_ids) % self.checkpoint_every == 0:
                    self._save_checkpoint(all_results, completed_ids)
                    pct = len(completed_ids) / total * 100
                    print(f"  [{len(completed_ids)}/{total} ({pct:.0f}%)] "
                          f"checkpoint saved")
            self._clear_checkpoint()
            return all_results

        # Group assets by MGRS tile
        pending["_mgrs_id"] = pending.apply(
            lambda r: self._lat_lon_to_mgrs_id(r["lat"], r["lon"]) or "unknown",
            axis=1
        )

        groups        = pending.groupby("_mgrs_id")
        n_groups      = groups.ngroups
        groups_done   = 0

        print(f"  Grouped {len(pending)} assets into "
              f"{n_groups} MGRS scene batches")

        for mgrs_id, group in groups:
            groups_done += 1

            # Find the best Sentinel-2 scene for this MGRS group
            # Use the group's bounding box to search
            group_bbox = (
                group["lon"].min() - 0.01,
                group["lat"].min() - 0.01,
                group["lon"].max() + 0.01,
                group["lat"].max() + 0.01,
            )
            item = self._get_best_sentinel2_scene(group_bbox)

            if item is None:
                # No scene found — mark all assets in group as no_scene
                for _, row in group.iterrows():
                    all_results.append(TileResult(
                        asset_id=row["asset_id"],
                        asset_type=row["asset_type"],
                        lat=row["lat"], lon=row["lon"],
                        bbox=_centroid_to_bbox(row["lat"], row["lon"],
                                               self.buffer_m),
                        source="sentinel2",
                        status="no_scene",
                        error_msg=f"No Sentinel-2 scene for MGRS {mgrs_id}",
                    ))
                    completed_ids.add(row["asset_id"])
                continue

            scene_date = str(item.datetime.date())

            # Clip each asset in the group from this scene
            for _, row in group.iterrows():
                bbox = _centroid_to_bbox(
                    float(row["lat"]), float(row["lon"]), self.buffer_m
                )
                image = self._clip_from_item(item, bbox)

                if image is not None:
                    result = TileResult(
                        asset_id=row["asset_id"],
                        asset_type=row["asset_type"],
                        lat=row["lat"], lon=row["lon"],
                        bbox=bbox, source="sentinel2",
                        image=image, image_date=scene_date,
                        status="ok",
                    )
                else:
                    result = TileResult(
                        asset_id=row["asset_id"],
                        asset_type=row["asset_type"],
                        lat=row["lat"], lon=row["lon"],
                        bbox=bbox, source="sentinel2",
                        status="empty_crop",
                        error_msg="Clip returned empty array",
                    )

                all_results.append(result)
                completed_ids.add(row["asset_id"])

            # Progress + checkpoint
            ok  = sum(1 for r in all_results if r.status == "ok")
            pct = len(completed_ids) / total * 100
            print(f"  [{groups_done}/{n_groups} groups | "
                  f"{len(completed_ids)}/{total} assets ({pct:.0f}%)] "
                  f"ok={ok} | scene={mgrs_id} ({len(group)} assets)")

            if len(completed_ids) % self.checkpoint_every == 0:
                self._save_checkpoint(all_results, completed_ids)

        # Also fetch NAIP if requested
        if self.naip_fallback:
            print("\n  Fetching NAIP tiles (individual, US only)...")
            naip_fetcher = ImageryFetcher(
                buffer_m=self.buffer_m, sources=["naip"]
            )
            naip_results = naip_fetcher.fetch_all(df)
            all_results.extend(naip_results)

        self._clear_checkpoint()
        ok_total = sum(1 for r in all_results if r.status == "ok")
        print(f"\n  Batched fetch complete: {ok_total} ok / "
              f"{len(all_results) - ok_total} failed")
        return all_results


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    (3, rows, cols) -> (rows, cols, 3) uint8 with percentile stretch.
    Works for both NAIP and Sentinel-2 output (already uint8 after fetch).
    """
    rgb_hwc = np.moveaxis(rgb, 0, -1).astype(np.float32)
    stretched = np.zeros_like(rgb_hwc)
    for i in range(3):
        band = rgb_hwc[:, :, i]
        lo, hi = np.percentile(band, 2), np.percentile(band, 98)
        if hi > lo:
            stretched[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)
    return (stretched * 255).astype(np.uint8)


def show_tiles(results: List[TileResult], max_show: int = 9) -> None:
    """
    Displays a grid of successfully fetched tiles.
    Subtitle includes source name so you can compare NAIP vs Sentinel-2.
    """
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.status == "ok" and r.image is not None]
    if not ok:
        print("No successful tiles to display.")
        return

    ok = ok[:max_show]
    ncols = min(3, len(ok))
    nrows = (len(ok) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if len(ok) > 1 else [axes]

    for ax, r in zip(axes, ok):
        ax.imshow(stretch_rgb(r.image))
        ax.set_title(
            f"{r.asset_type}\n"
            f"{r.source} | {r.image_date}\n"
            f"{r.lat:.4f}, {r.lon:.4f}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[len(ok):]:
        ax.axis("off")

    plt.suptitle("Imagery tiles", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("tile_preview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved tile_preview.png")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sources import GeoFabrikSource

    PBF_PATH  = "data/pbf/maine-latest.osm.pbf"
    INPUT_CSV = "data/maine_all_assets.csv"

    # Load from CSV if it exists, otherwise re-extract
    if os.path.exists(INPUT_CSV):
        import pandas as pd
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} assets from {INPUT_CSV}")
    else:
        src = GeoFabrikSource(PBF_PATH)
        df  = src.extract_all()

    print("\nFetching tiles for first 3 assets from all sources...")
    fetcher = ImageryFetcher(buffer_m=150, sources=["naip", "sentinel2"])
    results = fetcher.fetch_all(df, max_assets=3)

    print("\nSummary:")
    summary = fetcher.summarize(results)
    print(summary[["asset_type", "source", "image_date",
                   "status", "image_shape", "error_msg"]].to_string(index=False))

    print("\nDisplaying tiles...")
    show_tiles(results, max_show=6)