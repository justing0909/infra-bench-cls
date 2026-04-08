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

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import os

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


def _fetch_sentinel2(catalog, bbox: tuple, cfg: dict) -> tuple:
    """
    Fetches the least-cloudy recent Sentinel-2 tile for a bbox.
    Returns (image_array, date_str) or raises on failure.
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

    # Pick item with lowest cloud cover among most recent
    items_sorted = sorted(
        items,
        key=lambda i: (
            -i.datetime.timestamp(),                    # most recent first
            i.properties.get("eo:cloud_cover", 100),   # then least cloudy
        )
    )
    item = items_sorted[0]

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
    ):
        self.buffer_m = buffer_m
        self.sources  = sources or ["naip", "sentinel2"]
        self._catalog = None

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
                    image, date = _fetch_sentinel2(catalog, bbox, cfg)

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
    ) -> List[TileResult]:
        """
        Fetches tiles for all assets in a sources.py DataFrame.
        Returns a flat list of TileResult (multiple per asset if multi-source).

        Parameters
        ----------
        df         : DataFrame from GeoFabrikSource.extract_all()
        max_assets : cap for testing — set None for full run
        """
        if max_assets is not None:
            df = df.head(max_assets)

        all_results = []
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            tile_results = self.fetch_tile(row)
            all_results.extend(tile_results)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                ok    = sum(1 for r in all_results if r.status == "ok")
                fails = sum(1 for r in all_results if r.status != "ok")
                print(f"  [{i+1}/{total} assets] ok={ok}  failed={fails}")

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