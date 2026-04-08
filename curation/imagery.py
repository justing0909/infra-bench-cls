"""
imagery.py
----------
Retrieves NAIP imagery tiles for infrastructure assets using Microsoft
Planetary Computer. Takes asset records from sources.py and returns
cropped numpy arrays centered on each asset centroid.

Each tile is a fixed-size geographic crop (buffer_m meters around the
centroid) pulled from the most recent available NAIP scene.

Dependencies:
    pystac_client, planetary_computer, rasterio, numpy, pandas

Usage:
    from imagery import ImageryFetcher
    fetcher = ImageryFetcher(buffer_m=150)
    tile = fetcher.fetch_tile(asset_row)   # single asset (pd.Series)
    results = fetcher.fetch_all(df)        # whole DataFrame from sources.py
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from pystac_client import Client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NAIP_COLLECTION        = "naip"
NAIP_DATE_RANGE        = "2021-01-01/2024-12-31"
NAIP_SEARCH_LIMIT      = 12   # max scenes to consider per query

# Default buffer around asset centroid in meters.
# 150m gives ~300m x 300m crop — enough context for most infra assets.
DEFAULT_BUFFER_M = 150


# ---------------------------------------------------------------------------
# Data class for a tile result
# ---------------------------------------------------------------------------

@dataclass
class TileResult:
    """
    Holds the output of a single tile fetch attempt.

    Attributes
    ----------
    asset_id    : str    — from sources.py
    asset_type  : str    — sector label
    lat / lon   : float  — centroid
    bbox        : tuple  — (min_lon, min_lat, max_lon, max_lat) used for crop
    image       : np.ndarray or None — (bands, rows, cols), RGB uint8
    image_date  : str or None — acquisition date of the NAIP scene
    status      : str    — "ok", "no_scene", "empty_crop", "error"
    error_msg   : str    — populated if status != "ok"
    """
    asset_id   : str
    asset_type : str
    lat        : float
    lon        : float
    bbox       : tuple
    image      : Optional[np.ndarray] = None
    image_date : Optional[str]        = None
    status     : str                  = "ok"
    error_msg  : str                  = ""


# ---------------------------------------------------------------------------
# Helper: meters to degrees (approximate, good enough for small buffers)
# ---------------------------------------------------------------------------

def _meters_to_degrees(meters: float, lat: float) -> tuple:
    """
    Converts a meter buffer to approximate lat/lon degree offsets.
    Uses WGS84 approximation — sufficient for tile generation.

    Returns (delta_lat, delta_lon).
    """
    delta_lat = meters / 111_320
    delta_lon = meters / (111_320 * np.cos(np.radians(lat)))
    return delta_lat, delta_lon


# ---------------------------------------------------------------------------
# Helper: bbox overlap (from Ed's notebook)
# ---------------------------------------------------------------------------

def _bbox_overlap_area(a: tuple, b: tuple) -> float:
    """Calculate overlap area of two (min_lon, min_lat, max_lon, max_lat) boxes."""
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])
    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    if maxx <= minx or maxy <= miny:
        return 0.0
    return (maxx - minx) * (maxy - miny)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ImageryFetcher:
    """
    Fetches NAIP imagery tiles centered on infrastructure asset centroids.

    Parameters
    ----------
    buffer_m : float
        Buffer in meters around the asset centroid. Controls tile size.
        Default 150m gives a ~300m x 300m crop.
    date_range : str
        NAIP date range to search. Format: "YYYY-MM-DD/YYYY-MM-DD".
    """

    def __init__(self, buffer_m: float = DEFAULT_BUFFER_M,
                 date_range: str = NAIP_DATE_RANGE):
        self.buffer_m   = buffer_m
        self.date_range = date_range
        self._catalog   = None   # lazy-loaded on first use

    def _get_catalog(self):
        """Opens Planetary Computer STAC catalog (cached after first call)."""
        if self._catalog is None:
            self._catalog = Client.open(
                PLANETARY_COMPUTER_URL,
                modifier=pc.sign_inplace,
            )
        return self._catalog

    def _centroid_to_bbox(self, lat: float, lon: float) -> tuple:
        """
        Converts a centroid + buffer to a (min_lon, min_lat, max_lon, max_lat)
        bounding box.
        """
        delta_lat, delta_lon = _meters_to_degrees(self.buffer_m, lat)
        return (
            lon - delta_lon,  # min_lon
            lat - delta_lat,  # min_lat
            lon + delta_lon,  # max_lon
            lat + delta_lat,  # max_lat
        )

    def _get_best_naip_item(self, bbox: tuple):
        """
        Finds the most recent NAIP scene with the best overlap for a bbox.
        Mirrors Ed's get_latest_naip_item() logic.

        Returns the STAC item, or None if no scene found.
        """
        catalog = self._get_catalog()
        try:
            items = list(
                catalog.search(
                    collections=[NAIP_COLLECTION],
                    bbox=bbox,
                    datetime=self.date_range,
                    limit=NAIP_SEARCH_LIMIT,
                    method="POST",
                ).items()
            )
        except Exception:
            return None

        if not items:
            return None

        # Pick most recent date, then best overlap within that date
        latest_date = max(
            item.datetime for item in items if item.datetime
        ).date()
        same_day = [
            item for item in items
            if item.datetime and item.datetime.date() == latest_date
        ]
        return max(same_day, key=lambda item: _bbox_overlap_area(bbox, item.bbox))

    def _read_clip(self, item, bbox: tuple) -> np.ndarray:
        """
        Reads RGB bands from a NAIP STAC item clipped to bbox.
        Mirrors Ed's read_naip_clip() logic.

        Returns (3, rows, cols) uint8 array.
        """
        href = item.assets["image"].href
        with rasterio.open(href) as src:
            minx, miny, maxx, maxy = transform_bounds(
                "EPSG:4326", src.crs, *bbox
            )
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            rgb = src.read([1, 2, 3], window=window)

        if rgb.shape[1] == 0 or rgb.shape[2] == 0:
            raise ValueError("Empty crop — bbox may be outside scene coverage.")

        return rgb

    def fetch_tile(self, asset_row: pd.Series) -> TileResult:
        """
        Fetches a single imagery tile for one asset.

        Parameters
        ----------
        asset_row : pd.Series
            One row from the sources.py DataFrame. Must have 'lat', 'lon',
            'asset_id', 'asset_type'.

        Returns
        -------
        TileResult
        """
        lat        = float(asset_row["lat"])
        lon        = float(asset_row["lon"])
        asset_id   = str(asset_row["asset_id"])
        asset_type = str(asset_row["asset_type"])
        bbox       = self._centroid_to_bbox(lat, lon)

        result = TileResult(
            asset_id=asset_id,
            asset_type=asset_type,
            lat=lat,
            lon=lon,
            bbox=bbox,
        )

        # Step 1: find NAIP scene
        item = self._get_best_naip_item(bbox)
        if item is None:
            result.status    = "no_scene"
            result.error_msg = "No NAIP scene found for this location/date range."
            return result

        result.image_date = str(item.datetime.date())

        # Step 2: clip the tile
        try:
            rgb = self._read_clip(item, bbox)
            result.image  = rgb
            result.status = "ok"
        except ValueError as e:
            result.status    = "empty_crop"
            result.error_msg = str(e)
        except Exception as e:
            result.status    = "error"
            result.error_msg = str(e)

        return result

    def fetch_all(self, df: pd.DataFrame,
                  max_assets: Optional[int] = None) -> list:
        """
        Fetches tiles for all assets in a sources.py DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Output from InfrastructureSource.query_all().
        max_assets : int or None
            Cap the number of assets processed. Useful for small test runs.

        Returns
        -------
        list of TileResult
            One per asset. Check result.status for "ok" / "no_scene" / "error".
        """
        if max_assets is not None:
            df = df.head(max_assets)

        results = []
        total   = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            result = self.fetch_tile(row)
            results.append(result)

            # Progress logging every 10 assets
            if (i + 1) % 10 == 0 or (i + 1) == total:
                ok    = sum(1 for r in results if r.status == "ok")
                fails = sum(1 for r in results if r.status != "ok")
                print(f"  [{i+1}/{total}] ok={ok}  failed={fails}")

        return results

    def summarize(self, results: list) -> pd.DataFrame:
        """
        Converts a list of TileResults to a summary DataFrame.
        Useful for inspecting what fetched successfully before moving to QC.
        """
        rows = []
        for r in results:
            rows.append({
                "asset_id":   r.asset_id,
                "asset_type": r.asset_type,
                "lat":        r.lat,
                "lon":        r.lon,
                "image_date": r.image_date,
                "status":     r.status,
                "has_image":  r.image is not None,
                "image_shape": str(r.image.shape) if r.image is not None else "",
                "error_msg":  r.error_msg,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def stretch_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Converts a (3, rows, cols) float or uint8 array to a display-ready
    (rows, cols, 3) uint8 array using percentile stretching.
    Mirrors Ed's stretch_rgb() from the detection notebook.
    """
    # bands-first → bands-last
    rgb_hwc = np.moveaxis(rgb, 0, -1).astype(np.float32)

    stretched = np.zeros_like(rgb_hwc)
    for i in range(3):
        band  = rgb_hwc[:, :, i]
        lo    = np.percentile(band, 2)
        hi    = np.percentile(band, 98)
        if hi > lo:
            stretched[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)

    return (stretched * 255).astype(np.uint8)


def show_tiles(results: list, max_show: int = 9) -> None:
    """
    Displays a grid of successfully fetched tiles using matplotlib.

    Parameters
    ----------
    results  : list of TileResult
    max_show : int — cap the number of tiles shown (default 9)
    """
    import matplotlib.pyplot as plt

    ok_results = [r for r in results if r.status == "ok" and r.image is not None]
    if not ok_results:
        print("No successful tiles to display.")
        return

    ok_results = ok_results[:max_show]
    n     = len(ok_results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

    # normalise axes to always be a flat list
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, result in zip(axes, ok_results):
        display_img = stretch_rgb(result.image)
        ax.imshow(display_img)
        ax.set_title(
            f"{result.asset_type}\n"
            f"{result.lat:.4f}, {result.lon:.4f}\n"
            f"NAIP: {result.image_date}",
            fontsize=8
        )
        ax.axis("off")

    # hide any unused subplots
    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Imagery tiles — imagery.py demo", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig("tile_preview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Tile grid saved to tile_preview.png")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sources import InfrastructureSource

    # Use a small bbox (Portland, ME area) so this runs quickly as a demo
    portland_bbox = (-70.4, 43.5, -70.0, 43.9)

    print("Step 1: querying OSM sources...")
    src = InfrastructureSource(bbox=portland_bbox)
    df  = src.query_all()

    if df.empty:
        print("No assets found — try a different bbox.")
    else:
        print(f"\nStep 2: fetching imagery tiles (first 9 assets as demo)...")
        fetcher = ImageryFetcher(buffer_m=150)
        results = fetcher.fetch_all(df, max_assets=9)

        print("\nTile fetch summary:")
        summary = fetcher.summarize(results)
        print(summary[["asset_type", "image_date", "status",
                        "has_image", "image_shape", "error_msg"]].to_string(index=False))

        print("\nStep 3: displaying tiles...")
        show_tiles(results)