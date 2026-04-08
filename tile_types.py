"""
tile_types.py
-------------
Shared data types and helper functions used by both imagery.py and
gee_imagery.py. Extracted here to avoid circular imports.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


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
    source      : str    — "naip", "sentinel2", "sentinel2_gee", etc.
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