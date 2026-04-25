"""
tile_types.py
-------------
Shared data types and helper functions used by imagery fetchers,
qc.py, triage.py, and dataset.py.

Extended to support:
  - multimodal imagery (sentinel2_ms, sentinel1, landsat_thermal, naip)
  - temporal image stacks (T, C, H, W)
  - per-modality band metadata
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


# ---------------------------------------------------------------------------
# Modality definitions
# ---------------------------------------------------------------------------
# Central registry of all supported imagery modalities.
# Used by stac_imagery.py, qc.py, and dataset.py.
#
# value_range: expected (min, max) after normalization
#   - uint8 optical:  (0, 255)
#   - float32 SAR:    typically (-30, 10) dB after log scaling
#   - float32 thermal: surface temp in Kelvin or Celsius depending on scaling

MODALITY_REGISTRY = {
    "sentinel2_rgb": {
        "bands": ["B04", "B03", "B02"],
        "n_bands": 3,
        "resolution_m": 10,
        "dtype": "uint8",
        "value_range": (0, 255),
        "description": "Sentinel-2 RGB (10m)",
    },
    "sentinel2_ms": {
        "bands": ["B04", "B03", "B02", "B08", "B8A", "B11", "B12"],
        "n_bands": 7,
        "resolution_m": 10,
        "dtype": "uint8",
        "value_range": (0, 255),
        "description": "Sentinel-2 multispectral RGB+NIR+RedEdge+SWIR (10/20m)",
    },
    "sentinel1": {
        "bands": ["VV", "VH"],
        "n_bands": 2,
        "resolution_m": 10,
        "dtype": "float32",
        "value_range": (-30.0, 10.0),   # dB scale after log transform
        "description": "Sentinel-1 SAR GRD (10m, cloud-independent)",
    },
    "landsat_thermal": {
        "bands": ["ST_B10"],
        "n_bands": 1,
        "resolution_m": 30,
        "dtype": "float32",
        "value_range": (200.0, 350.0),  # Kelvin, surface temp
        "description": "Landsat Collection 2 thermal infrared Band 10 (30m)",
    },
    "naip": {
        "bands": ["R", "G", "B", "NIR"],
        "n_bands": 4,
        "resolution_m": 1,
        "dtype": "uint8",
        "value_range": (0, 255),
        "description": "NAIP aerial imagery RGBN (1m, CONUS only)",
    },
}

# Ordered list for stacking when multiple modalities are fetched.
# Determines band ordering in (C, H, W) stacked arrays.
MODALITY_STACK_ORDER = [
    "sentinel2_ms",
    "sentinel2_rgb",
    "sentinel1",
    "landsat_thermal",
    "naip",
]

# Total bands if all modalities are stacked
def total_bands(modalities: List[str]) -> int:
    return sum(MODALITY_REGISTRY[m]["n_bands"] for m in modalities
               if m in MODALITY_REGISTRY)


# ---------------------------------------------------------------------------
# Core tile dataclass
# ---------------------------------------------------------------------------

@dataclass
class TileResult:
    """
    Holds the result of fetching one imagery tile for one asset.

    image       : np.ndarray or None
                  Single-date:  (C, H, W) where C = total bands across modalities
                  Use image_stack for temporal tiles (see below)
    image_stack : np.ndarray or None
                  Temporal stack: (T, C, H, W) — T seasonal composites
                  Only populated when temporal_stack=True in the fetcher.
                  image holds the single best composite in that case too
                  for backward compatibility with qc.py / triage.py.

    modalities  : list of modality keys from MODALITY_REGISTRY
    n_bands     : total band count across all active modalities
    n_timesteps : number of temporal steps if image_stack is populated (else 1)

    source      : fetcher identifier, e.g. "stac_s2ms", "stac_s1", "gee_s2"
    status      : "ok", "no_scene", "empty_crop", "partial", "error"
    """
    asset_id    : str
    asset_type  : str
    lat         : float
    lon         : float
    bbox        : tuple
    source      : str

    # Primary image — single date or best composite
    image       : Optional[np.ndarray]  = None   # (C, H, W)

    # Temporal stack — populated when temporal_stack=True
    image_stack : Optional[np.ndarray]  = None   # (T, C, H, W)

    # Modality metadata
    modalities  : List[str]             = field(default_factory=lambda: ["sentinel2_rgb"])
    n_bands     : int                   = 3
    n_timesteps : int                   = 1

    # Acquisition metadata
    image_date  : Optional[str]         = None
    image_dates : List[str]             = field(default_factory=list)  # per timestep

    # Status
    status      : str                   = "ok"
    error_msg   : str                   = ""

    @property
    def has_temporal(self) -> bool:
        return self.image_stack is not None and self.n_timesteps > 1

    @property
    def image_shape(self) -> Optional[tuple]:
        if self.image is not None:
            return tuple(self.image.shape)
        return None

    @property
    def stack_shape(self) -> Optional[tuple]:
        if self.image_stack is not None:
            return tuple(self.image_stack.shape)
        return None


# ---------------------------------------------------------------------------
# Geometry helpers
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