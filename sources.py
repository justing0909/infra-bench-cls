"""
sources.py
----------
Extracts infrastructure asset geometries and weak labels from local
GeoFabrik OSM extracts (.osm.pbf files) using pyosmium.

Why GeoFabrik instead of live Overpass API queries?
  - No rate limits or timeouts
  - Works offline once downloaded
  - Scales globally — download a country or continent extract once,
    query it as many times as needed
  - Much faster for large areas

Download GeoFabrik extracts from: https://download.geofabrik.de/
Example: https://download.geofabrik.de/north-america/us/maine-latest.osm.pbf

Each query returns a pandas DataFrame with one row per asset:
  - asset_id      : unique ID (e.g. "osm_node_123456")
  - asset_type    : ontology label (e.g. "energy.transmission.substation")
  - lat / lon     : centroid coordinates
  - name          : OSM name tag if available
  - source        : always "osm_geofabrik"
  - osm_tags      : dict of all OSM tags (for provenance)

Usage:
    from sources import GeoFabrikSource
    src = GeoFabrikSource("data/pbf/maine-latest.osm.pbf")
    df = src.extract_all()
    df = src.extract_sector("energy")   # single sector only
"""

import os
import osmium
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Asset filter definitions
# ---------------------------------------------------------------------------
# Asset filter definitions + visual confidence levels
# ---------------------------------------------------------------------------
# Maps ontology asset_type to:
#   "tags"       : list of (key, value) filter tuples — ALL must match
#   "confidence" : "high", "medium", or "low"
#
# Visual confidence reflects how distinguishable the asset is from overhead
# imagery at Sentinel-2 resolution (10m). At NAIP/Maxar resolution (30-60cm)
# confidence is generally one level higher.
#
# high   — clearly visible, distinctive footprint at 10m
# medium — visible but may be small or ambiguous at 10m
# low    — likely below 10m resolution or visually indistinct
#
# Order matters: more specific filters must come before less specific ones.
# Based on ONTOLOGY.md — update here when the ontology evolves.

ASSET_FILTERS = {
    # --- Energy: Generation ---
    "energy.generation.solar_farm": {
        "tags": [("power", "generator"), ("generator:source", "solar")],
        "confidence": "high",
    },
    "energy.generation.wind_farm": {
        "tags": [("power", "generator"), ("generator:source", "wind")],
        "confidence": "high",
    },
    "energy.generation.power_plant": {
        "tags": [("power", "plant")],
        "confidence": "high",
    },
    "energy.generation.generator": {
        "tags": [("power", "generator")],
        "confidence": "medium",
    },

    # --- Energy: Transmission ---
    "energy.transmission.substation": {
        "tags": [("power", "substation"), ("substation", "transmission")],
        "confidence": "high",
    },
    "energy.transmission.tower": {
        "tags": [("power", "tower")],
        "confidence": "low",   # visible at 30cm, too small at 10m
    },

    # --- Energy: Distribution ---
    "energy.distribution.substation": {
        "tags": [("power", "substation"), ("substation", "distribution")],
        "confidence": "high",
    },
    "energy.distribution.substation_untyped": {
        "tags": [("power", "substation")],
        "confidence": "medium",   # substation but subtype unknown
    },
    "energy.distribution.transformer": {
        "tags": [("power", "transformer")],
        "confidence": "low",
    },
    "energy.distribution.pole": {
        "tags": [("power", "pole")],
        "confidence": "low",
    },

    # --- Stubs: uncomment when sectors are developed ---
    # "transport.airport":  {"tags": [("aeroway", "aerodrome")],        "confidence": "high"},
    # "water.treatment":    {"tags": [("man_made", "wastewater_plant")], "confidence": "high"},
    # "telecom.tower":      {"tags": [("man_made", "tower"), ("tower:type", "communication")], "confidence": "medium"},
}

# Confidence level ordering for threshold filtering
CONFIDENCE_LEVELS = {"high": 3, "medium": 2, "low": 1}

# Group asset_types by top-level sector
SECTORS = {
    "energy": [k for k in ASSET_FILTERS if k.startswith("energy.")],
}


def asset_types_by_confidence(min_confidence: str = "medium") -> set:
    """
    Returns the set of asset_types at or above the given confidence threshold.

    Parameters
    ----------
    min_confidence : str — "high", "medium", or "low"

    Example
    -------
    high_only   = asset_types_by_confidence("high")
    high_medium = asset_types_by_confidence("medium")  # default
    all_assets  = asset_types_by_confidence("low")
    """
    threshold = CONFIDENCE_LEVELS.get(min_confidence, 2)
    return {
        k for k, v in ASSET_FILTERS.items()
        if CONFIDENCE_LEVELS.get(v["confidence"], 0) >= threshold
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches(tags, filter_list: list) -> bool:
    """
    Returns True if all (key, value) pairs in filter_list match the tags.
    value=None matches any value for that key.
    """
    for key, value in filter_list:
        tag_val = tags.get(key)
        if tag_val is None:
            return False
        if value is not None and tag_val != value:
            return False
    return True


def _best_asset_type(tags) -> Optional[str]:
    """
    Returns the most specific matching asset_type for an OSM element's tags,
    or None if no filter matches.
    More specific filters appear first in ASSET_FILTERS so they win.
    """
    for asset_type, defn in ASSET_FILTERS.items():
        if _matches(tags, defn["tags"]):
            return asset_type
    return None


# ---------------------------------------------------------------------------
# pyosmium handler
# ---------------------------------------------------------------------------

class InfraHandler(osmium.SimpleHandler):
    """
    pyosmium handler that scans nodes and ways in a .osm.pbf file and
    collects those matching ASSET_FILTERS.

    Ways use their midpoint node as a centroid proxy.
    Relations are not currently handled.
    """

    def __init__(self, target_types: Optional[set] = None):
        super().__init__()
        self.rows = []
        self._target_types = target_types  # None means accept all

    def _add(self, osm_type: str, osm_id: int,
             lat: float, lon: float, tags) -> None:
        asset_type = _best_asset_type(tags)
        if asset_type is None:
            return
        if self._target_types and asset_type not in self._target_types:
            return

        tag_dict = {k: v for k, v in tags}
        self.rows.append({
            "asset_id":   f"osm_{osm_type}_{osm_id}",
            "asset_type": asset_type,
            "lat":        lat,
            "lon":        lon,
            "name":       tag_dict.get("name", ""),
            "source":     "osm_geofabrik",
            "osm_tags":   tag_dict,
        })

    def node(self, n):
        if not n.location.valid():
            return
        self._add("node", n.id, n.location.lat, n.location.lon, n.tags)

    def way(self, w):
        # Use midpoint node as centroid proxy
        # Requires locations=True in apply_file()
        try:
            mid = w.nodes[len(w.nodes) // 2]
            if mid.location.valid():
                self._add("way", w.id,
                          mid.location.lat, mid.location.lon, w.tags)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GeoFabrikSource:
    """
    Extracts infrastructure assets from a local GeoFabrik .osm.pbf file.

    Parameters
    ----------
    pbf_path : str
        Path to the local .osm.pbf file.
        Download from https://download.geofabrik.de/

    min_confidence : str
        Minimum visual confidence level to include.
        "high"   — only clearly visible assets (substations, solar, wind, plants)
        "medium" — adds untyped substations and generic generators (default)
        "low"    — includes everything (poles, towers, transformers)

    Example
    -------
        src = GeoFabrikSource("data/pbf/maine-latest.osm.pbf", min_confidence="high")
        df  = src.extract_all()
    """

    def __init__(self, pbf_path: str, min_confidence: str = "medium"):
        if not os.path.exists(pbf_path):
            raise FileNotFoundError(
                f"PBF file not found: {pbf_path}\n"
                f"Download from https://download.geofabrik.de/"
            )
        self.pbf_path       = pbf_path
        self.min_confidence = min_confidence
        self._target_types  = asset_types_by_confidence(min_confidence)
        print(f"Confidence filter: >= '{min_confidence}' "
              f"({len(self._target_types)} asset types active)")

    def _run(self, target_types: Optional[set] = None) -> pd.DataFrame:
        # intersect requested types with confidence filter
        active = self._target_types
        if target_types is not None:
            active = active & target_types
        handler = InfraHandler(target_types=active)
        handler.apply_file(self.pbf_path, locations=True)
        if not handler.rows:
            return pd.DataFrame()
        return pd.DataFrame(handler.rows)

    def extract_sector(self, sector: str) -> pd.DataFrame:
        """
        Extracts assets for a single sector (e.g. 'energy').
        Respects the min_confidence filter set at init.
        """
        if sector not in SECTORS:
            raise ValueError(
                f"Unknown sector '{sector}'. "
                f"Available: {list(SECTORS.keys())}"
            )
        print(f"  [{sector}] scanning {self.pbf_path}...")
        sector_types = set(SECTORS[sector])
        df = self._run(target_types=sector_types)
        if df.empty:
            print(f"  [{sector}] no assets found")
        else:
            print(f"  [{sector}] found {len(df)} assets")
            print(df["asset_type"].value_counts().to_string())
        return df

    def extract_all(self) -> pd.DataFrame:
        """
        Extracts assets for all defined sectors in a single PBF pass.
        Respects the min_confidence filter set at init.
        """
        print(f"Scanning {self.pbf_path}...")
        df = self._run(target_types=None)
        if df.empty:
            print("No assets found.")
            return df
        print(f"\nTotal assets: {len(df)}")
        print("\nCounts by asset type:")
        print(df["asset_type"].value_counts().to_string())
        return df


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# !! CHANGE OUTPUT_CSV before each new run to avoid overwriting earlier data.
#
# Convention: data/<region>_<scope>_assets.csv
# Examples:
#   "data/maine_all_assets.csv"
#   "data/new_hampshire_energy_assets.csv"
#   "data/northeast_usa_assets.csv"

OUTPUT_CSV = "data/maine_all_assets.csv"

# Path to your downloaded GeoFabrik PBF file.
# Download maine: https://download.geofabrik.de/north-america/us/maine-latest.osm.pbf

PBF_PATH = "data/pbf/maine-latest.osm.pbf"


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data/pbf", exist_ok=True)

    print(f"Output: {OUTPUT_CSV}")
    print(f"PBF:    {PBF_PATH}")
    print("(Change OUTPUT_CSV and PBF_PATH above before re-running.)\n")

    # "medium" skips poles and towers — good default for Sentinel-2 pipeline
    # use "low" to include everything, "high" for only the most distinctive assets
    src = GeoFabrikSource(PBF_PATH, min_confidence="medium")
    df  = src.extract_all()

    if not df.empty:
        df.drop(columns=["osm_tags"], errors="ignore").to_csv(
            OUTPUT_CSV, index=False
        )
        print(f"\nSaved {len(df)} assets to {OUTPUT_CSV}")