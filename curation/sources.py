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
    src = GeoFabrikSource("data/pbf/asia-260408.osm.pbf")
    df = src.extract_all()
    df = src.extract_sector("energy")   # single sector only
"""

import os
import osmium
import pandas as pd
from typing import Optional
from utils.timing_log_utils import update_timing_log, file_size_kb, file_size_mb


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

    def __init__(self, target_types: Optional[set] = None,
                 log_every: int = 1_000_000):
        super().__init__()
        self.rows          = []
        self._target_types = target_types
        self._log_every    = log_every
        self._n_nodes      = 0
        self._n_ways       = 0

    def _add(self, osm_type: str, osm_id: int,
             lat: float, lon: float, tags) -> None:
        asset_type = _best_asset_type(tags)
        if asset_type is None:
            return
        if self._target_types and asset_type not in self._target_types:
            return

        # --- Solar farm filter ---
        # OSM has tens of thousands of individual rooftop/small solar
        # installations tagged power=generator, generator:source=solar.
        # For a training corpus we only want utility-scale farms that are
        # visually distinctive at Sentinel-2 (10m) resolution.
        # Keep solar only if:
        #   - it's a way (polygon footprint, not a point)
        #   - OR it has a rated output tag suggesting utility scale
        #   - OR it has a plant:output tag
        if asset_type == "energy.generation.solar_farm":
            tag_dict_check = {k: v for k, v in tags}
            has_output = (
                "generator:output:electricity" in tag_dict_check
                or "plant:output:electricity" in tag_dict_check
                or "generator:output" in tag_dict_check
            )
            is_way = osm_type == "way"
            if not is_way and not has_output:
                return  # skip small rooftop / node-mapped solar

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
        self._n_nodes += 1
        if self._n_nodes % self._log_every == 0:
            print(f"    nodes scanned: {self._n_nodes:,}  "
                  f"ways scanned: {self._n_ways:,}  "
                  f"assets found: {len(self.rows):,}")
        self._add("node", n.id, n.location.lat, n.location.lon, n.tags)

    def way(self, w):
        self._n_ways += 1
        if self._n_ways % self._log_every == 0:
            print(f"    nodes scanned: {self._n_nodes:,}  "
                  f"ways scanned: {self._n_ways:,}  "
                  f"assets found: {len(self.rows):,}")

        # Only process ways that have power tags
        if "power" not in w.tags:
            return

        try:
            # Try midpoint node first
            mid = w.nodes[len(w.nodes) // 2]
            if mid.location.valid():
                self._add("way", w.id,
                          mid.location.lat, mid.location.lon, w.tags)
                return

            # Fallback: try any valid node location in the way
            for node in w.nodes:
                if node.location.valid():
                    self._add("way", w.id,
                              node.location.lat, node.location.lon, w.tags)
                    return

            # Fallback: use envelope/bounds if available
            if hasattr(w, "envelope"):
                env = w.envelope
                lat = (env.bottom_left.lat + env.top_right.lat) / 2
                lon = (env.bottom_left.lon + env.top_right.lon) / 2
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    self._add("way", w.id, lat, lon, w.tags)

        except Exception:
            pass

    def relation(self, r):
        """Handle relations — many substations are mapped as relations."""
        try:
            # Use centroid of member nodes as proxy location
            lats, lons = [], []
            for m in r.members:
                if m.type == "n":   # node member
                    try:
                        loc = m.ref  # node ID — location not directly available
                        # Skip — we can't get location from relation members
                        # without a full node index
                        pass
                    except Exception:
                        pass
            # Fall back: use bounding box center if available
            if hasattr(r, "envelope") and r.envelope:
                env = r.envelope
                lat = (env.bottom_left.lat + env.top_right.lat) / 2
                lon = (env.bottom_left.lon + env.top_right.lon) / 2
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    self._add("relation", r.id, lat, lon, r.tags)
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
        src = GeoFabrikSource("data/pbf/asia-260408.osm.pbf", min_confidence="high")
        df  = src.extract_all()
    """

    def __init__(self, pbf_path: str, min_confidence: str = "medium",
                 pre_filter: bool = True):
        if not os.path.exists(pbf_path):
            raise FileNotFoundError(
                f"PBF file not found: {pbf_path}\n"
                f"Download from https://download.geofabrik.de/"
            )
        self.pbf_path       = pbf_path
        self.min_confidence = min_confidence
        self.pre_filter     = pre_filter
        self._target_types  = asset_types_by_confidence(min_confidence)
        print(f"Confidence filter: >= '{min_confidence}' "
              f"({len(self._target_types)} asset types active)")
        
    def _region_name(self) -> str:
        filename = os.path.basename(self.pbf_path)
        filename = filename.replace(".osm_power_only.osm.pbf", "")
        filename = filename.replace(".osm.pbf", "")
        return filename

    def _pre_filter_pbf(self) -> str:
        """
        Pre-filters the PBF to power=* elements including all referenced
        node locations so ways can have their centroids resolved.

        Two-pass approach:
          Pass 1: collect IDs of all power nodes, ways, relations
          Pass 2: write those elements plus ALL nodes (for location lookup)

        Returns path to filtered PBF (cached). Skips if already exists.
        """
        import time

        base     = os.path.splitext(self.pbf_path)[0]
        out_path = f"{base}_power_only.osm.pbf"

        if os.path.exists(out_path):
            print(f"  Using cached pre-filtered PBF: {out_path}")
            try:
                update_timing_log(
                    workbook_path="Infra-FM-timing-log.xlsx",
                    region=self._region_name(),
                    starting_file_size_kb=file_size_kb(self.pbf_path),
                    power_only_file_size_mb=file_size_mb(out_path),
                )
                print("  Updated timing log from cached power_only file.")
            except Exception as e:
                print(f"  Warning: could not update timing log from cached file: {e}")
            return out_path

        print(f"  Pre-filtering PBF to power=* elements (2-pass)...")
        print(f"  (This runs once and is cached for future runs)")
        t0 = time.time()

        try:
            # Pass 1: collect power element IDs and ALL node IDs
            # We need all nodes so location lookup works for ways
            class IDCollector(osmium.SimpleHandler):
                def __init__(self):
                    super().__init__()
                    self.power_node_ids     = set()
                    self.power_way_ids      = set()
                    self.power_relation_ids = set()
                    self._n_scanned         = 0

                def node(self, n):
                    self._n_scanned += 1
                    if self._n_scanned % 5_000_000 == 0:
                        print(f"    pass 1: {self._n_scanned/1e6:.0f}M elements, "
                              f"{len(self.power_node_ids) + len(self.power_way_ids):,} power found")
                    if "power" in n.tags:
                        self.power_node_ids.add(n.id)

                def way(self, w):
                    self._n_scanned += 1
                    if self._n_scanned % 5_000_000 == 0:
                        print(f"    pass 1: {self._n_scanned/1e6:.0f}M elements, "
                              f"{len(self.power_node_ids) + len(self.power_way_ids):,} power found")
                    if "power" in w.tags:
                        self.power_way_ids.add(w.id)

                def relation(self, r):
                    if "power" in r.tags:
                        self.power_relation_ids.add(r.id)

            print(f"  Pass 1: collecting power element IDs...")
            collector = IDCollector()
            collector.apply_file(self.pbf_path)
            print(f"  Pass 1 complete: {len(collector.power_node_ids):,} nodes, "
                  f"{len(collector.power_way_ids):,} ways, "
                  f"{len(collector.power_relation_ids):,} relations")

            # Pass 2: write power elements + ALL nodes (for location lookup)
            writer = osmium.SimpleWriter(out_path)

            class PowerWriter(osmium.SimpleHandler):
                def __init__(self, node_ids, way_ids, relation_ids):
                    super().__init__()
                    self.node_ids     = node_ids
                    self.way_ids      = way_ids
                    self.relation_ids = relation_ids
                    self.n_written    = 0
                    self._n_scanned   = 0

                def node(self, n):
                    self._n_scanned += 1
                    if self._n_scanned % 5_000_000 == 0:
                        print(f"    pass 2: {self._n_scanned/1e6:.0f}M elements, "
                              f"{self.n_written:,} written")
                    # Write ALL nodes — needed for way location resolution
                    writer.add_node(n)
                    if n.id in self.node_ids:
                        self.n_written += 1

                def way(self, w):
                    if w.id in self.way_ids:
                        writer.add_way(w)
                        self.n_written += 1

                def relation(self, r):
                    if r.id in self.relation_ids:
                        writer.add_relation(r)
                        self.n_written += 1

            print(f"  Pass 2: writing filtered PBF with all nodes...")
            pw = PowerWriter(
                collector.power_node_ids,
                collector.power_way_ids,
                collector.power_relation_ids,
            )
            pw.apply_file(self.pbf_path)
            writer.close()

            elapsed = time.time() - t0
            size_mb = os.path.getsize(out_path) / 1_048_576
            print(f"  Pre-filter complete in {elapsed:.1f}s → "
                  f"{size_mb:.1f}MB ({out_path})")

            try:
                update_timing_log(
                    workbook_path="Infra-FM-timing-log.xlsx",
                    region=self._region_name(),
                    starting_file_size_kb=file_size_kb(self.pbf_path),
                    pre_filter_time_s=round(elapsed, 2),
                    power_only_file_size_mb=round(size_mb, 2),
                )
                print("  Updated timing log with pre-filter stats.")
            except Exception as e:
                print(f"  Warning: could not update timing log: {e}")

            return out_path

        except Exception as e:
            print(f"  Pre-filter failed ({e}), falling back to full scan")
            if os.path.exists(out_path):
                os.remove(out_path)
            return self.pbf_path

    def _run(self, target_types: Optional[set] = None) -> pd.DataFrame:
        import time

        # Use pre-filtered PBF if enabled — much faster for large files
        scan_path = self._pre_filter_pbf() if self.pre_filter else self.pbf_path

        active = self._target_types
        if target_types is not None:
            active = active & target_types
        handler = InfraHandler(target_types=active)
        print(f"  Scanning {scan_path}... (progress every 1M nodes/ways)")
        t0 = time.time()

        # Use disk-backed node location index for large files.
        # locations=True uses an in-memory index which requires enormous RAM
        # for continent-scale PBFs and rebuilds on every restart.
        # NodeLocationsForWays with a sparse disk index is much more efficient.
        try:
            lhandler = osmium.NodeLocationsForWays(handler)
            lhandler.ignore_errors()
            lhandler.apply_file(scan_path, locations=True,
                                idx="sparse_file_array,locations.idx")
        except Exception:
            # Fallback: standard locations=True (works for smaller files)
            handler.apply_file(scan_path, locations=True)

        elapsed = time.time() - t0
        print(f"  Scan complete in {elapsed:.1f}s — "
              f"{handler._n_nodes:,} nodes, "
              f"{handler._n_ways:,} ways, "
              f"{len(handler.rows):,} assets matched")
        try:
            update_timing_log(
                workbook_path="Infra-FM-timing-log.xlsx",
                region=self._region_name(),
                scanning_time_s=round(elapsed, 2),
                assets_extracted=len(handler.rows),
            )
            print("  Updated timing log with scan stats.")

        except Exception as e:
            print(f"  Warning: could not update timing log with scan stats: {e}")
        way_rows = [r for r in handler.rows if "_way_" in r["asset_id"]]
        node_rows = [r for r in handler.rows if "_node_" in r["asset_id"]]
        print(f"  Node-origin: {len(node_rows)}, Way-origin: {len(way_rows)}")
        if way_rows:
            lats = [r["lat"] for r in way_rows]
            print(f"  Way lat range: {min(lats):.4f} to {max(lats):.4f}")
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

OUTPUT_CSV = "data/asia_all_assets.csv"

# Path to your downloaded GeoFabrik PBF file.
# Download asia: https://download.geofabrik.de/asia/asia-latest.osm.pbf

PBF_PATH = "data/pbf/asia-260408.osm.pbf"


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