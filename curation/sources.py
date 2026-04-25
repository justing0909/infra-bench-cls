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

Each query returns a pandas DataFrame with one row per asset:
  - asset_id      : unique ID (e.g. "osm_node_123456")
  - asset_type    : ontology label (e.g. "energy.transmission.substation")
  - lat / lon     : centroid coordinates
  - name          : OSM name tag if available
  - source        : always "osm_geofabrik"
  - osm_tags      : dict of all OSM tags (for provenance)

Filter presets
--------------
"full"       : all asset types defined in ASSET_FILTERS (energy sector)
"substation" : transmission + distribution substations only — recommended
               for the current infra-FM paper scope (Ed's guidance: start
               small, ignore generators; CSDA handoff needs clean substation
               bounding boxes)

Usage:
    from sources import GeoFabrikSource
    src = GeoFabrikSource("data/pbf/asia-260408.osm.pbf",
                          filter_preset="substation")
    df = src.extract_all()
"""

import os
import osmium
import pandas as pd
from typing import Optional
from utils.timing_log_utils import update_timing_log, file_size_kb, file_size_mb


# ---------------------------------------------------------------------------
# Asset filter definitions — FULL ontology
# ---------------------------------------------------------------------------
# Maps ontology asset_type to OSM tag filters and visual confidence.
# Order matters: more specific filters must come before less specific ones.

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
        "confidence": "low",
    },

    # --- Energy: Distribution ---
    "energy.distribution.substation": {
        "tags": [("power", "substation"), ("substation", "distribution")],
        "confidence": "high",
    },
    "energy.distribution.substation_untyped": {
        "tags": [("power", "substation")],
        "confidence": "medium",
    },
    "energy.distribution.transformer": {
        "tags": [("power", "transformer")],
        "confidence": "low",
    },
    "energy.distribution.pole": {
        "tags": [("power", "pole")],
        "confidence": "low",
    },
}


# ---------------------------------------------------------------------------
# SUBSTATION_FILTERS — recommended preset for infra-FM paper runs
# ---------------------------------------------------------------------------
# Transmission substations, distribution substations, and untyped substations.
# Excludes generators, solar, wind, towers, poles, and lines.
#
# Rationale:
#   - Substations are network-critical nodes whose failure propagates
#   - Visually consistent at both Sentinel-2 (10m) and CSDA 30cm scales
#   - Manageable asset counts globally (vs. poles which number in the millions)
#   - Clean bounding boxes for future NASA CSDA 30cm handoff

SUBSTATION_FILTERS = {
    "energy.transmission.substation": {
        "tags": [("power", "substation"), ("substation", "transmission")],
        "confidence": "high",
    },
    "energy.distribution.substation": {
        "tags": [("power", "substation"), ("substation", "distribution")],
        "confidence": "high",
    },
    "energy.distribution.substation_untyped": {
        "tags": [("power", "substation")],
        "confidence": "medium",
    },
}

# Registry of available presets
FILTER_PRESETS = {
    "full": ASSET_FILTERS,
    "substation": SUBSTATION_FILTERS,
}

# Confidence level ordering for threshold filtering
CONFIDENCE_LEVELS = {"high": 3, "medium": 2, "low": 1}

# Group asset_types by top-level sector
SECTORS = {
    "energy": [k for k in ASSET_FILTERS if k.startswith("energy.")],
}


def asset_types_by_confidence(min_confidence: str = "medium",
                               filter_preset: str = "full") -> set:
    """
    Returns the set of asset_types at or above the given confidence threshold
    within the specified filter preset.
    """
    filters = FILTER_PRESETS.get(filter_preset, ASSET_FILTERS)
    threshold = CONFIDENCE_LEVELS.get(min_confidence, 2)
    return {
        k for k, v in filters.items()
        if CONFIDENCE_LEVELS.get(v["confidence"], 0) >= threshold
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches(tags, filter_list: list) -> bool:
    for key, value in filter_list:
        tag_val = tags.get(key)
        if tag_val is None:
            return False
        if value is not None and tag_val != value:
            return False
    return True


def _best_asset_type(tags, filters: dict) -> Optional[str]:
    """
    Returns the most specific matching asset_type for an OSM element's tags,
    or None if no filter matches. Uses the provided filters dict (not global).
    """
    for asset_type, defn in filters.items():
        if _matches(tags, defn["tags"]):
            return asset_type
    return None


# ---------------------------------------------------------------------------
# pyosmium handler
# ---------------------------------------------------------------------------

class InfraHandler(osmium.SimpleHandler):
    """
    pyosmium handler that scans nodes and ways in a .osm.pbf file and
    collects those matching the provided filters dict.
    """

    def __init__(self, filters: dict,
                 target_types: Optional[set] = None,
                 log_every: int = 1_000_000,
                 estimated_nodes_m: float = 0):
        super().__init__()
        self.rows                = []
        self._filters            = filters
        self._target_types       = target_types
        self._log_every          = log_every
        self._n_nodes            = 0
        self._n_ways             = 0
        self._estimated_nodes_m  = estimated_nodes_m

    def _add(self, osm_type: str, osm_id: int,
             lat: float, lon: float, tags) -> None:
        asset_type = _best_asset_type(tags, self._filters)
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
            "name":       tags.get("name", ""),
            "source":     "osm_geofabrik",
            "osm_tags":   tag_dict,
        })

    def node(self, n):
        self._n_nodes += 1
        if self._n_nodes % self._log_every == 0:
            current_m = self._n_nodes / 1e6
            if self._estimated_nodes_m > 0:
                pct = min(current_m / self._estimated_nodes_m * 100, 99.9)
                print(
                    f"    {current_m:.0f}M / ~{self._estimated_nodes_m:.0f}M nodes "
                    f"({pct:.0f}%) | "
                    f"{self._n_ways / 1e3:.1f}k ways | "
                    f"{len(self.rows)} assets matched so far",
                    flush=True,
                )
            else:
                print(
                    f"    {current_m:.0f}M nodes scanned | "
                    f"{self._n_ways / 1e3:.1f}k ways | "
                    f"{len(self.rows)} assets matched so far",
                    flush=True,
                )
        if not n.location.valid():
            return
        self._add("node", n.id, n.location.lat, n.location.lon, n.tags)

    def way(self, w):
        self._n_ways += 1
        try:
            lats = [nd.location.lat for nd in w.nodes if nd.location.valid()]
            lons = [nd.location.lon for nd in w.nodes if nd.location.valid()]
            if not lats:
                return
            lat = sum(lats) / len(lats)
            lon = sum(lons) / len(lons)
            self._add("way", w.id, lat, lon, w.tags)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main source class
# ---------------------------------------------------------------------------

class GeoFabrikSource:
    """
    Extracts infrastructure assets from a GeoFabrik .osm.pbf file.

    Parameters
    ----------
    pbf_path        : str  — path to local .osm.pbf file
    min_confidence  : str  — "high", "medium", or "low"
    filter_preset   : str  — "full" or "substation"
                      Use "substation" for infra-FM paper runs.
    pre_filter      : bool — write a power-only PBF before scanning
                      (dramatically faster for continent-scale files)
    """

    def __init__(self, pbf_path: str,
                 min_confidence: str = "medium",
                 filter_preset: str = "substation",
                 pre_filter: bool = True):
        self.pbf_path        = pbf_path
        self.min_confidence  = min_confidence
        self.filter_preset   = filter_preset
        self.pre_filter      = pre_filter

        if filter_preset not in FILTER_PRESETS:
            raise ValueError(
                f"Unknown filter_preset '{filter_preset}'. "
                f"Available: {list(FILTER_PRESETS.keys())}"
            )

        self._active_filters = FILTER_PRESETS[filter_preset]

        # Apply confidence threshold within the chosen preset
        threshold = CONFIDENCE_LEVELS.get(min_confidence, 2)
        self._target_types = {
            k for k, v in self._active_filters.items()
            if CONFIDENCE_LEVELS.get(v["confidence"], 0) >= threshold
        }

        print(f"GeoFabrikSource: preset='{filter_preset}', "
              f"min_confidence='{min_confidence}', "
              f"target_types={sorted(self._target_types)}")

    def _region_name(self) -> str:
        return os.path.splitext(os.path.basename(self.pbf_path))[0]

    def _pre_filter_pbf(self) -> str:
        """
        Writes a power-only PBF to speed up scanning on large files.
        Returns path to the filtered PBF, or original path on failure.
        """
        import time
        base = os.path.splitext(self.pbf_path)[0]
        out_path = f"{base}_power_only.osm.pbf"

        if os.path.exists(out_path):
            print(f"  Using existing power-only PBF: {out_path}")
            return out_path

        print(f"  Pre-filtering {self.pbf_path} -> {out_path}...")
        t0 = time.time()

        try:
            class IDCollector(osmium.SimpleHandler):
                def __init__(self):
                    super().__init__()
                    self.power_node_ids     = set()
                    self.power_way_ids      = set()
                    self.power_relation_ids = set()

                def node(self, n):
                    if "power" in n.tags:
                        self.power_node_ids.add(n.id)

                def way(self, w):
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
            print(f"  Pre-filter complete in {elapsed:.1f}s -> "
                  f"{size_mb:.1f}MB ({out_path})")

            try:
                update_timing_log(
                    workbook_path="Infra-FM-timing-log.xlsx",
                    region=self._region_name(),
                    starting_file_size_kb=file_size_kb(self.pbf_path),
                    pre_filter_time_s=round(elapsed, 2),
                    power_only_file_size_mb=round(size_mb, 2),
                )
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

        scan_path = self._pre_filter_pbf() if self.pre_filter else self.pbf_path

        active = self._target_types
        if target_types is not None:
            active = active & target_types

        # Estimate node count from file size
        # ~850 bytes/node is a reasonable estimate for power-only PBFs
        size_bytes        = os.path.getsize(scan_path)
        size_mb           = size_bytes / 1_048_576
        estimated_nodes_m = (size_bytes / 850) / 1_000_000

        print(f"  Scanning {scan_path} (preset='{self.filter_preset}')")
        print(f"  File size: {size_mb:.1f}MB | "
              f"estimated ~{estimated_nodes_m:.0f}M nodes | "
              f"progress every 1M nodes")

        handler = InfraHandler(
            filters=self._active_filters,
            target_types=active,
            estimated_nodes_m=estimated_nodes_m,
        )
        t0 = time.time()

        try:
            lhandler = osmium.NodeLocationsForWays(handler)
            lhandler.ignore_errors()
            lhandler.apply_file(scan_path, locations=True,
                                idx="sparse_file_array,locations.idx")
        except Exception:
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
        except Exception as e:
            print(f"  Warning: could not update timing log with scan stats: {e}")

        if not handler.rows:
            return pd.DataFrame()
        return pd.DataFrame(handler.rows)

    def extract_all(self) -> pd.DataFrame:
        print(f"Scanning {self.pbf_path} "
              f"(preset='{self.filter_preset}')...")
        df = self._run(target_types=None)
        if df.empty:
            print("No assets found.")
            return df
        print(f"\nTotal assets: {len(df)}")
        print("\nCounts by asset type:")
        print(df["asset_type"].value_counts().to_string())
        return df

    def extract_sector(self, sector: str) -> pd.DataFrame:
        if sector not in SECTORS:
            raise ValueError(
                f"Unknown sector '{sector}'. "
                f"Available: {list(SECTORS.keys())}"
            )
        print(f"  [{sector}] scanning {self.pbf_path}...")
        sector_types = set(SECTORS[sector]) & self._target_types
        df = self._run(target_types=sector_types)
        if df.empty:
            print(f"  [{sector}] no assets found")
        else:
            print(f"  [{sector}] found {len(df)} assets")
            print(df["asset_type"].value_counts().to_string())
        return df


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_CSV = "data/asia_all_assets.csv"
PBF_PATH   = "data/pbf/asia-260408.osm.pbf"


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data/pbf", exist_ok=True)

    print(f"Output: {OUTPUT_CSV}")
    print(f"PBF:    {PBF_PATH}")
    print("(Change OUTPUT_CSV and PBF_PATH above before re-running.)\n")

    # filter_preset="substation" is the recommended default for infra-FM runs.
    # Switch to "full" to restore the broader ontology.
    src = GeoFabrikSource(PBF_PATH,
                          min_confidence="medium",
                          filter_preset="substation")
    df = src.extract_all()

    if not df.empty:
        df.drop(columns=["osm_tags"], errors="ignore").to_csv(
            OUTPUT_CSV, index=False
        )
        print(f"\nSaved {len(df)} assets to {OUTPUT_CSV}")