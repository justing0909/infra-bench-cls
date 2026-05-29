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
  - asset_id      : unique ID (e.g. "osm_node_123456" or "osm_relation_42")
  - asset_type    : ontology label (e.g. "energy.transmission.substation")
  - lat / lon     : centroid coordinates
  - name          : OSM name tag if available
  - source        : always "osm_geofabrik"
  - osm_tags      : dict of all OSM tags (for provenance)

Filter presets
--------------
The available presets are derived from `curation/ontology.py`:
  "full"       : every energy-sector class (legacy default)
  "substation" : transmission + distribution + minor + untyped substations
  "energy"     : same as "full" (clarified name)
  "water"      : water-sector classes
  "transport"  : transport-sector classes
  "telecom"    : telecom-sector classes

Recommended preset for current infra-FM paper runs is "substation"
(Ed's guidance: start small, ignore generators; CSDA handoff needs clean
substation bounding boxes).

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

from ontology import (
    AssetClass,
    ASSET_CLASSES,
    SECTORS as ONTOLOGY_SECTORS,
    get_class_by_name,
    get_classes_for_sector,
)


# ---------------------------------------------------------------------------
# Confidence ordering
# ---------------------------------------------------------------------------

CONFIDENCE_LEVELS = {"high": 3, "medium": 2, "low": 1}


# ---------------------------------------------------------------------------
# Filter presets — derived from the ontology registry
# ---------------------------------------------------------------------------
# Each preset is a tuple of AssetClass instances in declaration order.
# Order matters: matchers prefer the earliest matching class so the untyped
# substation catch-all only fires when the subtype-specific entries don't.

_ENERGY_CLASSES    = tuple(c for c in ASSET_CLASSES if c.sector == "energy")
_WATER_CLASSES     = tuple(c for c in ASSET_CLASSES if c.sector == "water")
_TRANSPORT_CLASSES = tuple(c for c in ASSET_CLASSES if c.sector == "transport")
_TELECOM_CLASSES   = tuple(c for c in ASSET_CLASSES if c.sector == "telecom")

_SUBSTATION_CLASSES = tuple(
    c for c in _ENERGY_CLASSES if "substation" in c.name
)

FILTER_PRESETS: dict[str, tuple[AssetClass, ...]] = {
    "full":       _ENERGY_CLASSES,        # legacy alias for energy-only
    "energy":     _ENERGY_CLASSES,
    "substation": _SUBSTATION_CLASSES,
    "water":      _WATER_CLASSES,
    "transport":  _TRANSPORT_CLASSES,
    "telecom":    _TELECOM_CLASSES,
}

# Group asset_type *names* by top-level sector (used by extract_sector()).
SECTORS: dict[str, list[str]] = {
    sector: [c.name for c in classes]
    for sector, classes in ONTOLOGY_SECTORS.items()
}


def asset_types_by_confidence(
    min_confidence: str = "medium",
    filter_preset:  str = "full",
) -> set:
    """
    Return the set of asset_type names in `filter_preset` whose confidence
    meets or exceeds `min_confidence`. Backwards-compatible with the prior
    dict-based implementation.
    """
    classes = FILTER_PRESETS.get(filter_preset, _ENERGY_CLASSES)
    threshold = CONFIDENCE_LEVELS.get(min_confidence, 2)
    return {
        c.name for c in classes
        if CONFIDENCE_LEVELS.get(c.confidence, 0) >= threshold
    }


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------
# These operate on raw OSM tag sets (pyosmium tags object) and ontology
# AssetClass instances. They are intentionally side-effect-free and take all
# inputs as arguments so they can be unit-tested without instantiating the
# osmium handler.

def _tags_all_match(tags, required: tuple) -> bool:
    """Every (key, value) in `required` must be present. value=None matches any value."""
    for key, value in required:
        tv = tags.get(key)
        if tv is None:
            return False
        if value is not None and tv != value:
            return False
    return True


def _tags_any_match(tags, candidates: tuple) -> bool:
    """At least one (key, value) in `candidates` must be present. Empty tuple = vacuously True."""
    if not candidates:
        return True
    for key, value in candidates:
        tv = tags.get(key)
        if tv is None:
            continue
        if value is None or tv == value:
            return True
    return False


def _tags_none_match(tags, forbidden: tuple) -> bool:
    """No (key, value) in `forbidden` may be present. Empty tuple = vacuously True."""
    if not forbidden:
        return True
    for key, value in forbidden:
        tv = tags.get(key)
        if tv is None:
            continue
        if value is None or tv == value:
            return False
    return True


def _geometry_allowed(cls: AssetClass, osm_type: str) -> bool:
    """osm_type is one of 'node', 'way', 'relation'."""
    rg = cls.require_geometry
    if rg == "any":
        return True
    if rg == osm_type:
        return True
    if rg == "way_or_relation" and osm_type in ("way", "relation"):
        return True
    return False


def _class_accepts(
    cls:        AssetClass,
    tags,
    osm_type:   str,
    area_m2:    Optional[float],
) -> bool:
    """Full accept/reject test for a single AssetClass against an OSM element."""
    if not _geometry_allowed(cls, osm_type):
        return False
    if not _tags_all_match(tags, cls.tags):
        return False
    if cls.any_of_tags and not _tags_any_match(tags, cls.any_of_tags):
        return False
    if not _tags_none_match(tags, cls.exclude_tags):
        return False
    if cls.min_area_m2 is not None:
        if area_m2 is None or area_m2 < cls.min_area_m2:
            return False
    return True


def _best_class(
    classes:    tuple,
    tags,
    osm_type:   str,
    area_m2:    Optional[float] = None,
) -> Optional[AssetClass]:
    """Returns the first matching AssetClass from `classes`, or None.

    `classes` must be iterated in priority order: subtype-specific entries
    (e.g. transmission.substation) before catch-alls (substation_untyped)."""
    for cls in classes:
        if _class_accepts(cls, tags, osm_type, area_m2):
            return cls
    return None


# Legacy alias retained for any external callers that grepped this name.
# Returns the matched AssetClass.name (str) rather than the dataclass.
def _best_asset_type(tags, classes) -> Optional[str]:
    cls = _best_class(tuple(classes), tags, "any", None)
    return cls.name if cls is not None else None


# ---------------------------------------------------------------------------
# Pre-filter routing
# ---------------------------------------------------------------------------

# Sector -> output filename label for the pre-filter PBF. "power" is kept
# for the energy sector so existing `*_power_only.osm.pbf` files continue
# to be picked up by _pre_filter_pbf_by_tags' "already exists" shortcut.
_SECTOR_TO_PREFILTER_NAME = {
    "energy":    "power",
    "water":     "water",
    "transport": "transport",
    "telecom":   "telecom",
}


def _derive_prefilter_sector(
    classes: tuple,
) -> tuple[str, list[str]]:
    """
    Returns (sector_name, sector_tag_keys) suitable for passing to
    `_pre_filter_pbf_by_tags`. Both are derived from the active class set:

      - sector_name: from the unique sector(s) of `classes`. For a single
        sector, uses the sector-specific label (energy -> "power"). For a
        mixed set (rare), uses an alphabetically joined name.

      - sector_tag_keys: the union of the FIRST tag key from each class's
        `tags` tuple. This is the "anchor" key that every matching element
        must carry. Empty `tags` (any_of_tags only) falls back to the first
        key of any_of_tags.
    """
    sectors = sorted({c.sector for c in classes})
    if len(sectors) == 1:
        sector_name = _SECTOR_TO_PREFILTER_NAME.get(sectors[0], sectors[0])
    else:
        sector_name = "_".join(sectors)

    keys: set = set()
    for c in classes:
        if c.tags:
            keys.add(c.tags[0][0])
        elif c.any_of_tags:
            keys.add(c.any_of_tags[0][0])
        # (Validator guarantees at least one is non-empty.)
    return sector_name, sorted(keys)


# ---------------------------------------------------------------------------
# Note on area handling
# ---------------------------------------------------------------------------
# pyosmium 4.x's `SimpleHandler.apply_file(filename, locations=True)` already
# does the right thing when the handler defines an `area()` callback:
# from the docstring on apply_file —
#     "If an area callback is implemented, then the file will be scanned
#      twice and a location handler and a handler for assembling
#      multipolygons and areas from ways will be executed."
# That means we get both closed-way areas AND multipolygon-relation areas
# without any manual AreaManager wiring. The `_needs_area` flag inside
# InfraHandler controls only:
#   - whether way() defers closed ways to area() (so area_m2 is available
#     before matching), and
#   - whether area() does any work or early-returns to save WKB+geod cost.


# ---------------------------------------------------------------------------
# pyosmium handler
# ---------------------------------------------------------------------------

class InfraHandler(osmium.SimpleHandler):
    """
    pyosmium handler that scans an .osm.pbf and collects elements matching
    any of `active_classes`. When at least one active class declares
    `min_area_m2`, the handler is driven in two-pass area-aware mode:

      - way() ignores closed ways (they will reappear via area()).
      - area() computes the ellipsoidal polygon area in m² via shapely +
        pyproj, then runs the full matcher with area_m2 supplied.

    Matched elements are deduplicated by (osm_type, osm_id) so closed ways
    that also surface through area() are not recorded twice.
    """

    def __init__(
        self,
        active_classes:     tuple,
        log_every:          int   = 1_000_000,
        estimated_nodes_m:  float = 0,
    ):
        super().__init__()
        self.rows: list[dict]     = []
        self._classes             = tuple(active_classes)
        self._needs_area          = any(
            c.min_area_m2 is not None for c in self._classes
        )
        # Deduplication key: (osm_type, osm_id).
        self._seen_ids: set       = set()

        # Lazy-init heavy geometry helpers only when area mode is active.
        self._wkb_factory         = None
        self._geod                = None

        # Progress / stats.
        self._log_every           = log_every
        self._n_nodes             = 0
        self._n_ways              = 0
        self._n_areas             = 0
        self._estimated_nodes_m   = estimated_nodes_m

    # ---- internal helpers ------------------------------------------------

    def _record(
        self,
        osm_type: str,
        osm_id:   int,
        lat:      float,
        lon:      float,
        tags,
        cls:      AssetClass,
    ) -> None:
        key = (osm_type, osm_id)
        if key in self._seen_ids:
            return
        self._seen_ids.add(key)

        tag_dict = {k: v for k, v in tags}
        self.rows.append({
            "asset_id":   f"osm_{osm_type}_{osm_id}",
            "asset_type": cls.name,
            "lat":        lat,
            "lon":        lon,
            "name":       tags.get("name", ""),
            "source":     "osm_geofabrik",
            "osm_tags":   tag_dict,
        })

    def _log_progress(self) -> None:
        current_m = self._n_nodes / 1e6
        suffix = (
            f"{self._n_ways / 1e3:.1f}k ways | "
            f"{self._n_areas / 1e3:.1f}k areas | "
            f"{len(self.rows)} assets matched so far"
        )
        if self._estimated_nodes_m > 0:
            pct = min(current_m / self._estimated_nodes_m * 100, 99.9)
            print(
                f"    {current_m:.0f}M / ~{self._estimated_nodes_m:.0f}M nodes "
                f"({pct:.0f}%) | {suffix}",
                flush=True,
            )
        else:
            print(
                f"    {current_m:.0f}M nodes scanned | {suffix}",
                flush=True,
            )

    # ---- osmium callbacks ------------------------------------------------

    def node(self, n):
        self._n_nodes += 1
        if self._n_nodes % self._log_every == 0:
            self._log_progress()

        if not n.location.valid():
            return

        cls = _best_class(self._classes, n.tags, "node", None)
        if cls is not None:
            self._record(
                "node", n.id,
                n.location.lat, n.location.lon,
                n.tags, cls,
            )

    def way(self, w):
        self._n_ways += 1

        # In area mode, defer closed ways to area() — they'll surface there
        # with an actual area_m2 value. Open (linear) ways must still be
        # processed here because area() does not fire for them.
        if self._needs_area and w.is_closed():
            return

        try:
            lats = [nd.location.lat for nd in w.nodes if nd.location.valid()]
            lons = [nd.location.lon for nd in w.nodes if nd.location.valid()]
            if not lats:
                return
            lat = sum(lats) / len(lats)
            lon = sum(lons) / len(lons)
        except Exception:
            return

        cls = _best_class(self._classes, w.tags, "way", None)
        if cls is not None:
            self._record("way", w.id, lat, lon, w.tags, cls)

    def area(self, a):
        """
        Fires only in area mode (after AreaManager has assembled multipolygon
        relations, plus closed-way auto-assembled areas).

        Note: pyosmium 4.x auto-fires `area()` for closed ways whenever node
        locations are loaded, even outside the AreaManager flow. When
        `_needs_area` is False we early-return so we don't pay the WKB +
        geod cost for nothing — closed ways are already handled in `way()`.
        """
        self._n_areas += 1
        if not self._needs_area:
            return

        if self._wkb_factory is None:
            self._wkb_factory = osmium.geom.WKBFactory()
        try:
            wkb_hex = self._wkb_factory.create_multipolygon(a)
            from shapely import wkb as shapely_wkb
            poly = shapely_wkb.loads(bytes.fromhex(wkb_hex))
        except Exception:
            return

        if self._geod is None:
            from pyproj import Geod
            self._geod = Geod(ellps="WGS84")
        try:
            # Geod returns a signed area; absolute value gives m².
            area_signed, _ = self._geod.geometry_area_perimeter(poly)
            area_m2 = abs(area_signed)
            centroid = poly.centroid
            lat, lon = centroid.y, centroid.x
        except Exception:
            return

        osm_type = "way" if a.from_way() else "relation"
        osm_id   = a.orig_id()

        cls = _best_class(self._classes, a.tags, osm_type, area_m2)
        if cls is not None:
            self._record(osm_type, osm_id, lat, lon, a.tags, cls)


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

        # The preset is an *ordered* tuple of AssetClass instances. Apply the
        # confidence threshold here so downstream code only sees in-scope
        # classes, but PRESERVE order — matchers depend on subtype-specific
        # entries appearing before catch-alls (e.g. transmission.substation
        # before substation_untyped).
        preset_classes = FILTER_PRESETS[filter_preset]
        threshold = CONFIDENCE_LEVELS.get(min_confidence, 2)
        self._active_classes: tuple = tuple(
            c for c in preset_classes
            if CONFIDENCE_LEVELS.get(c.confidence, 0) >= threshold
        )
        # Name-set view of the active classes (used by extract_sector for
        # intersection with sector membership).
        self._target_types: set = {c.name for c in self._active_classes}

        print(f"GeoFabrikSource: preset='{filter_preset}', "
              f"min_confidence='{min_confidence}', "
              f"target_types={sorted(self._target_types)}")

    def _region_name(self) -> str:
        return os.path.splitext(os.path.basename(self.pbf_path))[0]

    def _pre_filter_pbf_by_tags(
        self,
        sector_name: str,
        sector_tag_keys: list,
    ) -> str:
        """
        Writes a sector-filtered PBF subset to speed up scanning on large files.

        Parameters
        ----------
        sector_name : str
            Short label embedded in the output filename, e.g. "power".
            The output PBF is written to "{base}_{sector_name}_only.osm.pbf".
        sector_tag_keys : list of str
            OSM tag keys that mark an element as in-sector.
            An element matches if ANY of these keys is present in its tags.
            Examples:
              ["power"]
              ["waterway", "water", "man_made"]
              ["highway", "railway", "aeroway"]

        Pass 1 collects:
          - IDs of nodes/ways/relations whose tags include any sector_tag_keys.
          - IDs of nodes referenced by matched ways (via w.nodes).
          - IDs of nodes referenced by matched relations (members of type 'n').
            Nested relation membership is intentionally not traversed.

        Pass 2 writes:
          - every node that is either tagged or referenced by a matched
            way/relation,
          - every matched way and matched relation.

        Returns path to the filtered PBF, or the original path on failure.
        """
        import time
        base = os.path.splitext(self.pbf_path)[0]
        out_path = f"{base}_{sector_name}_only.osm.pbf"

        if os.path.exists(out_path):
            print(f"  Using existing {sector_name}-only PBF: {out_path}")
            return out_path

        print(f"  Pre-filtering {self.pbf_path} -> {out_path} "
              f"(sector='{sector_name}', tag_keys={sector_tag_keys})")
        t0 = time.time()

        # Local copy for fast membership tests inside the handler classes.
        _tag_keys = tuple(sector_tag_keys)

        try:
            class IDCollector(osmium.SimpleHandler):
                def __init__(self):
                    super().__init__()
                    self.tagged_node_ids     = set()
                    self.tagged_way_ids      = set()
                    self.tagged_relation_ids = set()
                    # Nodes referenced by matched ways/relations.
                    # Kept in a separate set so we can report both numbers.
                    self.member_node_ids     = set()

                @staticmethod
                def _matches(tags) -> bool:
                    for key in _tag_keys:
                        if key in tags:
                            return True
                    return False

                def node(self, n):
                    if self._matches(n.tags):
                        self.tagged_node_ids.add(n.id)

                def way(self, w):
                    if self._matches(w.tags):
                        self.tagged_way_ids.add(w.id)
                        # Retain referenced nodes so the output remains valid.
                        for nd in w.nodes:
                            self.member_node_ids.add(nd.ref)

                def relation(self, r):
                    if self._matches(r.tags):
                        self.tagged_relation_ids.add(r.id)
                        # Only direct node members. Way/relation members are
                        # not followed — keep this pass simple and correct.
                        for m in r.members:
                            if m.type == 'n':
                                self.member_node_ids.add(m.ref)

            print(f"  Pass 1: collecting matched element IDs + member node refs...")
            collector = IDCollector()
            collector.apply_file(self.pbf_path)

            n_tagged_nodes      = len(collector.tagged_node_ids)
            n_member_only_nodes = len(
                collector.member_node_ids - collector.tagged_node_ids
            )
            n_keep_nodes        = n_tagged_nodes + n_member_only_nodes
            print(f"  Pass 1 complete: "
                  f"{n_tagged_nodes:,} tagged nodes + "
                  f"{n_member_only_nodes:,} member-only nodes "
                  f"= {n_keep_nodes:,} nodes to keep, "
                  f"{len(collector.tagged_way_ids):,} ways, "
                  f"{len(collector.tagged_relation_ids):,} relations")

            # Union the two sets for fast O(1) lookup during Pass 2.
            keep_node_ids = collector.tagged_node_ids | collector.member_node_ids

            writer = osmium.SimpleWriter(out_path)

            class FilterWriter(osmium.SimpleHandler):
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
                    if n.id in self.node_ids:
                        writer.add_node(n)
                        self.n_written += 1

                def way(self, w):
                    if w.id in self.way_ids:
                        writer.add_way(w)
                        self.n_written += 1

                def relation(self, r):
                    if r.id in self.relation_ids:
                        writer.add_relation(r)
                        self.n_written += 1

            print(f"  Pass 2: writing matched elements and their referenced nodes...")
            fw = FilterWriter(
                keep_node_ids,
                collector.tagged_way_ids,
                collector.tagged_relation_ids,
            )
            fw.apply_file(self.pbf_path)
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

        # Effective class set for this run = active classes (preset +
        # confidence threshold) intersected with `target_types` if supplied.
        if target_types is None:
            effective_classes = self._active_classes
        else:
            effective_classes = tuple(
                c for c in self._active_classes if c.name in target_types
            )

        if not effective_classes:
            print("  No active classes for this run; returning empty DataFrame.")
            return pd.DataFrame()

        # Pre-filter sector name + tag keys derived from the effective
        # classes. The pre-filter step itself is unchanged from Part 1 —
        # this just routes the right keys to it.
        sector_name, sector_tag_keys = _derive_prefilter_sector(
            effective_classes
        )

        if self.pre_filter:
            scan_path = self._pre_filter_pbf_by_tags(
                sector_name=sector_name,
                sector_tag_keys=sector_tag_keys,
            )
        else:
            scan_path = self.pbf_path

        # Estimate node count from file size.
        # ~850 bytes/node is a reasonable estimate for power-only PBFs;
        # the same heuristic works ballpark for other sector subsets.
        size_bytes        = os.path.getsize(scan_path)
        size_mb           = size_bytes / 1_048_576
        estimated_nodes_m = (size_bytes / 850) / 1_000_000

        print(f"  Scanning {scan_path} (preset='{self.filter_preset}')")
        print(f"  File size: {size_mb:.1f}MB | "
              f"estimated ~{estimated_nodes_m:.0f}M nodes | "
              f"progress every 1M nodes")

        handler = InfraHandler(
            active_classes=effective_classes,
            estimated_nodes_m=estimated_nodes_m,
        )

        if handler._needs_area:
            print(f"  Area mode: at least one active class has min_area_m2 — "
                  f"area() callback will compute polygon area in m².")

        t0 = time.time()
        # pyosmium 4.x auto-detects the area() callback and runs a two-pass
        # scan internally when locations=True. The `_needs_area` flag only
        # influences whether area() does work or early-returns.
        #
        # NOTE: an earlier version of this code passed
        #     idx="sparse_file_array,locations.idx"
        # to use a disk-backed location index. That parameter silently drops
        # node locations on some inputs, which then breaks both way()
        # centroid calc AND multipolygon-area assembly. Symptoms observed
        # on central-america transport: only 1 airport matched instead of
        # 88 (≥50 ha), 0 ports instead of 2. The pre-filter output is
        # always small enough that in-memory locations are affordable
        # (a few MB up to ~50 MB), so we just use the default.
        handler.apply_file(scan_path, locations=True)

        elapsed = time.time() - t0
        print(f"  Scan complete in {elapsed:.1f}s — "
              f"{handler._n_nodes:,} nodes, "
              f"{handler._n_ways:,} ways, "
              f"{handler._n_areas:,} areas, "
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

OUTPUT_PARQUET = "data/asia_all_assets.parquet"
PBF_PATH       = "data/pbf/asia-260408.osm.pbf"


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("data/pbf", exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PARQUET) or ".", exist_ok=True)

    print(f"Output: {OUTPUT_PARQUET}")
    print(f"PBF:    {PBF_PATH}")
    print("(Change OUTPUT_PARQUET and PBF_PATH above before re-running.)\n")

    # filter_preset="substation" is the recommended default for infra-FM runs.
    # Switch to "full" (or any sector preset) to broaden the scope.
    src = GeoFabrikSource(PBF_PATH,
                          min_confidence="medium",
                          filter_preset="substation")
    df = src.extract_all()

    if not df.empty:
        df.drop(columns=["osm_tags"], errors="ignore").to_parquet(
            OUTPUT_PARQUET, index=False
        )
        print(f"\nSaved {len(df)} assets to {OUTPUT_PARQUET}")