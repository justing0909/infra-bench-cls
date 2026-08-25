"""
ontology.py
-----------
canonical asset ontology for the multi-sector infrastructure pipeline.

each ontology class declares:
  - canonical `name` (dotted, e.g. "energy.transmission.substation")
  - `sector` ("energy" | "water" | "transport" | "telecom")
  - `tags`           : tuple of (key, value) tuples; ALL must match.
                       a value of `None` requires only key presence.
  - `any_of_tags`    : tuple of (key, value) tuples;
                       AT LEAST ONE must match (if the tuple is non-empty).
  - `exclude_tags`   : tuple of (key, value) tuples;
                       if ANY matches, the element is rejected.
  - `require_geometry`: "node" | "way" | "relation" | "way_or_relation" | "any"
                       default "any". when `min_area_m2` is set, this MUST be
                       one of "way", "relation", or "way_or_relation" — nodes
                       have no area.
  - `min_area_m2`    : optional numeric minimum area, applied only to way and
                       relation geometries.
  - `confidence`     : "high" | "medium" | "low"

naming convention follows the existing pipeline + ONTOLOGY.md
(`energy.transmission.substation`, not `energy.substation.transmission`) so
existing parquet labels and downstream consumers keep working.

order within a sector is meaningful: matchers should prefer the earlier
entry when multiple classes could match a single OSM element (e.g. a
substation tagged with both `substation=transmission` and `power=substation`
should match `energy.transmission.substation`, not the untyped fallback).

this module exposes:
  - ASSET_CLASSES           : tuple of every AssetClass instance
  - SECTORS                 : dict[sector_name -> tuple[AssetClass, ...]]
  - get_classes_for_sector(sector_name) -> tuple[AssetClass, ...]
  - get_class_by_name(name) -> AssetClass

this is the FIRST revision. many classes from ONTOLOGY.md are intentionally
omitted (lines, towers, generators, parking, rooftop telecom, etc.) pending
sector-specific verification. extend as those classes are reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# type aliases and constants
# ---------------------------------------------------------------------------

# (key, value). a value of None means "any value as long as the key exists".
TagTuple = tuple[str, Optional[str]]

VALID_SECTORS    = ("energy", "water", "transport", "telecom")
VALID_GEOMETRIES = ("any", "node", "way", "relation", "way_or_relation")
VALID_CONFIDENCE = ("high", "medium", "low")

# geometries for which `min_area_m2` is meaningful. setting `min_area_m2`
# with any other `require_geometry` value is rejected at construction time.
GEOMETRIES_WITH_AREA = ("way", "relation", "way_or_relation")


# ---------------------------------------------------------------------------
# AssetClass dataclass
# ---------------------------------------------------------------------------

# default dedup threshold if a class doesn't set `dedup_distance_m`.
# 200m matches the historical Deduplicator default and works well for
# typical substation-sized facilities. override per class when the
# physical scale of the asset is meaningfully different (huge solar
# farms; tiny subway-platform train stations).
DEFAULT_DEDUP_DISTANCE_M = 200.0


@dataclass(frozen=True)
class AssetClass:
    """
    declarative spec for one ontology class.

    frozen so the registry can be passed around without accidental mutation,
    and so a class instance is hashable (useful for set-membership and as a
    dict key in downstream code).
    """
    name             : str
    sector           : str
    tags             : tuple[TagTuple, ...] = ()
    any_of_tags      : tuple[TagTuple, ...] = ()
    exclude_tags     : tuple[TagTuple, ...] = ()
    require_geometry : str                  = "any"
    min_area_m2      : Optional[float]      = None
    # per-class deduplication threshold in meters. if None, the consumer
    # (Deduplicator) falls back to its own default. set this when the
    # asset's physical scale differs from the typical-substation default —
    # e.g. solar farms (~1 km), subway transfer stations (~50 m).
    dedup_distance_m : Optional[float]      = None
    confidence       : str                  = "medium"

    def __post_init__(self) -> None:
        if self.sector not in VALID_SECTORS:
            raise ValueError(
                f"AssetClass '{self.name}': unknown sector '{self.sector}'. "
                f"Must be one of {VALID_SECTORS}."
            )
        if self.require_geometry not in VALID_GEOMETRIES:
            raise ValueError(
                f"AssetClass '{self.name}': unknown require_geometry "
                f"'{self.require_geometry}'. Must be one of "
                f"{VALID_GEOMETRIES}."
            )
        if self.confidence not in VALID_CONFIDENCE:
            raise ValueError(
                f"AssetClass '{self.name}': unknown confidence "
                f"'{self.confidence}'. Must be one of {VALID_CONFIDENCE}."
            )
        if (
            self.min_area_m2 is not None
            and self.require_geometry not in GEOMETRIES_WITH_AREA
        ):
            raise ValueError(
                f"AssetClass '{self.name}': min_area_m2={self.min_area_m2} "
                f"is only meaningful when require_geometry is one of "
                f"{GEOMETRIES_WITH_AREA} (got '{self.require_geometry}')."
            )
        if (
            self.dedup_distance_m is not None
            and self.dedup_distance_m <= 0
        ):
            raise ValueError(
                f"AssetClass '{self.name}': dedup_distance_m must be a "
                f"positive number (got {self.dedup_distance_m})."
            )
        if not self.tags and not self.any_of_tags:
            raise ValueError(
                f"AssetClass '{self.name}': must specify at least one of "
                f"`tags` or `any_of_tags`."
            )


# ---------------------------------------------------------------------------
# energy sector
# ---------------------------------------------------------------------------
# order: most-specific substation subtypes first, untyped catch-all last.

_SUBSTATION_EXCLUDES: tuple[TagTuple, ...] = (
    # rules out tagged transformer kiosk buildings, which sometimes carry
    # `power=substation` as a secondary tag but should not be treated as
    # facility-scale substations.
    ("building", "transformer_tower"),
)

ENERGY_CLASSES: tuple[AssetClass, ...] = (
    AssetClass(
        name="energy.transmission.substation",
        sector="energy",
        tags=(
            ("power", "substation"),
            ("substation", "transmission"),
        ),
        exclude_tags=_SUBSTATION_EXCLUDES,
        dedup_distance_m=200.0,
        confidence="high",
    ),
    AssetClass(
        name="energy.distribution.substation",
        sector="energy",
        tags=(
            ("power", "substation"),
            ("substation", "distribution"),
        ),
        exclude_tags=_SUBSTATION_EXCLUDES,
        dedup_distance_m=200.0,
        confidence="high",
    ),
    AssetClass(
        # new in this revision: small low-voltage distribution substations.
        # `substation=minor_distribution` is a real but rare OSM value.
        name="energy.distribution.substation_minor",
        sector="energy",
        tags=(
            ("power", "substation"),
            ("substation", "minor_distribution"),
        ),
        exclude_tags=_SUBSTATION_EXCLUDES,
        dedup_distance_m=200.0,
        confidence="medium",
    ),
    AssetClass(
        # catch-all for substations with no `substation=*` subtype tag.
        # must come after the three subtype-specific entries above so the
        # matcher prefers them when their tag combinations apply.
        name="energy.distribution.substation_untyped",
        sector="energy",
        tags=(("power", "substation"),),
        exclude_tags=_SUBSTATION_EXCLUDES,
        dedup_distance_m=200.0,
        confidence="medium",
    ),
    # solar / wind farms FIRST so they match before the generic power_plant.
    # OSM convention: real utility-scale solar/wind farms are tagged
    # `power=plant + plant:source=solar/wind` (Pattern A). the older
    # `power=generator + generator:source=solar` (Pattern B) is mostly
    # used for individual panel clusters within a facility — those are
    # not facility-scale assets and we deliberately don't match them.
    # histogram audit on AU + CA confirmed every top-10 largest solar
    # polygon uses Pattern A. see archive/outputs_v1_curation/solar_histogram.py.
    AssetClass(
        name="energy.generation.solar_farm",
        sector="energy",
        tags=(
            ("power", "plant"),
            ("plant:source", "solar"),
        ),
        require_geometry="way_or_relation",
        dedup_distance_m=1000.0,
        confidence="high",
    ),
    AssetClass(
        name="energy.generation.wind_farm",
        sector="energy",
        tags=(
            ("power", "plant"),
            ("plant:source", "wind"),
        ),
        require_geometry="way_or_relation",
        dedup_distance_m=1000.0,
        confidence="high",
    ),
    AssetClass(
        # conventional power plants only — `plant:source=solar/wind` is
        # routed to solar_farm / wind_farm above. this entry must come
        # after the renewable-specific entries for the matcher's
        # priority-order rule to do the right thing; the explicit
        # exclude_tags is a belt-and-suspenders guard for that.
        name="energy.generation.power_plant",
        sector="energy",
        tags=(("power", "plant"),),
        exclude_tags=(
            ("plant:source", "solar"),
            ("plant:source", "wind"),
        ),
        # plants are larger than substations; legitimate plants within
        # 200m of each other are rare. 500m collapses duplicate mappings
        # without merging adjacent generating units mapped separately.
        dedup_distance_m=500.0,
        confidence="high",
    ),
)


# ---------------------------------------------------------------------------
# water sector
# ---------------------------------------------------------------------------
# reservoirs are intentionally NOT included in this revision (per spec).
# `landuse=reservoir` will be revisited separately.

WATER_CLASSES: tuple[AssetClass, ...] = (
    AssetClass(
        name="water.wastewater.plant",
        sector="water",
        tags=(("man_made", "wastewater_plant"),),
        # treatment plants are mid-sized facilities. 300m collapses
        # multiple polygon mappings of the same plant.
        dedup_distance_m=300.0,
        confidence="high",
    ),
    AssetClass(
        # OSM `man_made=water_works`. class name uses the short label
        # directly (renamed from `water.treatment.plant` for consistency
        # with the manuscript's "water works" terminology).
        name="water.water_works",
        sector="water",
        tags=(("man_made", "water_works"),),
        dedup_distance_m=300.0,
        confidence="high",
    ),
    AssetClass(
        # new in this revision. `content=water` distinguishes water tanks
        # from oil/gas storage tanks that share `man_made=storage_tank`.
        name="water.storage_tank",
        sector="water",
        tags=(
            ("man_made", "storage_tank"),
            ("content", "water"),
        ),
        # tanks can legitimately cluster very closely (private domestic
        # tanks, multi-tank facilities). 50m collapses obvious duplicates
        # without merging neighboring distinct tanks.
        dedup_distance_m=50.0,
        confidence="medium",
    ),
)


# ---------------------------------------------------------------------------
# transport sector
# ---------------------------------------------------------------------------

TRANSPORT_CLASSES: tuple[AssetClass, ...] = (
    AssetClass(
        # 50-hectare minimum filters out grass strips and small airfields.
        name="transport.airport",
        sector="transport",
        tags=(("aeroway", "aerodrome"),),
        require_geometry="way_or_relation",
        min_area_m2=500_000.0,
        # 1.5 km collapses duplicate polygon mappings of the same airfield
        # (e.g. way + relation for the same airport) without merging two
        # legitimately distinct nearby airports.
        dedup_distance_m=1500.0,
        confidence="high",
    ),
    AssetClass(
        # interpretation (flagged): user spec was "any-of-tags pattern on
        # name/train/public_transport/station". encoded as:
        #   required:  railway=station
        #   any-of:    (name=*) OR (train=yes) OR (public_transport=station)
        # this excludes unnamed bare `railway=station` halts that lack train
        # or public_transport tags. if a different shape is wanted, adjust
        # here.
        name="transport.train_station",
        sector="transport",
        tags=(("railway", "station"),),
        any_of_tags=(
            ("name", None),
            ("train", "yes"),
            ("public_transport", "station"),
        ),
        # subway transfer stations / paired platforms can legitimately
        # be 50–150m apart and should stay separate assets. 50m collapses
        # only obvious duplicate point mappings of the same platform.
        dedup_distance_m=50.0,
        confidence="high",
    ),
    AssetClass(
        # 20-hectare minimum on harbour-tagged area. doc note also mentions
        # "port relations" — supported here via require_geometry covering
        # relations as well as ways. alternative port taggings (e.g.
        # `landuse=harbour`, `industrial=port`) are NOT included in this
        # revision; extend if real-world recall is too low.
        name="transport.port_terminal",
        sector="transport",
        tags=(("harbour", "yes"),),
        require_geometry="way_or_relation",
        min_area_m2=200_000.0,
        # ports are smaller than airports but still substantial. 500m
        # collapses duplicate mappings without merging adjacent terminals.
        dedup_distance_m=500.0,
        confidence="high",
    ),
)


# ---------------------------------------------------------------------------
# telecom sector
# ---------------------------------------------------------------------------
# only `data_center` in this revision. communication towers, exchanges,
# and broadcast sites are documented in ONTOLOGY.md but deferred.

TELECOM_CLASSES: tuple[AssetClass, ...] = (
    AssetClass(
        name="telecom.data_center",
        sector="telecom",
        tags=(("building", "data_center"),),
        dedup_distance_m=200.0,
        confidence="high",
    ),
)


# ---------------------------------------------------------------------------
# registry + lookups
# ---------------------------------------------------------------------------

ASSET_CLASSES: tuple[AssetClass, ...] = (
    *ENERGY_CLASSES,
    *WATER_CLASSES,
    *TRANSPORT_CLASSES,
    *TELECOM_CLASSES,
)

SECTORS: dict[str, tuple[AssetClass, ...]] = {
    "energy":    ENERGY_CLASSES,
    "water":     WATER_CLASSES,
    "transport": TRANSPORT_CLASSES,
    "telecom":   TELECOM_CLASSES,
}

# built once at import time; cheap to keep in memory at ~14 entries.
_BY_NAME: dict[str, AssetClass] = {cls.name: cls for cls in ASSET_CLASSES}


def get_classes_for_sector(sector_name: str) -> tuple[AssetClass, ...]:
    """return every AssetClass in the given sector, in declaration order."""
    if sector_name not in SECTORS:
        raise KeyError(
            f"Unknown sector '{sector_name}'. "
            f"Available: {sorted(SECTORS.keys())}."
        )
    return SECTORS[sector_name]


def get_class_by_name(name: str) -> AssetClass:
    """look up an AssetClass by its canonical dotted name."""
    if name not in _BY_NAME:
        raise KeyError(
            f"Unknown asset class '{name}'. "
            f"Available: {sorted(_BY_NAME.keys())}."
        )
    return _BY_NAME[name]
