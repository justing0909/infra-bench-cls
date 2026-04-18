#!/usr/bin/env python3
"""
solar_collapse.py

Prototype collapsed extraction pass that runs on an existing power-only OSM PBF.

Design:
- Keep explicit solar plant ways as facility assets.
- Drop solar generator ways covered by explicit solar plant polygons.
- Cluster remaining solar generator ways into inferred facilities.
- Keep non-solar assets in the same output.
- Avoid materializing a huge raw solar-expanded asset table.

Recommended first run:
python solar_collapse.py ^
  --pbf data/pbf/maine-latest.osm_power_only.osm.pbf ^
  --output-parquet data/PIPELINE/01-extracted-assets/maine_all_assets.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import osmium
import pandas as pd
from scipy.spatial import KDTree
from shapely.geometry import Point, Polygon
from shapely.prepared import prep


DEFAULT_CLUSTER_RADIUS_M = 250.0
DEFAULT_PLANT_BUFFER_M = 35.0


# ----------------------------
# Tag helpers
# ----------------------------

def _matches(tag_dict: Dict[str, str], rules: List[Tuple[str, Optional[str]]]) -> bool:
    for key, value in rules:
        tag_val = tag_dict.get(key)
        if tag_val is None:
            return False
        if value is not None and tag_val != value:
            return False
    return True


def _is_solar_generator(tags: Dict[str, str]) -> bool:
    # Exclude rooftop/distributed building-scale solar from the core corpus.
    if tags.get("location") == "roof":
        return False
    return _matches(tags, [("power", "generator"), ("generator:source", "solar")])


def _is_solar_plant_way(tags: Dict[str, str]) -> bool:
    candidates = [
        [("power", "plant"), ("plant:source", "solar")],
        [("power", "plant"), ("generator:source", "solar")],
        [("power", "plant"), ("source", "solar")],
    ]
    return any(_matches(tags, c) for c in candidates)


def _is_solar_site_relation(tags: Dict[str, str]) -> bool:
    candidates = [
        [("type", "site"), ("plant:source", "solar")],
        [("type", "site"), ("generator:source", "solar")],
        [("power", "plant"), ("plant:source", "solar")],
        [("power", "plant"), ("generator:source", "solar")],
    ]
    return any(_matches(tags, c) for c in candidates)


def _classify_non_solar(tags: Dict[str, str]) -> Optional[str]:
    # Intentionally kept close to your current medium-confidence setup.
    if _matches(tags, [("power", "substation"), ("substation", "transmission")]):
        return "energy.transmission.substation"
    if _matches(tags, [("power", "substation"), ("substation", "distribution")]):
        return "energy.distribution.substation"
    if _matches(tags, [("power", "substation")]):
        return "energy.distribution.substation_untyped"
    if _matches(tags, [("power", "generator"), ("generator:source", "wind")]):
        return "energy.generation.wind_farm"
    if _matches(tags, [("power", "plant")]):
        # Solar plants are handled separately first.
        if not _is_solar_plant_way(tags):
            return "energy.generation.power_plant"
    if _matches(tags, [("power", "generator")]):
        # Keep generic non-solar generator.
        if tags.get("generator:source") != "solar":
            return "energy.generation.generator"
    return None


# ----------------------------
# Geometry + clustering helpers
# ----------------------------

def _meters_to_degrees_lat(meters: float) -> float:
    return meters / 111_320.0


def _point_to_local_xy_m(lat: float, lon: float, lat_ref: float) -> Tuple[float, float]:
    y = lat * 111_320.0
    x = lon * 111_320.0 * max(math.cos(math.radians(lat_ref)), 0.1)
    return x, y


def _cluster_points(rows: List[dict], radius_m: float) -> List[List[int]]:
    if not rows:
        return []

    coords = np.array([[r["lat"], r["lon"]] for r in rows], dtype=float)
    lat_ref = float(coords[:, 0].mean())

    xy = np.array([
        _point_to_local_xy_m(lat, lon, lat_ref)
        for lat, lon in coords
    ], dtype=float)

    tree = KDTree(xy)

    visited = np.zeros(len(rows), dtype=bool)
    clusters: List[List[int]] = []

    for i in range(len(rows)):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        cluster: List[int] = []

        while stack:
            idx = stack.pop()
            cluster.append(idx)
            neighbors = tree.query_ball_point(xy[idx], r=radius_m)
            for n in neighbors:
                if not visited[n]:
                    visited[n] = True
                    stack.append(n)

        clusters.append(cluster)

    return clusters


def _build_nearest_plant_tree(plants: List["SolarPlantWay"]) -> Tuple[Optional[KDTree], float]:
    if not plants:
        return None, 0.0
    lat_ref = float(np.mean([p.lat for p in plants]))
    xy = np.array([
        _point_to_local_xy_m(p.lat, p.lon, lat_ref)
        for p in plants
    ], dtype=float)
    return KDTree(xy), lat_ref


def _nearest_plant_distance_m(
    lat: float,
    lon: float,
    plant_tree: Optional[KDTree],
    lat_ref: float,
) -> Optional[float]:
    if plant_tree is None:
        return None
    q = np.array(_point_to_local_xy_m(lat, lon, lat_ref), dtype=float)
    dist_m, _ = plant_tree.query(q)
    return float(dist_m)


def _cluster_confidence(member_count: int) -> str:
    if member_count == 1:
        return "low"
    if member_count <= 5:
        return "medium"
    return "high"


# ----------------------------
# Data classes
# ----------------------------

@dataclass
class SolarPlantWay:
    osm_id: int
    lat: float
    lon: float
    polygon: Optional[Polygon]
    tags: Dict[str, str]


@dataclass
class SolarGeneratorWay:
    osm_id: int
    lat: float
    lon: float
    tags: Dict[str, str]


# ----------------------------
# Handler
# ----------------------------

class SolarCollapseHandler(osmium.SimpleHandler):
    """
    Streaming-ish extractor:
    - Non-solar assets are collected directly in final-row form.
    - Solar-specific structures are collected for later collapse.
    """

    def __init__(self, log_every: int = 500_000):
        super().__init__()
        self.log_every = log_every

        self.non_solar_rows: List[dict] = []
        self.solar_plant_ways: List[SolarPlantWay] = []
        self.solar_generator_ways: List[SolarGeneratorWay] = []
        self.solar_rooftop_rows: List[dict] = []

        self.solar_relation_ids: Set[int] = set()
        self.solar_relation_way_ids: Set[int] = set()

        self._n_nodes = 0
        self._n_ways = 0
        self._n_relations = 0

    def _maybe_log(self) -> None:
        total = self._n_nodes + self._n_ways + self._n_relations
        if total > 0 and total % self.log_every == 0:
            print(
                f"scanned={total:,} "
                f"non_solar={len(self.non_solar_rows):,} "
                f"solar_plants={len(self.solar_plant_ways):,} "
                f"solar_generators={len(self.solar_generator_ways):,} "
                f"solar_relations={len(self.solar_relation_ids):,}"
            )

    def node(self, n) -> None:
        self._n_nodes += 1
        self._maybe_log()

        if not n.location.valid():
            return

        tags = {k: v for k, v in n.tags}
        asset_type = _classify_non_solar(tags)
        if asset_type is None:
            return

        self.non_solar_rows.append({
            "asset_id": f"osm_node_{n.id}",
            "asset_type": asset_type,
            "lat": float(n.location.lat),
            "lon": float(n.location.lon),
            "name": tags.get("name", ""),
            "source": "osm_geofabrik",
            "osm_tags": tags,
            "solar_facility_type": "",
            "inferred_confidence": "",
            "n_merged_parts": 1,
        })

    def way(self, w) -> None:
        self._n_ways += 1
        self._maybe_log()

        tags = {k: v for k, v in w.tags}
        valid_nodes = [node for node in w.nodes if node.location.valid()]
        if not valid_nodes:
            return

        mid = valid_nodes[len(valid_nodes) // 2]
        lat = float(mid.location.lat)
        lon = float(mid.location.lon)

        if _is_solar_plant_way(tags):
            polygon = None
            try:
                coords = [(float(node.location.lon), float(node.location.lat)) for node in valid_nodes]
                if len(coords) >= 4 and coords[0] == coords[-1]:
                    polygon = Polygon(coords)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
            except Exception:
                polygon = None

            self.solar_plant_ways.append(SolarPlantWay(
                osm_id=int(w.id),
                lat=lat,
                lon=lon,
                polygon=polygon,
                tags=tags,
            ))
            return

        if tags.get("power") == "generator" and tags.get("generator:source") == "solar":
            if tags.get("location") == "roof":
                self.solar_rooftop_rows.append({
                    "asset_id": f"osm_way_{w.id}",
                    "asset_type": "energy.generation.solar_rooftop_distributed",
                    "lat": lat,
                    "lon": lon,
                    "name": tags.get("name", ""),
                    "source": "osm_geofabrik",
                    "osm_tags": tags,
                    "solar_facility_type": "rooftop",
                    "inferred_confidence": "",
                    "n_merged_parts": 1,
                })
                return

            self.solar_generator_ways.append(SolarGeneratorWay(
                osm_id=int(w.id),
                lat=lat,
                lon=lon,
                tags=tags,
            ))
            return

        asset_type = _classify_non_solar(tags)
        if asset_type is None:
            return

        self.non_solar_rows.append({
            "asset_id": f"osm_way_{w.id}",
            "asset_type": asset_type,
            "lat": lat,
            "lon": lon,
            "name": tags.get("name", ""),
            "source": "osm_geofabrik",
            "osm_tags": tags,
            "solar_facility_type": "",
            "inferred_confidence": "",
            "n_merged_parts": 1,
        })

    def relation(self, r) -> None:
        self._n_relations += 1
        self._maybe_log()

        tags = {k: v for k, v in r.tags}
        if not _is_solar_site_relation(tags):
            return

        self.solar_relation_ids.add(int(r.id))
        for member in r.members:
            if member.type == "w":
                self.solar_relation_way_ids.add(int(member.ref))


# ----------------------------
# Main collapse routine
# ----------------------------

def collapse_solar(
    pbf_path: str,
    output_parquet: str,
    output_csv: Optional[str] = None,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    plant_buffer_m: float = DEFAULT_PLANT_BUFFER_M,
) -> pd.DataFrame:
    if not os.path.exists(pbf_path):
        raise FileNotFoundError(f"PBF not found: {pbf_path}")

    print(f"Reading power-only PBF: {pbf_path}")
    handler = SolarCollapseHandler()

    # pyosmium compatibility
    try:
        handler.apply_file(
            pbf_path,
            locations=True,
            idx="sparse_file_array,solar_collapse_locations.idx",
        )
    except TypeError:
        handler.apply_file(pbf_path, locations=True)

    print("\nInitial counts:")
    print(f"  Non-solar assets:       {len(handler.non_solar_rows):,}")
    print(f"  Solar plant ways:       {len(handler.solar_plant_ways):,}")
    print(f"  Solar generator ways:   {len(handler.solar_generator_ways):,}")
    print(f"  Solar relation IDs:     {len(handler.solar_relation_ids):,}")
    print(f"  Solar relation way IDs: {len(handler.solar_relation_way_ids):,}")
    print(f"  Solar rooftop ways:     {len(handler.solar_rooftop_rows):,}")

    if handler.solar_relation_way_ids:
        print(f"  Sample solar relation way IDs: {list(sorted(handler.solar_relation_way_ids))[:10]}")
    if handler.solar_generator_ways:
        print(f"  Sample solar generator way IDs: {[g.osm_id for g in handler.solar_generator_ways[:10]]}")

    # Prepare plant polygons
    prepared_plants: List[Tuple[SolarPlantWay, object]] = []
    for plant in handler.solar_plant_ways:
        if plant.polygon is None or plant.polygon.is_empty:
            continue
        try:
            buffer_deg = _meters_to_degrees_lat(plant_buffer_m)
            buffered = plant.polygon.buffer(buffer_deg)
            if not buffered.is_valid:
                buffered = buffered.buffer(0)
            prepared_plants.append((plant, prep(buffered)))
        except Exception:
            prepared_plants.append((plant, prep(plant.polygon)))

    plant_tree, plant_lat_ref = _build_nearest_plant_tree(handler.solar_plant_ways)

    kept_solar_rows: List[dict] = []

    # Keep explicit plant ways
    for plant in handler.solar_plant_ways:
        kept_solar_rows.append({
            "asset_id": f"osm_way_{plant.osm_id}",
            "asset_type": "energy.generation.solar_farm",
            "lat": plant.lat,
            "lon": plant.lon,
            "name": plant.tags.get("name", ""),
            "source": "osm_geofabrik",
            "osm_tags": plant.tags,
            "solar_facility_type": "plant_way",
            "inferred_confidence": "",
            "n_merged_parts": 1,
        })

    leftovers: List[dict] = []
    dropped_by_relation = 0
    dropped_by_plant = 0

    for gen in handler.solar_generator_ways:
        if gen.osm_id in handler.solar_relation_way_ids:
            dropped_by_relation += 1
            continue

        point = Point(gen.lon, gen.lat)
        covered = False
        for _plant, prepared in prepared_plants:
            try:
                if prepared.contains(point):
                    covered = True
                    break
            except Exception:
                continue

        if covered:
            dropped_by_plant += 1
            continue

        nearest_plant_m = _nearest_plant_distance_m(
            gen.lat,
            gen.lon,
            plant_tree,
            plant_lat_ref,
        )

        leftovers.append({
            "asset_id": f"osm_way_{gen.osm_id}",
            "lat": gen.lat,
            "lon": gen.lon,
            "name": gen.tags.get("name", ""),
            "source": "osm_geofabrik",
            "osm_tags": gen.tags,
            "nearest_plant_m": nearest_plant_m,
        })

    print("\nCoverage suppression:")
    print(f"  Dropped by solar relation membership: {dropped_by_relation:,}")
    print(f"  Dropped by plant polygon containment: {dropped_by_plant:,}")
    print(f"  Leftover standalone generators:       {len(leftovers):,}")

    if leftovers:
        nearest_vals = [r["nearest_plant_m"] for r in leftovers if r["nearest_plant_m"] is not None]
        if nearest_vals:
            s = pd.Series(nearest_vals)
            print("\nLeftover generator nearest-plant distance stats (m):")
            print(f"  min:   {s.min():.1f}")
            print(f"  p25:   {s.quantile(0.25):.1f}")
            print(f"  p50:   {s.quantile(0.50):.1f}")
            print(f"  p75:   {s.quantile(0.75):.1f}")
            print(f"  p90:   {s.quantile(0.90):.1f}")
            print(f"  max:   {s.max():.1f}")
            print("  leftover generators near plant centroids:")
            print(f"    <= 50 m:  {sum(v <= 50 for v in nearest_vals)}")
            print(f"    <= 100 m: {sum(v <= 100 for v in nearest_vals)}")
            print(f"    <= 250 m: {sum(v <= 250 for v in nearest_vals)}")

    clusters = _cluster_points(leftovers, radius_m=cluster_radius_m)
    print(f"  Inferred standalone solar clusters:   {len(clusters):,}")

    cluster_sizes = [len(c) for c in clusters]
    if cluster_sizes:
        s = pd.Series(cluster_sizes)
        print("\nGenerator-cluster size stats:")
        print(f"  min:   {s.min()}")
        print(f"  p25:   {s.quantile(0.25):.1f}")
        print(f"  p50:   {s.quantile(0.50):.1f}")
        print(f"  p75:   {s.quantile(0.75):.1f}")
        print(f"  p90:   {s.quantile(0.90):.1f}")
        print(f"  max:   {s.max()}")
        print("  counts by size bucket:")
        print(f"    size = 1:   {(s == 1).sum()}")
        print(f"    size = 2-5: {((s >= 2) & (s <= 5)).sum()}")
        print(f"    size = 6-20:{((s >= 6) & (s <= 20)).sum()}")
        print(f"    size > 20:  {(s > 20).sum()}")

    for cluster_id, indices in enumerate(clusters, start=1):
        group = [leftovers[i] for i in indices]
        member_count = len(group)
        inferred_confidence = _cluster_confidence(member_count)

        kept_solar_rows.append({
            "asset_id": f"inferred_solar_cluster_{cluster_id}",
            "asset_type": "energy.generation.solar_facility_inferred",
            "lat": float(np.mean([g["lat"] for g in group])),
            "lon": float(np.mean([g["lon"] for g in group])),
            "name": "",
            "source": "osm_geofabrik_inferred",
            "osm_tags": {
                "inferred_from": "solar_generator_cluster",
                "member_count": member_count,
                "member_ids": [g["asset_id"] for g in group],
            },
            "solar_facility_type": "generator_cluster",
            "inferred_confidence": inferred_confidence,
            "n_merged_parts": member_count,
        })

    print(f"\nExcluded distributed-edge rooftop solar ways: {len(handler.solar_rooftop_rows):,}")
    final_rows = handler.non_solar_rows + kept_solar_rows
    df = pd.DataFrame(final_rows)

    if df.empty:
        print("No assets written.")
        return df

    os.makedirs(os.path.dirname(output_parquet) or ".", exist_ok=True)

    # Make both CSV and Parquet portable by serializing osm_tags
    out_df = df.copy()
    out_df["osm_tags"] = out_df["osm_tags"].apply(json.dumps)

    out_df.to_parquet(output_parquet, index=False)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        out_df.to_csv(output_csv, index=False)

    print("\nFinal output:")
    print(f"  Wrote Parquet: {output_parquet}")
    if output_csv:
        print(f"  Wrote CSV:     {output_csv}")
    print(f"  Total assets: {len(df):,}")

    print("\nAsset counts:")
    print(df["asset_type"].value_counts().to_string())

    solar_df = df[df["asset_type"].isin([
        "energy.generation.solar_farm",
        "energy.generation.solar_facility_inferred",
    ])].copy()

    if not solar_df.empty:
        print("\nSolar facility breakdown:")
        if "solar_facility_type" in solar_df.columns:
            print(solar_df["solar_facility_type"].value_counts().to_string())

        inferred_df = solar_df[solar_df["asset_type"] == "energy.generation.solar_facility_inferred"]
        if not inferred_df.empty and "inferred_confidence" in inferred_df.columns:
            print("\nInferred facility confidence breakdown:")
            print(inferred_df["inferred_confidence"].value_counts().to_string())

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Collapse solar hierarchy from a power-only OSM PBF")
    parser.add_argument("--pbf", required=True, help="Path to existing *_power_only.osm.pbf")
    parser.add_argument(
        "--output-parquet",
        required=True,
        help="Path to output collapsed Parquet",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV export path for debugging/inspection",
    )
    parser.add_argument(
        "--cluster-radius-m",
        type=float,
        default=DEFAULT_CLUSTER_RADIUS_M,
        help="Clustering radius for leftover standalone solar generators",
    )
    parser.add_argument(
        "--plant-buffer-m",
        type=float,
        default=DEFAULT_PLANT_BUFFER_M,
        help="Outward buffer applied to plant polygons before containment check",
    )
    args = parser.parse_args()

    collapse_solar(
        pbf_path=args.pbf,
        output_parquet=args.output_parquet,
        output_csv=args.output_csv,
        cluster_radius_m=args.cluster_radius_m,
        plant_buffer_m=args.plant_buffer_m,
    )


if __name__ == "__main__":
    main()