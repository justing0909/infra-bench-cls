"""
asset_footprint_share.py
------------------------
How much of each 600 m x 600 m tile is the labelled asset itself, and how much
is surrounding land? Answers the reviewer concern that classifications may
lean on context rather than the facility.

For every v1 asset this fetches its mapped OSM geometry from the Overpass API,
clips it to the tile footprint (60 px x 10 m, centred on the asset point), and
records the clipped area as a share of the tile. Assets mapped as a single
node have no footprint and are counted separately rather than given an area.

Overpass serves current OSM, not the 2026-05-26 snapshot the dataset was
built from, so a few geometries may have been edited or deleted since; those
are reported as missing.

    python -m evaluation.analysis.asset_footprint_share

Writes data/asset_footprints/asset_footprint_share_v1.csv (one row per asset)
and prints the per-class summary. Raw Overpass responses are cached under
data/asset_footprints/overpass/ so a rerun makes no requests.
"""

import json
import math
import time
import urllib.parse
import urllib.request

import pandas as pd
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, unary_union

from curation.paths import DATA_DIR, SPLIT_ARTIFACT

OVERPASS = "https://overpass-api.de/api/interpreter"
OUT_DIR = DATA_DIR / "asset_footprints"
CACHE_DIR = OUT_DIR / "overpass"
RESULT = OUT_DIR / "asset_footprint_share_v1.csv"
TILE_M = 600
BATCH = 1000


def _fetch(kind: str, ids: list, batch_no: int) -> list:
    path = CACHE_DIR / f"{kind}_{batch_no:03d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["elements"]
    query = f"[out:json][timeout:180];{kind}(id:{','.join(map(str, ids))});out geom;"
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS, data=data,
                                 headers={"User-Agent": "infra-bench-cls footprint analysis"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=240) as r:
                body = json.load(r)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(20 * (attempt + 1))  # Overpass asks clients to back off
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding="utf-8")
    time.sleep(2)
    return body["elements"]


def _local(lat0: float, lon0: float):
    """lat/lon -> metres on an equirectangular plane centred on the asset."""
    kx = 111320 * math.cos(math.radians(lat0))
    return lambda lat, lon: ((lon - lon0) * kx, (lat - lat0) * 111320)


def _geometry(el: dict, to_m):
    if el["type"] == "way":
        pts = [to_m(p["lat"], p["lon"]) for p in el.get("geometry", [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            return Polygon(pts).buffer(0)
        return None  # an open way, e.g. a station mapped as a line, has no area
    # multipolygon relation: polygonize the outer members, subtract the inner
    rings = {"outer": [], "inner": []}
    for m in el.get("members", []):
        if m.get("type") == "way" and m.get("geometry"):
            rings.setdefault(m.get("role") or "outer", []).append(
                LineString([to_m(p["lat"], p["lon"]) for p in m["geometry"]]))
    outer = unary_union(list(polygonize(unary_union(rings["outer"])))) if rings["outer"] else None
    if outer is None or outer.is_empty:
        return None
    inner = unary_union(list(polygonize(unary_union(rings["inner"])))) if rings["inner"] else None
    return outer.difference(inner) if inner is not None else outer


def build() -> pd.DataFrame:
    split = pd.read_parquet(SPLIT_ARTIFACT)
    parts = split.asset_id.str.extract(r"osm_(node|way|relation)_(\d+)")
    split["osm_type"], split["osm_id"] = parts[0], parts[1].astype(int)

    elements = {}
    for kind in ("way", "relation"):
        ids = sorted(split.loc[split.osm_type == kind, "osm_id"])
        for b, start in enumerate(range(0, len(ids), BATCH)):
            for el in _fetch(kind, ids[start:start + BATCH], b):
                elements[(kind, el["id"])] = el
            print(f"  {kind}s: {min(start + BATCH, len(ids)):,} / {len(ids):,}")

    half = TILE_M / 2
    tile = box(-half, -half, half, half)
    rows = []
    for r in split.itertuples():
        row = {"asset_id": r.asset_id, "asset_type": r.asset_type, "region": r.region,
               "osm_type": r.osm_type, "status": "", "area_m2": None, "extent_m": None,
               "tile_share": None, "exceeds_tile": None}
        if r.osm_type == "node":
            row["status"] = "point"
        elif (r.osm_type, r.osm_id) not in elements:
            row["status"] = "missing_now"
        else:
            geom = _geometry(elements[(r.osm_type, r.osm_id)], _local(r.lat, r.lon))
            if geom is None or geom.is_empty or geom.area == 0:
                row["status"] = "no_area"
            else:
                inside = geom.intersection(tile).area
                # long side of the minimum rotated rectangle: the asset's extent
                rect = geom.minimum_rotated_rectangle
                xs, ys = rect.exterior.coords.xy
                sides = [math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(2)]
                row.update(status="polygon", area_m2=round(geom.area, 1),
                           extent_m=round(max(sides), 1),
                           tile_share=round(inside / tile.area, 5),
                           exceeds_tile=bool(geom.area - inside > 1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def per_class(g):
        poly = g[g.status == "polygon"]
        s = poly.tile_share * 100
        return pd.Series({
            "tiles": len(g),
            "point_mapped_%": round(100 * (g.status == "point").mean(), 1),
            "with_area": len(poly),
            "median_share_%": round(s.median(), 2) if len(s) else None,
            "p25_%": round(s.quantile(.25), 2) if len(s) else None,
            "p75_%": round(s.quantile(.75), 2) if len(s) else None,
            "under_1%_of_tile": round(100 * (s < 1).mean(), 1) if len(s) else None,
            "under_5%_of_tile": round(100 * (s < 5).mean(), 1) if len(s) else None,
            "exceeds_tile_%": round(100 * poly.exceeds_tile.mean(), 1) if len(s) else None,
        })
    return (df.groupby("asset_type").apply(per_class, include_groups=False)
              .sort_values("median_share_%"))


# Proposed tile size, as a note for future versions rather than a rule: wide
# enough that the 75th-percentile asset fits with a margin of its own extent on
# every side (so a typical asset spans the middle half of the tile, about a
# quarter of its area), rounded up to whole 10 m pixels, never below
# MIN_TILE_M, and never above the current 600 m.
PIXEL_M = 10
MIN_TILE_M = 160   # 16 px: one ViT/16 patch at native resolution
CONTEXT_FACTOR = 2


def proposed_tiles(df: pd.DataFrame) -> pd.DataFrame:
    def per_class(g):
        e = g.loc[g.status == "polygon", "extent_m"].astype(float)
        if len(e) < 10:
            return pd.Series({"assets_with_extent": len(e), "extent_p50_m": None, "extent_p75_m": None,
                              "extent_p90_m": None, "proposed_tile_m": None, "proposed_tile_px": None})
        side = CONTEXT_FACTOR * e.quantile(.75)
        side = min(max(math.ceil(side / PIXEL_M) * PIXEL_M, MIN_TILE_M), TILE_M)
        return pd.Series({"assets_with_extent": len(e),
                          "extent_p50_m": round(e.median()), "extent_p75_m": round(e.quantile(.75)),
                          "extent_p90_m": round(e.quantile(.9)),
                          "proposed_tile_m": side, "proposed_tile_px": side // PIXEL_M})
    return df.groupby("asset_type").apply(per_class, include_groups=False).sort_values("extent_p75_m")


def main():
    df = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT, index=False)
    print(f"\nwrote {len(df):,} assets to {RESULT}")
    print(df.status.value_counts().to_string())
    print("\nasset share of its 600 m tile, by class (polygons only):")
    summary = summarise(df)
    print(summary.to_string())
    proposal = proposed_tiles(df)
    print(f"\nproposed tile size (a note, not a rule): {CONTEXT_FACTOR}x the p75 extent, "
          f"floor {MIN_TILE_M} m, cap {TILE_M} m")
    print(proposal.to_string())
    summary.join(proposal).to_csv(OUT_DIR / "asset_footprint_summary_v1.csv")


if __name__ == "__main__":
    main()
