"""
Scan available power_only PBFs for solar features and produce a size +
tag-pattern histogram so we can think about a more nuanced filter.

OSM solar tagging conventions (per ONTOLOGY.md notes):
  (A) facility-level:  power=plant      + plant:source=solar         (canonical)
  (B) generator-level: power=generator  + generator:source=solar     (current match)
       — can be a real facility OR an individual panel cluster
  (C) rooftop:         (B) + location=roof / rooftop                 (excluded)
"""
from __future__ import annotations
import os
import osmium
import collections
from shapely import wkb as shapely_wkb
from pyproj import Geod

PATHS = [
    ("central-america",
     r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm\data\pbf\power_only"
     r"\central-america-latest.osm_power_only.osm.pbf"),
    ("australia-oceania",
     r"D:\australia-oceania-260526.osm_power_only.osm.pbf"),
]

GEOD = Geod(ellps="WGS84")
WKBF = osmium.geom.WKBFactory()


# Histogram buckets in m². Show count per bucket per tag pattern.
BUCKETS = [
    (0,          100,        "  0 – 100 m²        (panel-scale)"),
    (100,        1_000,      "  100 – 1k m²"),
    (1_000,      10_000,     "  1k – 10k m²       (current 10k cutoff)"),
    (10_000,     100_000,    "  10k – 100k m²     (1–10 ha; small facility)"),
    (100_000,    1_000_000,  "  100k – 1M m²      (10–100 ha; typical utility-scale)"),
    (1_000_000,  10_000_000, "  1M – 10M m²       (>100 ha; large utility)"),
    (10_000_000, float("inf"),
                             "  >10M m²           (>1000 ha; mega-farm)"),
]


def bucket_index(area_m2: float) -> int:
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if lo <= area_m2 < hi:
            return i
    return len(BUCKETS) - 1


class SolarScanner(osmium.SimpleHandler):
    """
    Collects every solar-tagged way/relation/node and tracks:
      - area (for way/relation; via shapely + pyproj.Geod)
      - tag pattern bucket
    """
    def __init__(self):
        super().__init__()
        # Counter keyed by (tag_pattern, geom_type, area_bucket_index).
        self.counter: collections.Counter = collections.Counter()
        # Also gather a sample of the largest few for manual inspection.
        self.largest: list = []   # list[(area_m2, name, osm_type, osm_id, tags)]
        self.node_count_by_pattern: collections.Counter = collections.Counter()

    @staticmethod
    def _tag_pattern(tags) -> str | None:
        # Filter to solar-related features only.
        power = tags.get("power")
        plant_src = tags.get("plant:source")
        gen_src   = tags.get("generator:source")
        loc       = tags.get("location", "")

        rooftop = loc in ("roof", "rooftop")

        if power == "plant" and plant_src == "solar":
            return "A_plant_solar" + ("_rooftop" if rooftop else "")
        if power == "generator" and gen_src == "solar":
            return "B_gen_solar" + ("_rooftop" if rooftop else "")
        return None

    def node(self, n):
        pat = self._tag_pattern(n.tags)
        if pat is not None:
            self.node_count_by_pattern[pat] += 1

    def area(self, a):
        pat = self._tag_pattern(a.tags)
        if pat is None:
            return
        try:
            wkb_hex = WKBF.create_multipolygon(a)
            poly = shapely_wkb.loads(bytes.fromhex(wkb_hex))
            area_m2 = abs(GEOD.geometry_area_perimeter(poly)[0])
        except Exception:
            return
        geom_type = "way" if a.from_way() else "relation"
        bi = bucket_index(area_m2)
        self.counter[(pat, geom_type, bi)] += 1

        name = a.tags.get("name", "")
        tags_dict = dict(a.tags)
        self.largest.append((area_m2, name, geom_type, a.orig_id(), tags_dict))


def fmt_int(x: int) -> str:
    return f"{x:>6,}"


def print_report(region: str, sc: SolarScanner) -> None:
    print(f"\n========== {region} ==========")

    # Node-tagged features by pattern
    print("\nNode-tagged solar features (rejected by way_or_relation rule):")
    if not sc.node_count_by_pattern:
        print("  (none)")
    else:
        for pat, n in sorted(sc.node_count_by_pattern.items()):
            print(f"  {pat:35s} {fmt_int(n)}")

    # Per-pattern, per-geom histogram
    patterns_seen = sorted({key[0] for key in sc.counter})
    geoms_seen    = sorted({key[1] for key in sc.counter})

    if not patterns_seen:
        print("\nNo solar polygons matched.")
        return

    for pat in patterns_seen:
        print(f"\nPattern '{pat}' — area histogram (polygons only):")
        # Compute totals per bucket per geom
        rows = []
        totals = collections.Counter()
        for bi, (lo, hi, label) in enumerate(BUCKETS):
            per_geom = {g: sc.counter.get((pat, g, bi), 0) for g in geoms_seen}
            total    = sum(per_geom.values())
            rows.append((label, per_geom, total))
            totals[bi] = total
        for (label, per_geom, total) in rows:
            geom_str = "  ".join(
                f"{g}={fmt_int(per_geom[g])}" for g in geoms_seen
            )
            print(f"  {label:55s} {geom_str:25s}  total={fmt_int(total)}")
        grand = sum(totals.values())
        print(f"  {'TOTAL':55s} {' ' * 25}    {fmt_int(grand)}")

    # Show the 10 largest features overall for manual sanity check
    print("\nTop 10 largest solar polygons (any pattern):")
    for area_m2, name, geom_type, oid, tags in sorted(sc.largest, key=lambda r: -r[0])[:10]:
        ha = area_m2 / 10_000
        nm = name or "(no name)"
        # Compress tag dict to one line, dropping noise
        keys_to_keep = ("power", "plant:source", "generator:source",
                        "location", "plant:method", "generator:method",
                        "name", "operator", "ref")
        compact = {k: tags[k] for k in keys_to_keep if k in tags}
        print(f"  {ha:>10,.1f} ha  {geom_type}/{oid}  {nm}")
        print(f"             tags: {compact}")


def main():
    for region, path in PATHS:
        if not os.path.exists(path):
            print(f"=== {region}: MISSING {path}")
            continue
        sc = SolarScanner()
        sc.apply_file(path, locations=True)
        print_report(region, sc)


if __name__ == "__main__":
    main()
