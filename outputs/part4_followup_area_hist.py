"""
Follow-up: histogram of computed areas for aeroway=aerodrome and harbour=yes
polygons (closed ways + multipolygon relations) in central-america.

Purpose: understand whether the 50-ha airport / 20-ha port thresholds are
appropriate for central-america's OSM coverage, given that the first
transport extract returned only 1 airport and 0 ports.
"""
from __future__ import annotations
import os
import osmium
from shapely import wkb as shapely_wkb
from pyproj import Geod

PATH = r"D:\central-america-260526.osm_transport_only.osm.pbf"

GEOD = Geod(ellps="WGS84")
WKBF = osmium.geom.WKBFactory()

aerodrome_areas: list = []
harbour_yes_areas: list = []
landuse_harbour_areas: list = []
# Track names for inspection
aerodrome_records: list = []   # [(area_m2, name, osm_id, from_way)]
harbour_records: list = []


class Scanner(osmium.SimpleHandler):
    def area(self, a):
        tags = dict(a.tags)
        aw = tags.get("aeroway")
        hb = tags.get("harbour")
        lu = tags.get("landuse")
        name = tags.get("name", "")
        otype = "way" if a.from_way() else "relation"
        oid = a.orig_id()

        # Only categories we care about for this audit
        if aw != "aerodrome" and hb != "yes" and lu != "harbour":
            return

        try:
            wkb_hex = WKBF.create_multipolygon(a)
            poly = shapely_wkb.loads(bytes.fromhex(wkb_hex))
            area_signed, _ = GEOD.geometry_area_perimeter(poly)
            area_m2 = abs(area_signed)
        except Exception:
            return

        if aw == "aerodrome":
            aerodrome_areas.append(area_m2)
            aerodrome_records.append((area_m2, name, otype, oid, tags.get("iata", "")))
        if hb == "yes":
            harbour_yes_areas.append(area_m2)
            harbour_records.append((area_m2, name, otype, oid, "harbour=yes"))
        if lu == "harbour":
            landuse_harbour_areas.append(area_m2)
            harbour_records.append((area_m2, name, otype, oid, "landuse=harbour"))


def histogram(label: str, values: list, thresholds: list[float]) -> None:
    if not values:
        print(f"{label}: 0 polygons")
        return
    print(f"{label}: {len(values)} polygons")
    values_sorted = sorted(values)
    print(
        f"  min = {values_sorted[0]:>12,.0f} m²  "
        f"({values_sorted[0]/10_000:>8,.1f} ha)"
    )
    print(
        f"  med = {values_sorted[len(values_sorted)//2]:>12,.0f} m²  "
        f"({values_sorted[len(values_sorted)//2]/10_000:>8,.1f} ha)"
    )
    print(
        f"  max = {values_sorted[-1]:>12,.0f} m²  "
        f"({values_sorted[-1]/10_000:>8,.1f} ha)"
    )
    for t in thresholds:
        passes = sum(1 for v in values if v >= t)
        print(
            f"  >= {t:>12,.0f} m² ({t/10_000:>5,.0f} ha): "
            f"{passes:>4d} polygons ({100*passes/len(values):.1f}%)"
        )


if __name__ == "__main__":
    print(f"Scanning {PATH}")
    Scanner().apply_file(PATH, locations=True)
    print()
    histogram(
        "aeroway=aerodrome (ways+relations)",
        aerodrome_areas,
        [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000],
    )
    print()
    histogram(
        "harbour=yes (ways+relations)",
        harbour_yes_areas,
        [10_000, 50_000, 100_000, 200_000, 500_000],
    )
    print()
    histogram(
        "landuse=harbour (ways+relations)",
        landuse_harbour_areas,
        [10_000, 50_000, 100_000, 200_000, 500_000],
    )
    print()
    # Top 10 largest aerodromes
    print("Top 10 largest aerodromes:")
    for rec in sorted(aerodrome_records, key=lambda r: -r[0])[:10]:
        area, name, otype, oid, iata = rec
        nm = name or "(no name)"
        ic = f"  iata={iata}" if iata else ""
        print(f"  {area:>12,.0f} m²  ({area/10_000:>7,.1f} ha)  {otype}/{oid}  {nm}{ic}")
    print()
    print("Top 10 largest harbours (any tag):")
    for rec in sorted(harbour_records, key=lambda r: -r[0])[:10]:
        area, name, otype, oid, tag = rec
        nm = name or "(no name)"
        print(f"  {area:>12,.0f} m²  ({area/10_000:>7,.1f} ha)  {otype}/{oid}  [{tag}]  {nm}")
