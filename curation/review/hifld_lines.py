"""
hifld_lines.py
--------------
Secondary evidence for US substation tiles from the HIFLD electric power
transmission lines layer (94,619 lines, data last edited 2025-08-26):

    https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/US_Electric_Power_Transmission_Lines/FeatureServer/0

For every substation tile it asks the layer for the lines inside the tile's
~600 m footprint and records which of them end at the asset (an endpoint
within ENDS_AT_M of the OSM point) and at what voltage. That settles whether a
substation is transmission-connected, and at which voltage. It cannot always
separate a transmission substation from a transmission-to-distribution one,
because HIFLD does not carry distribution feeders, so the evidence is shown to
the reviewer rather than turned into a label.

About a quarter of HIFLD lines were themselves traced from OpenStreetMap
(SOURCE contains "OpenStreetMap"). Those are shown but never count towards
`corroborated`, so a tile is never confirmed by OSM agreeing with itself.

    python -m curation.review.hifld_lines          # then rebuild the queue

Responses are cached per tile under data/label_review/sources/hifld_lines/,
so a rerun only asks for tiles it has not seen.
"""

import json
import math
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..paths import REVIEW_DIR, REVIEW_QUEUE

LAYER = ("https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
         "US_Electric_Power_Transmission_Lines/FeatureServer/0")
CACHE_DIR = REVIEW_DIR / "sources" / "hifld_lines"
EVIDENCE = REVIEW_DIR / "sources" / "hifld_lines.csv"

SUBSTATION_CLASSES = ("energy.transmission.substation", "energy.distribution.substation",
                      "energy.distribution.other")
# HIFLD lines cover the US; a loose box keeps Canada and Mexico out of the requests.
US_BOX = (24.0, 50.0, -125.0, -66.0)
HALF_TILE_M = 300      # 60 px x 10 m tiles
ENDS_AT_M = 250        # an endpoint this close to the OSM point terminates there
TRANSMISSION_KV = 100
FIELDS = "ID,VOLTAGE,VOLT_CLASS,STATUS,SOURCE,SUB_1,SUB_2"


def _metres(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371008.8 * math.asin(math.sqrt(a))


def _query(asset_id: str, lat: float, lon: float) -> dict:
    path = CACHE_DIR / f"{asset_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    dlat = HALF_TILE_M / 111320
    dlon = HALF_TILE_M / (111320 * math.cos(math.radians(lat)))
    params = {
        "geometry": f"{lon - dlon},{lat - dlat},{lon + dlon},{lat + dlat}",
        "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326,
        "spatialRel": "esriSpatialRelIntersects", "outFields": FIELDS,
        "returnGeometry": "true", "f": "json",
    }
    url = f"{LAYER}/query?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.load(r)
            if "error" in data:
                raise RuntimeError(data["error"])
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


_PLACEHOLDER = re.compile(r"^(UNKNOWN|TAP|DEADEND|NOT AVAILABLE)", re.I)
# words that name the kind of facility rather than which one it is, so two
# names only "match" when they share something specific like a place name.
_FILLER = re.compile(r"\b(substation|sub|station|switching|switchyard|power|electric|kv|rail|railway|"
                     r"train|amtrak|airport|aerodrome|airfield|international|intl|regional|rgnl|"
                     r"municipal|muni|county|\d+)\b", re.I)


def _tokens(name: str) -> set:
    return {t for t in re.findall(r"[a-z]+", _FILLER.sub(" ", name.lower())) if len(t) > 2}


def _summarise(lat: float, lon: float, data: dict, osm_name: str = "") -> dict:
    ends, passes, names = [], [], []
    for f in data.get("features", []):
        a = f["attributes"]
        kv = a.get("VOLTAGE")
        kv = None if kv is None or kv < 0 else float(kv)
        from_osm = "openstreetmap" in (a.get("SOURCE") or "").lower()
        endpoints = [pt for path in f.get("geometry", {}).get("paths", []) for pt in (path[0], path[-1])]
        near = min((_metres(lat, lon, y, x) for x, y in endpoints), default=float("inf"))
        (ends if near <= ENDS_AT_M else passes).append((kv, from_osm))
        if near <= ENDS_AT_M:
            # SUB_1/SUB_2 name both ends; which one is this end is not
            # recorded reliably, so keep both real names and let the match decide.
            names += [s for s in (a.get("SUB_1"), a.get("SUB_2"))
                      if s and not _PLACEHOLDER.match(s) and s not in names]

    def describe(lines):
        kvs = sorted({int(k) for k, _ in lines if k is not None}, reverse=True)
        unknown = sum(1 for k, _ in lines if k is None)
        parts = [f"{'/'.join(map(str, kvs))} kV" if kvs else ""]
        if unknown:
            parts.append(f"{unknown} unknown voltage")
        osm = sum(1 for _, o in lines if o)
        if osm:
            parts.append(f"{osm} traced from OSM")
        return ", ".join(p for p in parts if p)

    independent_kv = [k for k, o in ends if k is not None and not o]
    osm_tokens = _tokens(osm_name)
    matched = [n for n in names if osm_tokens and osm_tokens & _tokens(n)]
    n_osm = sum(1 for _, o in ends + passes if o)

    lines = []
    if ends:
        lines.append(f"HIFLD: {len(ends)} line(s) end here ({describe(ends)})")
    elif passes:
        lines.append(f"HIFLD: no line ends here; {len(passes)} pass within 300 m ({describe(passes)})")
    else:
        lines.append("HIFLD: no transmission line within 300 m")
    if names:
        lines.append("HIFLD line ends named: " + ", ".join(names[:4]) +
                     (f" (matches OSM name)" if matched else ""))
    if n_osm:
        lines.append(f"Caution: {n_osm} of these HIFLD lines were traced from OSM, so they don't count as independent")
    return {
        "hifld_lines_end": len(ends),
        "hifld_max_kv": max(independent_kv, default=None),
        "hifld_names": "; ".join(names),
        "hifld_name_match": bool(matched),
        "hifld_osm_lines": n_osm,
        "hifld_text": " | ".join(lines),
        # transmission-connected on lines HIFLD did not take from OSM
        "hifld_transmission": bool(independent_kv) and max(independent_kv) >= TRANSMISSION_KV,
    }


def build(workers: int = 4) -> pd.DataFrame:
    q = pd.read_csv(REVIEW_QUEUE, keep_default_na=False).drop_duplicates("asset_id")
    lo_lat, hi_lat, lo_lon, hi_lon = US_BOX
    tiles = q[q.asset_type.isin(SUBSTATION_CLASSES) & q.lat.between(lo_lat, hi_lat)
              & q.lon.between(lo_lon, hi_lon)]
    rows = tiles[["asset_id", "asset_type", "lat", "lon", "name"]].to_dict("records")

    def one(r):
        return {"asset_id": r["asset_id"], "asset_type": r["asset_type"],
                **_summarise(r["lat"], r["lon"], _query(r["asset_id"], r["lat"], r["lon"]), r["name"])}

    with ThreadPoolExecutor(workers) as pool:
        out = pd.DataFrame(list(pool.map(one, rows)))
    # what counts as the source agreeing with the OSM label
    out["corroborated"] = ((out.asset_type == "energy.transmission.substation")
                           & out.hifld_transmission)
    # strict enough to pass without review: confirmed and the line-end name matches
    out["auto_ok"] = out.corroborated & out.hifld_name_match
    return out


def main():
    ev = build()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(EVIDENCE, index=False)
    print(f"wrote {len(ev):,} tiles to {EVIDENCE}")
    kind = pd.Series("no line within 300 m", index=ev.index)
    kind[ev.hifld_text.str.startswith("HIFLD: no line ends")] = "line passes, none ends"
    kind[(ev.hifld_lines_end > 0) & (ev.hifld_max_kv.fillna(0) < TRANSMISSION_KV)] = "ends here, <100 kV or unknown"
    kind[ev.hifld_transmission] = "ends here, >=100 kV"
    print(pd.crosstab(ev.asset_type, kind, margins=True).to_string())
    print("name matches:", int(ev.hifld_name_match.sum()),
          "| tiles with OSM-traced lines:", int((ev.hifld_osm_lines > 0).sum()),
          "| corroborated:", int(ev.corroborated.sum()))


if __name__ == "__main__":
    main()
