"""
eia_plants.py
-------------
Secondary evidence for US energy tiles from the EIA power plant inventory as
published through HIFLD (13,446 plants, data last edited 2025-07-01):

    https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/Power_Plants_in_the_US/FeatureServer/0

Plant coordinates and primary energy source are reported by operators to EIA,
so they are independent of OSM. For every US energy tile this records the
nearest plants within SEARCH_M, their source, capacity and name.

A tile is corroborated when its OSM class agrees with a plant within MATCH_M:
solar_farm with a solar plant, wind_farm with a wind plant, and power_plant
with a plant that is neither (the ontology routes solar and wind to their own
classes first). A plant of the other kind is shown as a flag, since it points
at a mislabel rather than confirming one.

    python -m curation.review.eia_plants           # then rebuild the queue

The layer is fetched once and cached at data/label_review/sources/eia_plants/.
"""

import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ..paths import REVIEW_DIR, REVIEW_QUEUE
from .hifld_lines import US_BOX, _tokens

LAYER = ("https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
         "Power_Plants_in_the_US/FeatureServer/0")
CACHE = REVIEW_DIR / "sources" / "eia_plants" / "plants.csv"
EVIDENCE = REVIEW_DIR / "sources" / "eia_plants.csv"
FIELDS = "Plant_Code,Plant_Name,Utility_Na,PrimSource,Total_MW,Latitude,Longitude"
SEARCH_M = 1500
MATCH_M = 1000
EARTH_R = 6371008.8

CLASS_SOURCES = {
    "energy.generation.solar_farm": lambda s: s == "solar",
    "energy.generation.wind_farm": lambda s: s == "wind",
    "energy.generation.power_plant": lambda s: s not in ("solar", "wind"),
}


def fetch() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, keep_default_na=False)
    rows, offset = [], 0
    while True:
        params = {"where": "1=1", "outFields": FIELDS, "returnGeometry": "false",
                  "orderByFields": "OBJECTID", "resultOffset": offset,
                  "resultRecordCount": 2000, "f": "json"}
        with urllib.request.urlopen(f"{LAYER}/query?{urllib.parse.urlencode(params)}", timeout=120) as r:
            data = json.load(r)
        if "error" in data:
            raise RuntimeError(data["error"])
        rows += [f["attributes"] for f in data["features"]]
        if not data.get("exceededTransferLimit"):
            break
        offset += len(data["features"])
    plants = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    plants.to_csv(CACHE, index=False)
    return plants


def build() -> pd.DataFrame:
    plants = fetch()
    plants = plants[plants.Latitude.astype(float).abs() > 0].reset_index(drop=True)
    q = pd.read_csv(REVIEW_QUEUE, keep_default_na=False).drop_duplicates("asset_id")
    lo_lat, hi_lat, lo_lon, hi_lon = US_BOX
    tiles = q[(q.sector == "energy") & q.lat.between(lo_lat, hi_lat)
              & q.lon.between(lo_lon, hi_lon)].reset_index(drop=True)

    tree = BallTree(np.radians(plants[["Latitude", "Longitude"]].astype(float).values), metric="haversine")
    idx, dist = tree.query_radius(np.radians(tiles[["lat", "lon"]].values),
                                  r=SEARCH_M / EARTH_R, return_distance=True, sort_results=True)

    out = []
    for t, ii, dd in zip(tiles.to_dict("records"), idx, dist):
        near = [(plants.iloc[i], d * EARTH_R) for i, d in zip(ii, dd)]
        agrees = CLASS_SOURCES.get(t["asset_type"])
        osm_tokens = _tokens(t["name"])
        corroborated = bool(agrees) and any(m <= MATCH_M and agrees(str(p.PrimSource)) for p, m in near)
        lines, strict = [], False
        for p, m in near[:3]:
            src = str(p.PrimSource)
            name_match = bool(osm_tokens & _tokens(str(p.Plant_Name)))
            line = (f"EIA plant: {p.Plant_Name} ({src}, {float(p.Total_MW or 0):.0f} MW) {m:.0f} m away"
                    + (" (matches OSM name)" if name_match else ""))
            kind = t["asset_type"].split(".")[-1]
            # a plant of the wrong kind only matters when nothing nearby agrees;
            # batteries sit beside every kind of plant, so they never argue.
            if agrees and not corroborated and m <= MATCH_M and not agrees(src) and src != "batteries":
                line = f"Caution: {line}, not what {kind} implies"
            # substations sit beside plants, so only a name match this close
            # suggests the tile is the plant itself.
            elif not agrees and name_match and m <= 300:
                line = f"Caution: {line}; this may be the plant, not a substation"
            # strict enough to pass without review: the right kind of plant,
            # close, and named like the OSM asset
            strict = strict or bool(agrees and agrees(src) and name_match and m <= 300)
            lines.append(line)
        if not near and agrees:
            lines.append(f"EIA: no plant within {SEARCH_M / 1000:.1f} km")
        out.append({"asset_id": t["asset_id"], "asset_type": t["asset_type"],
                    "eia_nearest_m": round(near[0][1]) if near else None,
                    "eia_source": str(near[0][0].PrimSource) if near else "",
                    "eia_text": " | ".join(lines), "corroborated": corroborated,
                    "auto_ok": strict and not any(l.startswith("Caution") for l in lines)})
    return pd.DataFrame(out)


def main():
    ev = build()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(EVIDENCE, index=False)
    has = ev.eia_text != ""
    print(f"wrote {len(ev):,} US energy tiles to {EVIDENCE}")
    print(pd.DataFrame({
        "tiles": ev.groupby("asset_type").size(),
        "plant within 1.5 km": ev[ev.eia_nearest_m.notna()].groupby("asset_type").size(),
        "corroborated": ev[ev.corroborated].groupby("asset_type").size(),
        "flagged": ev[ev.eia_text.str.contains("Caution")].groupby("asset_type").size(),
    }).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
