"""
grid_operators.py
-----------------
Substation evidence from grid operators' own open data, for the regions they
serve. Unlike HIFLD, these come straight from the operator.

  Transpower   New Zealand's national grid operator: its 229 sites (AC and HVDC
               substations and tee points) and its transmission lines with design
               voltage. CC BY 4.0, "Transpower Geospatial (CC BY)".
  Vector       Auckland's distribution company: its 114 zone substations, named
               with their voltages ("MANUREWA 33/11KV"). CC BY 4.0, credit Vector.
  EDF          EDF SEI Martinique's medium-voltage (HTA, 20 kV) lines, overhead and
               underground. Licence Ouverte 2.0 (Etalab): credit EDF SEI Martinique
               and the date of the last update.

  Geoscience Australia  national electricity infrastructure substations (CC BY
               4.0); the 678 GA itself took from OpenStreetMap are dropped. Zone
               substations under 100 kV imply distribution; transmission, terminal,
               switchyard and bulk supply points, and anything of 220 kV or more,
               imply transmission; plain "Substation" says neither.
  Vicmap       Victoria's Features of Interest (CC BY 4.0): power terminal stations
               (transmission) and power sub stations (zone, distribution).
  NSW Spatial Services  NSW Features of Interest building complexes named as
               substations (CC BY 4.0): ZONE implies distribution, TRANSMISSION or
               BULK SUPPLY transmission.
  KPLC, ZESCO  Kenya Power primary substations and Zambia's distribution
               substations, from energydata.info (CC BY 4.0), downloaded by hand
               into sources/grid_operators/manual/.
  Secretaría de Energía  Argentina's high-voltage transformer stations (CC BY
               4.0), from a 2023 snapshot of the national portal, with input and
               output voltages: all outputs of 100 kV or more imply transmission,
               100 kV in and less out steps down and implies neither.
  OSINERGMIN, Enedis  Peru's distribution substations (ODC-BY) and Enedis's
               HTA/BT postes in France (Licence Ouverte 2.0). Both list every small
               distribution substation, so they are queried per tile, support a
               distribution or untyped label within 50 m, and never flag another.

A point source says what kind of substation stands there: a Transpower site is
on the national grid, a Vector zone substation is a distribution substation. A
substation tile with one within the radius is corroborated when its OSM class
agrees, and passes without review (auto_ok) only when the names match too. A
tile whose OSM class says the opposite type is flagged; an untyped substation
gets the suggestion as a note, since "type not stated" is still true of it.

A line source counts the lines that end at the asset, as hifld_lines does:
Transpower lines of 110 kV and up support a transmission label, EDF HTA lines a
distribution one. Lines corroborate but never auto-accept, as they carry no
substation name to match.

    python -m curation.review.grid_operators          # then rebuild the queue

Each source is fetched once and cached under data/label_review/sources/grid_operators/.
"""

import json
import math
import re
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ..paths import REVIEW_DIR, REVIEW_QUEUE
from .hifld_lines import SUBSTATION_CLASSES, _metres, _tokens

CACHE_DIR = REVIEW_DIR / "sources" / "grid_operators"
EVIDENCE = REVIEW_DIR / "sources" / "grid_operators.csv"
EARTH_R = 6371008.8
TX, DX, UNTYPED = SUBSTATION_CLASSES[0], SUBSTATION_CLASSES[1], SUBSTATION_CLASSES[2]
NZ = (-47.6, -34.0, 166.0, 179.0)
AUCKLAND = (-37.4, -36.0, 174.2, 175.9)
MARTINIQUE = (14.35, 14.92, -61.25, -60.78)


def _arcgis(url: str, fields: str) -> list:
    rows, offset = [], 0
    while True:
        params = {"where": "1=1", "outFields": fields, "returnGeometry": "true", "outSR": 4326,
                  "orderByFields": "OBJECTID", "resultOffset": offset, "resultRecordCount": 2000, "f": "json"}
        with urllib.request.urlopen(f"{url}/query?{urllib.parse.urlencode(params)}", timeout=120) as r:
            data = json.load(r)
        if "error" in data:
            raise RuntimeError(data["error"])
        rows += data["features"]
        if not data.get("exceededTransferLimit"):
            return rows
        offset += len(data["features"])


def _cached(name: str, fetch) -> list:
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fetch()), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


TRANSPOWER = "https://services3.arcgis.com/AkUq3zcWf7TVqyR9/arcgis/rest/services"
VECTOR = ("https://services6.arcgis.com/8RWEO35G1ALMME0I/arcgis/rest/services/"
          "distribution_feeder_network_and_zone_substations/FeatureServer")
EDF = "https://opendata-martinique.edf.fr/data-fair/api/v1/datasets"


def transpower_sites(_tiles=None) -> pd.DataFrame:
    feats = _cached("transpower_sites", lambda: _arcgis(f"{TRANSPOWER}/Sites/FeatureServer/0",
                                                        "OBJECTID,MXLOCATION,type,status,description"))
    rows = [{"name": f["attributes"]["description"] or "", "lat": f["geometry"]["y"], "lon": f["geometry"]["x"],
             "detail": f"Transpower {f['attributes']['type']} {f['attributes']['MXLOCATION']}"}
            for f in feats if f.get("geometry") and f["attributes"].get("type") in ("ACSTN", "HVDC")]
    return pd.DataFrame(rows)


def vector_zone_subs(_tiles=None) -> pd.DataFrame:
    feats = _cached("vector_zone_substations", lambda: _arcgis(
        f"{VECTOR}/1", "OBJECTID,Primary_Zone_Substation_Code,Primary_Zone_Substation_Name"))
    return pd.DataFrame([{"name": f["attributes"]["Primary_Zone_Substation_Name"] or "",
                          "lat": f["geometry"]["y"], "lon": f["geometry"]["x"],
                          "detail": f"Vector zone substation {f['attributes']['Primary_Zone_Substation_Code']}"}
                         for f in feats if f.get("geometry")])


def transpower_lines() -> list:
    """(kv, [(lat, lon) endpoints]) per Transpower line."""
    feats = _cached("transpower_lines", lambda: _arcgis(f"{TRANSPOWER}/TransmissionLines/FeatureServer/0",
                                                        "OBJECTID,designvolt,type"))
    out = []
    for f in feats:
        kv = f["attributes"].get("designvolt")
        ends = [(p[1], p[0]) for path in f.get("geometry", {}).get("paths", []) for p in (path[0], path[-1])]
        out.append((float(kv) if kv else None, ends))
    return out


def edf_hta_lines() -> list:
    """(20.0, endpoints) per EDF Martinique HTA line, overhead and underground."""
    def fetch():
        lines = []
        for ds in ("lignes-haute-tension-hta-aerien-mar", "lignes-haute-tension-hta-souterrain-mar"):
            url = f"{EDF}/{ds}/lines?size=5000&select=geo_shape"
            while url:
                with urllib.request.urlopen(url, timeout=120) as r:
                    page = json.load(r)
                lines += [json.loads(x["geo_shape"])["coordinates"] for x in page["results"] if x.get("geo_shape")]
                url = page.get("next") if page["results"] else None
        return lines
    out = []
    for coords in _cached("edf_martinique_hta", fetch):
        if coords and isinstance(coords[0][0], (int, float)):
            coords = [coords]  # LineString -> one path
        out.append((20.0, [(p[1], p[0]) for path in coords for p in (path[0], path[-1])]))
    return out


GA = "https://services.ga.gov.au/gis/rest/services/Electricity_Infrastructure/MapServer/0"
NSW_FOI = ("https://portal.spatial.nsw.gov.au/server/rest/services/"
           "NSW_Features_of_Interest_Category_multiCRS/MapServer/3")
VICMAP = "https://opendata.maps.vic.gov.au/geoserver/wfs"
PERU_SED = ("https://gisem.osinergmin.gob.pe/serverosih/rest/services/OGC/"
            "PeruOsinergmin019SubestacionDistribucionElectrica/MapServer/25")
ENEDIS = "https://opendata.enedis.fr/data-fair/api/v1/datasets/mz4i8bhjlm0k18i0af3aovtf"
MANUAL = CACHE_DIR / "manual"

AUSTRALIA = (-44.0, -10.0, 112.0, 154.0)
VICTORIA = (-39.2, -33.9, 140.9, 150.0)
NEW_SOUTH_WALES = (-37.6, -28.1, 140.9, 153.7)
PERU = (-18.4, 0.0, -81.4, -68.6)
FRANCE = (41.3, 51.1, -5.2, 9.6)
KENYA = (-4.7, 5.0, 33.9, 41.9)
ZAMBIA = (-18.1, -8.2, 21.9, 33.7)
ARGENTINA = (-55.1, -21.7, -73.6, -53.6)


def _kvs(text) -> list:
    """the voltages in a label like 'PARKLANDS ( 66 / 11 KV)' or '13.2 | 33'."""
    return [float(x) for x in re.findall(r"\d+(?:[.,]\d+)?", str(text).replace(",", "."))
            if 1 <= float(x) <= 1200]


def _by_voltage(high: list, low: list = None):
    """the class a substation's voltages imply. One that takes in 100 kV or more
    and sends out less steps transmission down to distribution, which OSM
    files under either, so it implies neither."""
    low = high if low is None else low
    if not high:
        return None
    if min(low or high) >= 100:
        return TX
    return DX if max(high) < 100 else None


def _ga_kind(kind, kv):
    """GA files some 275 and 330 kV sites as "Zone", so voltage overrides the
    subtype: 220 kV and up is transmission, and a "Zone" at 100-219 kV may
    step down from transmission, so it says neither."""
    if kv and kv >= 220:
        return TX
    if kind == DX and kv and kv >= 100:
        return None
    return kind


def ga_substations(_tiles=None) -> pd.DataFrame:
    """Geoscience Australia's substations, less the ones GA took from OSM."""
    feats = _cached("ga_substations", lambda: _arcgis(
        GA, "OBJECTID,FEATURE_NAME,FEATURE_SUBTYPE,VOLTAGE_KV,ATTRIBUTE_SOURCE,STATUS"))
    kinds = {"Zone": DX, "Transmission": TX, "Terminal": TX, "Switchyard": TX, "Bulk Supply Point": TX}
    rows = []
    for f in feats:
        a = {k.upper(): v for k, v in f["attributes"].items()}  # the service answers in lower case
        if not f.get("geometry") or "open street map" in (a.get("ATTRIBUTE_SOURCE") or "").lower():
            continue
        kv = a.get("VOLTAGE_KV")
        rows.append({"name": a.get("FEATURE_NAME") or "", "lat": f["geometry"]["y"], "lon": f["geometry"]["x"],
                     "detail": f"GA {a.get('FEATURE_SUBTYPE') or 'substation'}" + (f", {kv:g} kV" if kv else ""),
                     "implies": _ga_kind(kinds.get(a.get("FEATURE_SUBTYPE")), kv)})
    return pd.DataFrame(rows)


def nsw_substations(_tiles=None) -> pd.DataFrame:
    """NSW Features of Interest building complexes named as substations."""
    def fetch():
        rows, offset = [], 0
        while True:
            params = {"where": "generalname like '%SUBSTATION%'", "outFields": "objectid,generalname",
                      "returnGeometry": "true", "outSR": 4326, "resultOffset": offset,
                      "resultRecordCount": 1000, "f": "json"}
            with urllib.request.urlopen(f"{NSW_FOI}/query?{urllib.parse.urlencode(params)}", timeout=120) as r:
                data = json.load(r)
            if "error" in data:
                raise RuntimeError(data["error"])
            rows += data["features"]
            if not data.get("exceededTransferLimit") or not data["features"]:
                return rows
            offset += len(data["features"])
    out = []
    for f in _cached("nsw_foi_substations", fetch):
        name = f["attributes"]["generalname"] or ""
        implies = (DX if "ZONE SUBSTATION" in name else
                   TX if any(w in name for w in ("TRANSMISSION SUBSTATION", "BULK SUPPLY", "SWITCHING STATION"))
                   else None)
        out.append({"name": name, "lat": f["geometry"]["y"], "lon": f["geometry"]["x"],
                    "detail": "NSW Features of Interest", "implies": implies})
    return pd.DataFrame(out)


def vicmap_substations(_tiles=None) -> pd.DataFrame:
    """Vicmap Features of Interest power facilities: terminal stations are on the
    transmission network, sub stations are zone (distribution) substations."""
    def fetch():
        params = {"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                  "typeNames": "open-data-platform:foi_polygon",
                  "cql_filter": "feature_type='power facility' AND feature_subtype IN "
                                "('power sub station','power terminal station')",
                  "outputFormat": "application/json", "srsName": "EPSG:4326", "count": 5000}
        with urllib.request.urlopen(f"{VICMAP}?{urllib.parse.urlencode(params)}", timeout=180) as r:
            return json.load(r)["features"]
    from shapely.geometry import shape
    out = []
    for f in _cached("vicmap_power_facilities", fetch):
        if not f.get("geometry"):
            continue
        c = shape(f["geometry"]).centroid
        sub = f["properties"]["feature_subtype"]
        out.append({"name": f["properties"].get("name") or "", "lat": c.y, "lon": c.x,
                    "detail": f"Vicmap {sub}", "implies": TX if sub == "power terminal station" else DX})
    return pd.DataFrame(out)


def _per_tile(source: str, tiles: pd.DataFrame, half_m: float, fetch) -> list:
    """for sources too big to download whole: ask for the points around each
    tile, cached per tile."""
    folder = CACHE_DIR / source
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for t in tiles.itertuples():
        path = folder / f"{t.asset_id}.json"
        if not path.exists():
            dlat = half_m / 111320
            dlon = half_m / (111320 * math.cos(math.radians(t.lat)))
            path.write_text(json.dumps(fetch(t.lon - dlon, t.lat - dlat, t.lon + dlon, t.lat + dlat)),
                            encoding="utf-8")
        out += json.loads(path.read_text(encoding="utf-8"))
    return out


def peru_sed(tiles: pd.DataFrame) -> pd.DataFrame:
    """OSINERGMIN's distribution substations (SED): 139k points, so per tile."""
    def fetch(x0, y0, x1, y1):
        params = {"geometry": f"{x0},{y0},{x1},{y1}", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                  "outSR": 4326, "spatialRel": "esriSpatialRelIntersects",
                  "outFields": "CODEMP,CODSED,ETIQUETACAMPO", "returnGeometry": "true", "f": "json"}
        with urllib.request.urlopen(f"{PERU_SED}/query?{urllib.parse.urlencode(params)}", timeout=120) as r:
            data = json.load(r)
        if "error" in data:
            raise RuntimeError(data["error"])
        return [{"name": "", "lat": f["geometry"]["y"], "lon": f["geometry"]["x"],
                 "detail": f"OSINERGMIN SED {f['attributes']['CODSED']} ({f['attributes']['CODEMP']})"}
                for f in data["features"] if f.get("geometry")]
    return pd.DataFrame(_per_tile("peru_sed", tiles, 300, fetch))


def enedis_postes(tiles: pd.DataFrame) -> pd.DataFrame:
    """Enedis's HTA/BT substations (postes de distribution publique): 977k
    points, so per tile."""
    def fetch(x0, y0, x1, y1):
        url = f"{ENEDIS}/lines?bbox={x0},{y0},{x1},{y1}&size=1000&select=geometry,nom_commune"
        out = []
        while url:
            with urllib.request.urlopen(url, timeout=120) as r:
                page = json.load(r)
            for x in page["results"]:
                lon, lat = json.loads(x["geometry"])["coordinates"]
                out.append({"name": "", "lat": lat, "lon": lon, "detail": f"Enedis poste HTA/BT, {x.get('nom_commune', '')}"})
            url = page.get("next") if page["results"] else None
        return out
    return pd.DataFrame(_per_tile("enedis_postes", tiles, 300, fetch))


def _shapefile(zip_name: str):
    import zipfile
    import geopandas as gpd
    path = MANUAL / zip_name
    if not path.exists():
        raise FileNotFoundError(f"download it by hand to {path} (see README)")
    inner = next(n for n in zipfile.ZipFile(path).namelist() if n.lower().endswith(".shp"))
    gdf = gpd.read_file(f"zip://{path}!{inner}").to_crs(4326)
    pts = gdf.geometry.representative_point()
    return gdf.assign(lat=pts.y, lon=pts.x)


def kenya_kplc(_tiles=None) -> pd.DataFrame:
    """KPLC primary substations (energydata.info, CC BY 4.0), named with their
    voltages, e.g. 'PARKLANDS ( 66 / 11 KV)'."""
    g = _shapefile("kenya_kplc_primary_substations.zip")
    out = []
    for r in g.itertuples():
        label = str(getattr(r, "Name_IMS11", "") or "")
        kv = _kvs(label.split("(")[-1]) if "(" in label else []
        out.append({"name": label.split("(")[0].strip() or str(getattr(r, "Name8", "") or ""),
                    "lat": r.lat, "lon": r.lon, "detail": f"KPLC primary substation {label}",
                    "implies": _by_voltage(kv[:1], kv[1:]) if kv else DX})
    return pd.DataFrame(out)


def zambia_zesco(_tiles=None) -> pd.DataFrame:
    """ZESCO distribution substations (energydata.info, CC BY 4.0)."""
    g = _shapefile("zambia_distribution_substations.zip")
    return pd.DataFrame([{"name": str(r.NAME or ""), "lat": r.lat, "lon": r.lon,
                          "detail": f"ZESCO distribution substation, {r.VOLATGELEV} kV", "implies": DX}
                         for r in g.itertuples()])


def argentina_at(_tiles=None) -> pd.DataFrame:
    """Argentina's high-voltage transformer stations (Secretaría de Energía,
    CC BY 4.0), with their input and output voltages."""
    g = _shapefile("argentina_at_estaciones_transformadoras.zip")
    out = []
    for r in g.itertuples():
        high, low = _kvs(r.TENSION_EN), _kvs(r.TENSION_SA)
        out.append({"name": str(r.NOMBRE or ""), "lat": r.lat, "lon": r.lon,
                    "detail": f"Secretaría de Energía ET, {r.TENSION_EN} kV in, {r.TENSION_SA} kV out",
                    "implies": _by_voltage(high, low)})
    return pd.DataFrame(out)


# point sources: (loader, region box, radius m, the substation class it implies
# when a row does not say, dense). A loader returns name, lat, lon, detail and
# optionally implies per row (None: the row does not say which type it is).
# Dense sources list every small distribution substation, so one stands near
# almost any substation: they support a distribution label but never flag
# another, and carry no names to auto-accept on.
POINTS = {
    "Transpower": (transpower_sites, NZ, 250, TX, False),
    "Vector": (vector_zone_subs, AUCKLAND, 250, DX, False),
    "Geoscience Australia": (ga_substations, AUSTRALIA, 250, None, False),
    "Vicmap": (vicmap_substations, VICTORIA, 250, None, False),
    "NSW Spatial Services": (nsw_substations, NEW_SOUTH_WALES, 250, None, False),
    "KPLC": (kenya_kplc, KENYA, 300, DX, False),
    "ZESCO": (zambia_zesco, ZAMBIA, 300, DX, False),
    "Secretaría de Energía": (argentina_at, ARGENTINA, 300, None, False),
    "OSINERGMIN": (peru_sed, PERU, 50, DX, True),
    "Enedis": (enedis_postes, FRANCE, 50, DX, True),
}
# line sources: (loader, region box, endpoint radius m, min kV, the class it implies)
LINES = {
    "Transpower lines": (transpower_lines, NZ, 250, 110, TX),
    "EDF HTA": (edf_hta_lines, MARTINIQUE, 150, 0, DX),
}
SHORT = {TX: "transmission", DX: "distribution", UNTYPED: "untyped"}


def _in(df, box):
    a, b, c, d = box
    return df.lat.between(a, b) & df.lon.between(c, d)


def build() -> pd.DataFrame:
    q = pd.read_csv(REVIEW_QUEUE, keep_default_na=False).drop_duplicates("asset_id")
    subs = q[q.asset_type.isin(SUBSTATION_CLASSES)]
    out = []
    for source, (load, box, radius, default, dense) in POINTS.items():
        tiles = subs[_in(subs, box)]
        if tiles.empty:
            continue
        try:
            pts = load(tiles)
        except FileNotFoundError as e:  # a hand-downloaded zip that is not there
            print(f"skipping {source}: {e}")
            continue
        if pts.empty:
            for t in tiles.to_dict("records"):
                out.append({"asset_id": t["asset_id"], "asset_type": t["asset_type"], "source": source,
                            "grid_text": f"{source}: no substation within {2 * radius} m",
                            "corroborated": False, "auto_ok": False})
            continue
        if "implies" not in pts:
            pts["implies"] = default
        tree = BallTree(np.radians(pts[["lat", "lon"]].astype(float).values), metric="haversine")
        dist, idx = tree.query(np.radians(tiles[["lat", "lon"]].values), k=1)
        for t, d, i in zip(tiles.to_dict("records"), dist[:, 0] * EARTH_R, idx[:, 0]):
            p = pts.iloc[i]
            implies = p["implies"] if isinstance(p["implies"], str) else None
            near = d <= radius
            name_match = bool(_tokens(t["name"]) & _tokens(str(p["name"])))
            agrees = near and t["asset_type"] == implies
            label = f"{p['name']} ({p['detail']})" if p["name"] else p["detail"]
            line = (f"{source}: {label} {d:.0f} m away" + (" (matches OSM name)" if name_match else "")
                    if d <= 2 * radius else f"{source}: no substation within {2 * radius} m")
            if implies is None or dense:
                # a row that does not say its type, or a dense source, informs
                # but never argues with the label
                if near and implies is None:
                    line += "; the source does not say whether it is transmission or distribution"
                elif near and t["asset_type"] == UNTYPED:
                    line += f"; suggests a {SHORT[implies]} substation"
                out.append({"asset_id": t["asset_id"], "asset_type": t["asset_type"], "source": source,
                            "grid_text": line, "corroborated": agrees, "auto_ok": False})
                continue
            if near and t["asset_type"] not in (implies, UNTYPED):
                # grid and distribution substations often stand side by side, so
                # this asks the reviewer to check, rather than calling OSM wrong
                line = (f"Caution: {line}; the operator lists a {SHORT[implies]} substation here and OSM "
                        f"says {SHORT[t['asset_type']]}: check whether they are the same facility")
            elif near and t["asset_type"] == UNTYPED:
                line += f"; suggests a {SHORT[implies]} substation"
            out.append({"asset_id": t["asset_id"], "asset_type": t["asset_type"], "source": source,
                        "grid_text": line, "corroborated": agrees, "auto_ok": agrees and name_match})
    for source, (load, box, radius, min_kv, implies) in LINES.items():
        tiles = subs[_in(subs, box)]
        if tiles.empty:
            continue
        lines = load()
        for t in tiles.to_dict("records"):
            ending = [kv for kv, ends in lines
                      if any(_metres(t["lat"], t["lon"], la, lo) <= radius for la, lo in ends)]
            strong = [kv for kv in ending if kv is not None and kv >= min_kv]
            line = (f"{source}: {len(ending)} line(s) end here"
                    + (f" (max {max(k for k in ending if k):g} kV)" if any(ending) else "")
                    if ending else f"{source}: no line ends within {radius} m")
            agrees = bool(strong) and t["asset_type"] == implies
            if strong and t["asset_type"] not in (implies, UNTYPED) and implies == TX:
                line = f"Caution: {line}, which a {SHORT[t['asset_type']]} label does not expect"
            elif strong and t["asset_type"] == UNTYPED:
                line += f"; suggests a {SHORT[implies]} substation"
            elif strong and implies == DX and t["asset_type"] == TX:
                # distribution feeders leaving a substation make it a
                # transmission-to-distribution one, which OSM calls distribution
                line += "; distribution feeders leave it, so it steps down to distribution"
            out.append({"asset_id": t["asset_id"], "asset_type": t["asset_type"], "source": source,
                        "grid_text": line, "corroborated": agrees, "auto_ok": False})
    return pd.DataFrame(out, columns=["asset_id", "asset_type", "source", "grid_text", "corroborated", "auto_ok"])


def main():
    ev = build()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(EVIDENCE, index=False)
    print(f"wrote {len(ev):,} rows to {EVIDENCE}")
    print(ev.groupby("source").agg(tiles=("asset_id", "size"), corroborated=("corroborated", "sum"),
                                   auto_ok=("auto_ok", "sum"),
                                   flagged=("grid_text", lambda s: s.str.startswith("Caution").sum())).to_string())


if __name__ == "__main__":
    main()
