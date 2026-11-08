"""
registers.py
------------
Secondary evidence from official point registers of stations and airports:

  NARN    North American Rail Network nodes (BTS NTAD), the 1,733 nodes with a
          passenger station name (PASSNGRSTN), for North American stations.
          https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_North_American_Rail_Network_Nodes/FeatureServer/0
  NaPTAN  the UK national rail stations (DfT NaPTAN, ATCO area 910, stop type
          RLY), for stations in Great Britain. Open Government Licence v3.
          https://naptan.api.dft.gov.uk/v1/access-nodes?atcoAreaCodes=910&dataFormat=csv
  FAA     aviation facilities (BTS NTAD, from FAA NASR), the 13,005 sites of
          type A, for North American airports.
          https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_Aviation_Facilities/FeatureServer/0
  DB      Deutsche Bahn InfraGO OpenStation NeTEx, every DB passenger station
          (CC0), for Germany. https://mobilithek.info/offers/879076212433727488
  Trafikverket  INSPIRE railway station nodes (CC0), for Sweden.
  IE-NaPTAN     Ireland's NaPTAN rail entries (NTA, CC-BY 4.0), for Ireland.
  SNCF          SNCF passenger stations, "gares de voyageurs" (ODbL), France.
  Digitraffic   Fintraffic rail stations with passenger traffic (CC BY 4.0), Finland.
  SBB           SBB stations with published station plans (Trafimage), Switzerland.

Spain's ADIF network was also considered, but its WFS refuses access.

None of these are derived from OSM. A tile is corroborated when a register
entry lies within that register's radius of the OSM point, and passes without
review (auto_ok) only when the register entry's name also matches the OSM
name. Station radii are tight because station points sit on the platform or
the track; airport radii are wide because an airport's reference point can
be a kilometre or more from the centroid of its mapped outline.

    python -m curation.review.registers            # then rebuild the queue

Each register is fetched once and cached under data/label_review/sources/registers/.
"""

import gzip
import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ..paths import REVIEW_DIR, REVIEW_QUEUE
from .hifld_lines import _tokens

CACHE_DIR = REVIEW_DIR / "sources" / "registers"
EVIDENCE = REVIEW_DIR / "sources" / "registers.csv"
EARTH_R = 6371008.8
BTS = "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services"


def _arcgis(url: str, where: str, fields: str) -> list:
    rows, offset = [], 0
    while True:
        params = {"where": where, "outFields": fields, "returnGeometry": "true", "outSR": 4326,
                  "orderByFields": "OBJECTID", "resultOffset": offset,
                  "resultRecordCount": 2000, "f": "json"}
        with urllib.request.urlopen(f"{url}/query?{urllib.parse.urlencode(params)}", timeout=120) as r:
            data = json.load(r)
        if "error" in data:
            raise RuntimeError(data["error"])
        rows += [{**f["attributes"], "lon": f["geometry"]["x"], "lat": f["geometry"]["y"]}
                 for f in data["features"] if f.get("geometry")]
        if not data.get("exceededTransferLimit"):
            return rows
        offset += len(data["features"])


def _narn() -> pd.DataFrame:
    rows = _arcgis(f"{BTS}/NTAD_North_American_Rail_Network_Nodes/FeatureServer/0",
                   "PASSNGRSTN IS NOT NULL AND PASSNGRSTN <> ''", "OBJECTID,PASSNGRSTN,PASSNGR,STATE,COUNTRY")
    df = pd.DataFrame(rows)
    return pd.DataFrame({"name": df.PASSNGRSTN, "lat": df.lat, "lon": df.lon,
                         "detail": "passenger station node, " + df.STATE.fillna("") + " " + df.COUNTRY.fillna("")})


def _naptan() -> pd.DataFrame:
    url = "https://naptan.api.dft.gov.uk/v1/access-nodes?atcoAreaCodes=910&dataFormat=csv"
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = CACHE_DIR / "naptan_910_raw.csv"
        raw.write_bytes(r.read())
    df = pd.read_csv(raw, low_memory=False)
    df = df[(df.StopType == "RLY") & (df.Status.fillna("active") == "active")]
    return pd.DataFrame({"name": df.CommonName, "lat": df.Latitude, "lon": df.Longitude,
                         "detail": "national rail station " + df.ATCOCode})


def _faa() -> pd.DataFrame:
    rows = _arcgis(f"{BTS}/NTAD_Aviation_Facilities/FeatureServer/0", "SITE_TYPE_CODE = 'A'",
                   "OBJECTID,ARPT_NAME,ARPT_ID,ICAO_ID,FACILITY_USE_CODE,ARPT_STATUS")
    df = pd.DataFrame(rows)
    use = df.FACILITY_USE_CODE.map({"PU": "public", "PR": "private"}).fillna("")
    return pd.DataFrame({"name": df.ARPT_NAME, "lat": df.lat, "lon": df.lon,
                         "detail": ("FAA " + df.ARPT_ID.fillna("") + ", " + use + ", " + df.ARPT_STATUS.fillna(""))})


def _download(url: str, name: str):
    path = CACHE_DIR / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=600) as r, open(path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    return path


def _db() -> pd.DataFrame:
    """Deutsche Bahn InfraGO OpenStation NeTEx (CC0): every DB passenger
    station. The file is ~300 MB of NeTEx, so stream it and keep only each
    StopPlace's name and position. StopPlaces carry no centroid of their own;
    only some equipment inside them (lifts, stairs, entrances) does, so a
    station sits at the mean of its equipment centroids, and stations with
    none are left out."""
    import xml.etree.ElementTree as ET
    path = _download("https://mobilithek.info/mdp-api/mdp-conn-server/v1/publication/"
                     "879076212433727488/file/noauth", "db_openstation_netex.xml")
    local = lambda tag: tag.rsplit("}", 1)[-1]
    rows = []
    for _, el in ET.iterparse(path, events=("end",)):
        if local(el.tag) != "StopPlace":
            continue
        name = mode = None
        for child in el:
            t = local(child.tag)
            if t == "Name":
                name = (child.text or "").strip() or next((c.text for c in child), None)
            elif t == "TransportMode":
                mode = child.text
        lats = [float(c.text) for c in el.iter() if local(c.tag) == "Latitude"]
        lons = [float(c.text) for c in el.iter() if local(c.tag) == "Longitude"]
        if name and lats and len(lats) == len(lons) and mode in (None, "rail"):
            rows.append({"name": name, "lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons),
                         "detail": f"DB station {el.get('id', '')}, located from {len(lats)} equipment point(s)"})
        el.clear()
    return pd.DataFrame(rows).drop_duplicates(["name", "lat", "lon"])


def _trafikverket() -> pd.DataFrame:
    """Trafikverket INSPIRE railway station nodes (CC0), Sweden. The WFS
    returns EPSG:4326 in latitude, longitude order."""
    path = _download("http://geo-inspire.trafikverket.se/MapService/wfs.axd/TN_RailTransportNetwork?"
                     "service=WFS&version=2.0.0&request=GetFeature&typenames=RailwayStationNode"
                     "&outputFormat=application/json&srsName=urn:ogc:def:crs:EPSG::4326",
                     "se_railway_station_node.json")
    feats = json.loads(path.read_text(encoding="utf-8-sig"))["features"]
    return pd.DataFrame([{"name": f["properties"].get("geographicalName"),
                          "lat": f["geometry"]["coordinates"][0], "lon": f["geometry"]["coordinates"][1],
                          "detail": "Trafikverket " + (f["properties"].get("formOfNode") or "")}
                         for f in feats if f.get("geometry")])


def _ie_naptan() -> pd.DataFrame:
    """Ireland's NaPTAN (National Transport Authority, CC-BY 4.0), rail entries."""
    path = _download("https://www.transportforireland.ie/transitData/Data/NaPTAN_Stop_Points.csv",
                     "ie_naptan_stop_points.csv")
    df = pd.read_csv(path, low_memory=False)
    df = df[df.StopType == "RLY"]
    return pd.DataFrame({"name": df.CommonName, "lat": df.Latitude, "lon": df.Longitude,
                         "detail": "Irish rail stop " + df.AtcoCode.astype(str)})


def _explore(base: str, dataset: str, select: str) -> list:
    """every record of an Opendatasoft explore v2.1 dataset, via its JSON export."""
    url = f"{base}/api/explore/v2.1/catalog/datasets/{dataset}/exports/json?{urllib.parse.urlencode({'select': select})}"
    path = _download(url, f"{dataset}.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _sncf() -> pd.DataFrame:
    """SNCF passenger stations, 'gares de voyageurs' (ODbL, SNCF's own data), France."""
    rows = _explore("https://ressources.data.sncf.com", "gares-de-voyageurs", "nom,codes_uic,position_geographique")
    return pd.DataFrame([{"name": r["nom"], "lat": r["position_geographique"]["lat"],
                          "lon": r["position_geographique"]["lon"], "detail": f"SNCF station UIC {r.get('codes_uic')}"}
                         for r in rows if r.get("position_geographique")])


def _digitraffic() -> pd.DataFrame:
    """Fintraffic Digitraffic rail stations with passenger traffic (CC BY 4.0), Finland."""
    path = CACHE_DIR / "fi_digitraffic_stations.json"
    if not path.exists():
        # Digitraffic refuses requests that do not accept gzip
        req = urllib.request.Request("https://rata.digitraffic.fi/api/v1/metadata/stations",
                                     headers={"Digitraffic-User": "infra-bench-cls", "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read()
        path.write_bytes(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)
    rows = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([{"name": r["stationName"], "lat": r["latitude"], "lon": r["longitude"],
                          "detail": f"Digitraffic {r['type'].lower().replace('_', ' ')} {r['stationShortCode']}"}
                         for r in rows if r.get("passengerTraffic")])


def _sbb() -> pd.DataFrame:
    """SBB stations with published station plans, 'haltestelle-karte-trafimage', Switzerland."""
    rows = _explore("https://data.sbb.ch", "haltestelle-karte-trafimage", "bezeichnung_offiziell,bpuic,geopos")
    return pd.DataFrame([{"name": r["bezeichnung_offiziell"], "lat": r["geopos"]["lat"], "lon": r["geopos"]["lon"],
                          "detail": f"SBB station {r.get('bpuic')}"} for r in rows if r.get("geopos")])


# register -> (fetcher, class it speaks to, tile box (lat lo, lat hi, lon lo, lon hi), radius m)
REGISTERS = {
    "NARN": (_narn, "transport.train_station", (14.0, 72.0, -170.0, -50.0), 300),
    "NaPTAN": (_naptan, "transport.train_station", (49.8, 60.9, -8.2, 1.8), 300),
    "FAA": (_faa, "transport.airport", (17.0, 72.0, -180.0, -64.0), 2000),
    "DB": (_db, "transport.train_station", (47.2, 55.1, 5.8, 15.1), 300),
    "Trafikverket": (_trafikverket, "transport.train_station", (55.3, 69.1, 11.0, 24.2), 300),
    "IE-NaPTAN": (_ie_naptan, "transport.train_station", (51.4, 55.4, -10.6, -5.9), 300),
    "SNCF": (_sncf, "transport.train_station", (41.3, 51.1, -5.2, 9.6), 300),
    "Digitraffic": (_digitraffic, "transport.train_station", (59.7, 70.1, 20.5, 31.6), 300),
    "SBB": (_sbb, "transport.train_station", (45.8, 47.85, 5.9, 10.5), 300),
}


def load(register: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{register}.csv"
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = REGISTERS[register][0]().dropna(subset=["lat", "lon"])
    df.to_csv(path, index=False)
    return df


def build() -> pd.DataFrame:
    q = pd.read_csv(REVIEW_QUEUE, keep_default_na=False).drop_duplicates("asset_id")
    out = []
    for register, (_, cls, (la0, la1, lo0, lo1), radius) in REGISTERS.items():
        reg = load(register)
        tiles = q[(q.asset_type == cls) & q.lat.between(la0, la1) & q.lon.between(lo0, lo1)]
        if tiles.empty:
            continue
        tree = BallTree(np.radians(reg[["lat", "lon"]].astype(float).values), metric="haversine")
        dist, idx = tree.query(np.radians(tiles[["lat", "lon"]].values), k=3)
        for t, dd, ii in zip(tiles.to_dict("records"), dist * EARTH_R, idx):
            osm_tokens = _tokens(t["name"])
            near = [(reg.iloc[i], d) for i, d in zip(ii, dd) if d <= 2 * radius]
            hits = [(r, d, bool(osm_tokens & _tokens(str(r["name"])))) for r, d in near]
            within = [(r, d, nm) for r, d, nm in hits if d <= radius]
            lines = [f"{register}: {r['name']} ({r['detail']}) {d:.0f} m away"
                     + (" (matches OSM name)" if nm else "") for r, d, nm in hits]
            if not lines:
                lines = [f"{register}: nothing within {2 * radius / 1000:g} km"]
            out.append({"asset_id": t["asset_id"], "asset_type": cls, "register": register,
                        "register_text": " | ".join(lines),
                        "corroborated": bool(within),
                        "auto_ok": any(nm for _, _, nm in within)})
    return pd.DataFrame(out)


def main():
    ev = build()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(EVIDENCE, index=False)
    print(f"wrote {len(ev):,} tiles to {EVIDENCE}")
    print(ev.groupby("register").agg(tiles=("asset_id", "size"),
                                     corroborated=("corroborated", "sum"),
                                     auto_ok=("auto_ok", "sum")).to_string())


if __name__ == "__main__":
    main()
