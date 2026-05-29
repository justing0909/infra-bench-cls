"""
Extract the 1 data_center way in central-america via two lightweight passes
(no location cache needed for the whole 112M-node source).

Pass 1: find ways with building=data_center; collect referenced node IDs.
Pass 2: capture lat/lon for just those node IDs.
Then compute centroid + write parquet.
"""
from __future__ import annotations
import os, sys, time, json
import osmium
import pandas as pd

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
SRC  = r"D:\central-america-260526.osm.pbf"

class WayCollector(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.matched_ways: list = []   # [(way_id, [node_ids], dict(tags))]
        self.target_node_ids: set = set()
        self.matched_relations: list = []  # [(rel_id, [(type, ref)], dict(tags))]
    def node(self, n):
        # Node-tagged data_centers (probably 0 in this dataset, but be safe)
        if n.tags.get("building") == "data_center":
            self.matched_ways.append(("node", n.id, [], dict(n.tags),
                                      n.location.lat, n.location.lon))
    def way(self, w):
        if w.tags.get("building") == "data_center":
            node_ids = [nd.ref for nd in w.nodes]
            self.target_node_ids.update(node_ids)
            self.matched_ways.append(("way", w.id, node_ids, dict(w.tags), None, None))
    def relation(self, r):
        if r.tags.get("building") == "data_center":
            members = [(m.type, m.ref) for m in r.members]
            for mtype, mref in members:
                if mtype == 'n':
                    self.target_node_ids.add(mref)
            self.matched_relations.append((r.id, members, dict(r.tags)))


class LocLookup(osmium.SimpleHandler):
    def __init__(self, target_ids: set):
        super().__init__()
        self.target_ids = target_ids
        self.locs: dict = {}
    def node(self, n):
        if n.id in self.target_ids:
            if n.location.valid():
                self.locs[n.id] = (n.location.lat, n.location.lon)


def main():
    print("Pass 1: find building=data_center features...", flush=True)
    t = time.time()
    wc = WayCollector()
    wc.apply_file(SRC)
    print(f"  pass 1 wall: {time.time()-t:.1f}s", flush=True)
    print(f"  matched ways/nodes: {len(wc.matched_ways)}", flush=True)
    print(f"  matched relations:  {len(wc.matched_relations)}", flush=True)
    print(f"  member node IDs to look up: {len(wc.target_node_ids):,}", flush=True)

    if not wc.matched_ways and not wc.matched_relations:
        print("No data_center features found; nothing to extract.", flush=True)
        return

    print("Pass 2: gather member node locations...", flush=True)
    t = time.time()
    ll = LocLookup(wc.target_node_ids)
    ll.apply_file(SRC)
    print(f"  pass 2 wall: {time.time()-t:.1f}s", flush=True)
    print(f"  locations resolved: {len(ll.locs):,}/{len(wc.target_node_ids):,}", flush=True)

    rows = []
    for entry in wc.matched_ways:
        otype, oid, node_ids, tags, lat, lon = entry
        if otype == "node":
            rows.append({
                "asset_id":   f"osm_node_{oid}",
                "asset_type": "telecom.data_center",
                "lat":        lat,
                "lon":        lon,
                "name":       tags.get("name", ""),
                "source":     "osm_geofabrik",
            })
        else:
            coords = [ll.locs[nid] for nid in node_ids if nid in ll.locs]
            if not coords:
                print(f"  skipping {otype}/{oid}: no resolved node locations", flush=True)
                continue
            lat = sum(c[0] for c in coords) / len(coords)
            lon = sum(c[1] for c in coords) / len(coords)
            rows.append({
                "asset_id":   f"osm_{otype}_{oid}",
                "asset_type": "telecom.data_center",
                "lat":        lat,
                "lon":        lon,
                "name":       tags.get("name", ""),
                "source":     "osm_geofabrik",
            })

    # Relations: for each relation, average lat/lon of resolved member nodes
    for rel_id, members, tags in wc.matched_relations:
        node_member_refs = [ref for (mtype, ref) in members if mtype == 'n']
        coords = [ll.locs[nid] for nid in node_member_refs if nid in ll.locs]
        if not coords:
            print(f"  skipping relation/{rel_id}: no resolved node members", flush=True)
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)
        rows.append({
            "asset_id":   f"osm_relation_{rel_id}",
            "asset_type": "telecom.data_center",
            "lat":        lat,
            "lon":        lon,
            "name":       tags.get("name", ""),
            "source":     "osm_geofabrik",
        })

    df = pd.DataFrame(rows)
    out_pq = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets",
                          "central-america_all_assets_telecom.parquet")
    os.makedirs(os.path.dirname(out_pq), exist_ok=True)
    df.to_parquet(out_pq, index=False)
    print(f"\nWrote {len(df)} rows -> {out_pq}", flush=True)
    print("\nExtracted rows:", flush=True)
    for _, row in df.iterrows():
        # Show tags for the spot-check
        print(f"  {row['asset_id']}  lat={row['lat']:.5f}, lon={row['lon']:.5f}  "
              f"name={row['name']!r}", flush=True)

    # Print full tags
    print("\nFull tag dump (for spot-check):", flush=True)
    for entry in wc.matched_ways:
        otype, oid, node_ids, tags, _, _ = entry
        print(f"  {otype}/{oid}:", flush=True)
        print(f"    {json.dumps(tags, ensure_ascii=False)}", flush=True)
    for rel_id, members, tags in wc.matched_relations:
        print(f"  relation/{rel_id}:", flush=True)
        print(f"    {json.dumps(tags, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
