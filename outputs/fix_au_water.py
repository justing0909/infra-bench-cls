"""
Recover from the corrupted australia-oceania_water_only.osm.pbf (a 0-byte
file left over from a killed pre-filter, currently locked by the overnight
pipeline's leaked file handle).

We can't write to the standard path while it's locked, so this script
inlines the two-pass pre-filter logic (mirrors _pre_filter_pbf_by_tags)
to a different output path. After the overnight finishes and releases
the lock, the 0-byte file can be deleted manually.
"""
from __future__ import annotations
import os, sys, time
import osmium

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

import pandas as pd
from sources import GeoFabrikSource
from deduplication import Deduplicator

SRC      = r"D:\australia-oceania-260526.osm.pbf"
# Alternate output path so we don't fight with the locked 0-byte file.
WATER_PBF = r"D:\australia-oceania-260526.osm_water_only_v2.osm.pbf"
EXTRACT = os.path.join(REPO, "data", "PIPELINE", "01-extracted-assets",
                       "australia-oceania_all_assets_water.parquet")
DEDUP   = os.path.join(REPO, "data", "PIPELINE", "02-deduped-assets",
                       "australia-oceania_deduped_assets_water.parquet")

_TAG_KEYS = ("man_made",)


class IDCollector(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.tagged_node_ids: set = set()
        self.tagged_way_ids: set = set()
        self.tagged_relation_ids: set = set()
        self.member_node_ids: set = set()

    @staticmethod
    def _matches(tags) -> bool:
        for k in _TAG_KEYS:
            if k in tags:
                return True
        return False

    def node(self, n):
        if self._matches(n.tags):
            self.tagged_node_ids.add(n.id)

    def way(self, w):
        if self._matches(w.tags):
            self.tagged_way_ids.add(w.id)
            for nd in w.nodes:
                self.member_node_ids.add(nd.ref)

    def relation(self, r):
        if self._matches(r.tags):
            self.tagged_relation_ids.add(r.id)
            for m in r.members:
                if m.type == 'n':
                    self.member_node_ids.add(m.ref)


def make_writer_class(writer, keep_node_ids, way_ids, rel_ids):
    class FilterWriter(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.n_written = 0
            self._n_scanned = 0
        def node(self, n):
            self._n_scanned += 1
            if self._n_scanned % 5_000_000 == 0:
                print(f"    pass 2: {self._n_scanned/1e6:.0f}M scanned, "
                      f"{self.n_written:,} written")
            if n.id in keep_node_ids:
                writer.add_node(n)
                self.n_written += 1
        def way(self, w):
            if w.id in way_ids:
                writer.add_way(w)
                self.n_written += 1
        def relation(self, r):
            if r.id in rel_ids:
                writer.add_relation(r)
                self.n_written += 1
    return FilterWriter


if not os.path.exists(WATER_PBF):
    print(f"Pre-filtering {SRC} -> {WATER_PBF}")
    print("Pass 1: collect matched element IDs + member node refs...")
    t = time.time()
    collector = IDCollector()
    collector.apply_file(SRC)
    print(f"Pass 1 done in {time.time()-t:.0f}s — "
          f"{len(collector.tagged_node_ids):,} tagged nodes, "
          f"{len(collector.tagged_way_ids):,} ways, "
          f"{len(collector.tagged_relation_ids):,} relations")
    keep_node_ids = collector.tagged_node_ids | collector.member_node_ids
    print(f"Pass 2: write to {WATER_PBF}")
    t = time.time()
    writer = osmium.SimpleWriter(WATER_PBF)
    FW = make_writer_class(writer, keep_node_ids,
                           collector.tagged_way_ids,
                           collector.tagged_relation_ids)
    FW().apply_file(SRC)
    writer.close()
    print(f"Pass 2 done in {time.time()-t:.0f}s -> "
          f"{os.path.getsize(WATER_PBF)/1_048_576:.1f} MB")
else:
    print(f"Using existing {WATER_PBF} ({os.path.getsize(WATER_PBF)/1_048_576:.1f} MB)")

print("\nExtract water...")
t = time.time()
src2 = GeoFabrikSource(WATER_PBF, min_confidence="medium",
                       filter_preset="water", pre_filter=False)
df = src2.extract_all()
print(f"Extract done in {time.time()-t:.0f}s -> {len(df):,} rows")
print(df["asset_type"].value_counts().to_string())
df.drop(columns=["osm_tags"], errors="ignore").to_parquet(EXTRACT, index=False)
print(f"Saved -> {EXTRACT}")

print("\nDedup...")
t = time.time()
dedup = Deduplicator()
clean, _ = dedup.run(df)
print(f"Dedup done in {time.time()-t:.0f}s -> {len(clean):,} kept")
clean.drop(columns=["osm_tags"], errors="ignore").to_parquet(DEDUP, index=False)
print(f"Saved -> {DEDUP}")
