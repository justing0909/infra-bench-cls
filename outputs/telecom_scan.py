"""Quick lightweight count of building=data_center in central-america source PBF."""
import osmium, time, collections
PATH = r"D:\central-america-260526.osm.pbf"

class Counter(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.node_dc = 0
        self.way_dc  = 0
        self.rel_dc  = 0
        self.node_seen = 0
        self.way_seen  = 0
    def node(self, n):
        self.node_seen += 1
        if n.tags.get("building") == "data_center":
            self.node_dc += 1
    def way(self, w):
        self.way_seen += 1
        if w.tags.get("building") == "data_center":
            self.way_dc += 1
    def relation(self, r):
        if r.tags.get("building") == "data_center":
            self.rel_dc += 1

t = time.time()
c = Counter()
c.apply_file(PATH)  # no locations=True — much faster, sufficient for counting
elapsed = time.time() - t
print(f"Scanned {PATH} in {elapsed:.1f}s")
print(f"  nodes scanned    : {c.node_seen:,}")
print(f"  ways scanned     : {c.way_seen:,}")
print(f"  building=data_center nodes     : {c.node_dc}")
print(f"  building=data_center ways      : {c.way_dc}")
print(f"  building=data_center relations : {c.rel_dc}")
print(f"  total data_center features     : {c.node_dc + c.way_dc + c.rel_dc}")
