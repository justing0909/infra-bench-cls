"""
Diagnostic: does idx="sparse_file_array,locations.idx" cause area assembly
to be incomplete? Compare apply_file with and without the idx kwarg using
the same InfraHandler matcher logic.
"""
from __future__ import annotations
import os, sys, time

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
PATH = r"D:\central-america-260526.osm_transport_only.osm.pbf"

sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

from sources import InfraHandler
from ontology import get_classes_for_sector

classes = get_classes_for_sector("transport")
print(f"Active transport classes: {[c.name for c in classes]}")
print()

# Run A: with idx
print("Run A: apply_file(PATH, locations=True, idx='sparse_file_array,locations.idx')")
hA = InfraHandler(active_classes=classes)
t = time.time()
hA.apply_file(PATH, locations=True, idx="sparse_file_array,locations.idx")
print(
    f"  elapsed={time.time()-t:.1f}s  "
    f"nodes={hA._n_nodes:,} ways={hA._n_ways:,} areas={hA._n_areas:,} "
    f"matched={len(hA.rows)}"
)
print()

# Cleanup the on-disk index file if it was created
for f in ("locations.idx", "central-america-260526.osm_transport_only.locations.idx"):
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"removed leftover index file: {f}")
        except Exception:
            pass

# Run B: in-memory locations
print("Run B: apply_file(PATH, locations=True)  [in-memory locations]")
hB = InfraHandler(active_classes=classes)
t = time.time()
hB.apply_file(PATH, locations=True)
print(
    f"  elapsed={time.time()-t:.1f}s  "
    f"nodes={hB._n_nodes:,} ways={hB._n_ways:,} areas={hB._n_areas:,} "
    f"matched={len(hB.rows)}"
)
print()

# Compare
print("=" * 60)
print(f"areas:    A={hA._n_areas:>5d}    B={hB._n_areas:>5d}    delta={hB._n_areas-hA._n_areas:+d}")
print(f"matched:  A={len(hA.rows):>5d}    B={len(hB.rows):>5d}    delta={len(hB.rows)-len(hA.rows):+d}")
print()
# By class
import pandas as pd
dfA = pd.DataFrame(hA.rows)
dfB = pd.DataFrame(hB.rows)
print("By class:")
if len(dfA): print("  A:", dfA["asset_type"].value_counts().to_dict())
else:        print("  A: (no matches)")
if len(dfB): print("  B:", dfB["asset_type"].value_counts().to_dict())
else:        print("  B: (no matches)")
