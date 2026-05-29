"""
Check whether the idx parameter also affected the substation extract baseline
(1,920 from May 26). Substations match in way() and area() (closed ways),
not relying solely on multipolygon-relation assembly, so impact may differ.
"""
from __future__ import annotations
import os, sys, time

REPO = r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm"
PATH = (
    r"C:\Users\j.guthrie\Downloads\RESEARCH\infra_fm\data\pbf\power_only"
    r"\central-america-latest.osm_power_only.osm.pbf"
)
sys.path.insert(0, os.path.join(REPO, "curation"))
sys.path.insert(0, os.path.join(REPO, "curation", "utils"))
os.chdir(REPO)

from sources import InfraHandler
from sources import FILTER_PRESETS

classes = FILTER_PRESETS["substation"]
print(f"Substation classes: {[c.name for c in classes]}")
print()

print("Run A: apply_file(PATH, locations=True, idx='sparse_file_array,locations.idx')")
hA = InfraHandler(active_classes=classes)
t = time.time()
hA.apply_file(PATH, locations=True, idx="sparse_file_array,locations.idx")
print(f"  elapsed={time.time()-t:.1f}s  matched={len(hA.rows):,}  areas={hA._n_areas:,}")
print()

print("Run B: apply_file(PATH, locations=True)  [in-memory]")
hB = InfraHandler(active_classes=classes)
t = time.time()
hB.apply_file(PATH, locations=True)
print(f"  elapsed={time.time()-t:.1f}s  matched={len(hB.rows):,}  areas={hB._n_areas:,}")
print()

import pandas as pd
dfA = pd.DataFrame(hA.rows); dfB = pd.DataFrame(hB.rows)
print(f"Total:    A={len(dfA):,}    B={len(dfB):,}    delta={len(dfB)-len(dfA):+d}")
print("\nBy class:")
print("  A:", dfA["asset_type"].value_counts().to_dict() if len(dfA) else "(empty)")
print("  B:", dfB["asset_type"].value_counts().to_dict() if len(dfB) else "(empty)")
