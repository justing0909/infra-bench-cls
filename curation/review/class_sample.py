"""
class_sample.py
---------------
A random sample of every class, to estimate how accurate the OSM labels are
class by class.

The auto-accept audit (auto_accept.py) measures the strict source rules, and
those mostly confirm stations and airports. This measures the labels
themselves: for each class it draws SHARE of its tiles at random, at most
MAX_PER_CLASS and at least MIN_PER_CLASS (or the whole class when it is
smaller), and leaves every one to a person. The draw is uniform over the
whole class, including tiles already auto-accepted or reviewed, so the
accuracy it gives is an unbiased estimate for that class; a tile you have
already decided counts with that decision.

Each sampled tile carries hints that make the call quicker without making it:
the asset's current OSM tags (voltage, substation type, operator, plant
source, and the tags that define the other classes) fetched from Overpass,
and its mapped footprint size beside the class median.

    python -m curation.review.class_sample            # draw (once) and fetch hints
    python -m curation.review.class_sample --report   # accuracy per class, 95% CI

The draw is kept in data/label_review/class_sample.csv and never redrawn, so
rerunning only refreshes hints.
"""

import argparse
import json
import math
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import beta

from ..paths import DATA_DIR, REVIEW_DIR, REVIEW_QUEUE
from . import decisions

SHARE = 0.10
MAX_PER_CLASS = 100
MIN_PER_CLASS = 20
SEED = 2026
# Early stopping. The sample pass walks each class in draw order, so the tiles
# reviewed so far are a random subset and a class can stop once it has at
# least STOP_AFTER decisions and its 95% interval is at most STOP_WIDTH points
# wide (30 tiles with no errors gives 88-100%). A class where errors turn up
# keeps going until its interval tightens or its sample runs out.
STOP_AFTER = 30
STOP_WIDTH = 12
REVIEWER = "justin"   # whose decisions the report reads; override with --reviewer
SAMPLE = REVIEW_DIR / "class_sample.csv"
CACHE_DIR = REVIEW_DIR / "sources" / "class_sample"
FOOTPRINTS = DATA_DIR / "asset_footprints" / "asset_footprint_share_v1.csv"
OVERPASS = "https://overpass-api.de/api/interpreter"
TAG_KEYS = ["power", "substation", "voltage", "operator", "plant:source", "generator:source",
            "plant:output:electricity", "railway", "public_transport", "station", "aeroway",
            "aerodrome:type", "iata", "man_made", "content", "water_works", "wastewater",
            "harbour", "industrial", "telecom", "building", "landuse", "disused", "abandoned"]


def sample_size(n: int) -> int:
    return min(n, MAX_PER_CLASS, max(MIN_PER_CLASS, math.ceil(SHARE * n)))


def draw() -> pd.DataFrame:
    if SAMPLE.exists():
        return pd.read_csv(SAMPLE, keep_default_na=False)
    q = pd.read_csv(REVIEW_QUEUE, keep_default_na=False)
    rows = []
    for k, cls in enumerate(sorted(q.asset_type.unique())):
        ids = np.sort(q.loc[q.asset_type == cls, "asset_id"].to_numpy())
        rng = np.random.default_rng([SEED, k])
        pick = rng.choice(ids, size=sample_size(len(ids)), replace=False)
        rows += [{"asset_id": a, "asset_type": cls, "class_size": len(ids), "draw": i}
                 for i, a in enumerate(pick)]
    df = pd.DataFrame(rows)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE, index=False)
    return df


def _osm_tags(ids: list) -> dict:
    """asset_id -> current OSM tags, one Overpass request per element type."""
    path = CACHE_DIR / "tags.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    parts = pd.Series(ids).str.extract(r"osm_(node|way|relation)_(\d+)")
    out = {}
    for kind, g in parts.groupby(0):
        query = f"[out:json][timeout:180];{kind}(id:{','.join(g[1])});out tags;"
        req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": query}).encode(),
                                     headers={"User-Agent": "infra-bench-cls label sample"})
        with urllib.request.urlopen(req, timeout=240) as r:
            for el in json.load(r)["elements"]:
                out[f"osm_{el['type']}_{el['id']}"] = el.get("tags", {})
        time.sleep(2)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out), encoding="utf-8")
    return out


def _max_kv(voltage: str):
    kv = [float(v) / 1000 for v in str(voltage).replace(",", ";").split(";") if v.strip().isdigit()]
    return max(kv) if kv else None


def hints(sample: pd.DataFrame) -> pd.DataFrame:
    tags = _osm_tags(sample.asset_id.tolist())
    fp = pd.read_csv(FOOTPRINTS, keep_default_na=False) if FOOTPRINTS.exists() else pd.DataFrame()
    medians = {}
    if len(fp):
        poly = fp[fp.status == "polygon"].copy()
        medians = poly.assign(extent_m=poly.extent_m.astype(float)).groupby("asset_type").extent_m.median().to_dict()
        fp = fp.set_index("asset_id")
    out = []
    for r in sample.itertuples():
        t = tags.get(r.asset_id)
        lines = []
        if t is None:
            lines.append("OSM now: this element no longer exists")
        else:
            shown = [f"{k}={t[k]}" for k in TAG_KEYS if k in t]
            kv = _max_kv(t.get("voltage", ""))
            lines.append("OSM tags: " + (", ".join(shown) if shown else "none of the defining tags")
                         + (f" (max {kv:g} kV)" if kv else ""))
        if len(fp) and r.asset_id in fp.index:
            f = fp.loc[r.asset_id]
            if f.status == "polygon":
                ref = f"median for this class {medians.get(r.asset_type, float('nan')):.0f} m"
                if "substation" in r.asset_type or r.asset_type == "energy.distribution.other":
                    ref = (f"medians: transmission {medians['energy.transmission.substation']:.0f} m, "
                           f"distribution {medians['energy.distribution.substation']:.0f} m")
                lines.append(f"Footprint {float(f.extent_m):.0f} m across ({ref})")
            elif f.status == "point":
                lines.append("Mapped as a point in OSM: no footprint")
        out.append({"asset_id": r.asset_id, "hint": " | ".join(lines)})
    return sample.drop(columns=["hint"], errors="ignore").merge(pd.DataFrame(out), on="asset_id")


def _ci(k: int, n: int) -> tuple:
    """Clopper-Pearson 95% interval for k successes in n."""
    if n == 0:
        return (np.nan, np.nan)
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def report(reviewer: str = REVIEWER) -> tuple:
    sample = pd.read_csv(SAMPLE, keep_default_na=False)
    latest = decisions.load()
    rows = []
    for r in sample.itertuples():
        e = latest.get(r.asset_id, {}).get(reviewer)
        if e:
            rows.append({"asset_type": r.asset_type, "decision": e["decision"], "label": e["label"],
                         "correct": e["decision"] == "accept" or e["label"] == r.asset_type})
    df = pd.DataFrame(rows, columns=["asset_type", "decision", "label", "correct"])
    decided = df[df.decision != "unsure"]
    summary = []
    for cls, g in sample.groupby("asset_type"):
        d = decided[decided.asset_type == cls]
        k, n = int(d.correct.sum()), len(d)
        lo, hi = _ci(k, n)
        reviewed = int((df.asset_type == cls).sum())
        # stop a class once its interval is tight enough, or its sample is done
        enough = reviewed >= len(g) or (n >= min(STOP_AFTER, len(g)) and 100 * (hi - lo) <= STOP_WIDTH)
        summary.append({"class": cls, "class_size": int(g.class_size.iloc[0]), "sampled": len(g),
                        "reviewed": reviewed,
                        "unsure": int(((df.asset_type == cls) & (df.decision == "unsure")).sum()),
                        "correct": k, "accuracy_%": round(100 * k / n, 1) if n else None,
                        "ci95_%": f"{100 * lo:.0f}-{100 * hi:.0f}" if n else "",
                        "next": "enough" if enough else "keep going"})
    wrong = decided[decided.correct.eq(False)]
    confusion = (wrong.groupby(["asset_type", "label"]).size().rename("tiles").reset_index()
                 if len(wrong) else pd.DataFrame(columns=["asset_type", "label", "tiles"]))
    return pd.DataFrame(summary), confusion


def main():
    p = argparse.ArgumentParser(description="random per-class sample for OSM label accuracy")
    p.add_argument("--report", action="store_true")
    p.add_argument("--reviewer", default=REVIEWER)
    args = p.parse_args()
    if args.report:
        summary, confusion = report(args.reviewer)
        print(summary.to_string(index=False))
        if len(confusion):
            print("\nwhat wrong labels should have been:")
            print(confusion.to_string(index=False))
        return
    sample = hints(draw())
    sample.to_csv(SAMPLE, index=False)
    print(f"{len(sample)} tiles in {SAMPLE}")
    print(sample.groupby("asset_type").size().to_string())


if __name__ == "__main__":
    main()
