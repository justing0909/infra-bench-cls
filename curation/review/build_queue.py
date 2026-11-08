"""
build_queue.py
--------------
Builds data/label_review/queue.csv: one row per v1 tile, with the OSM label,
where its image lives, and two weak hints that order the review within each
class. Every tile still gets a human decision; the hints only decide which
tiles a reviewer sees first.

Hints
  model_suggests / model_p_label
      out-of-fold logistic regression on the 64-D AlphaEarth embeddings,
      with folds grouped by spatial block so a tile is never scored by a model
      that saw its neighbours. AlphaEarth reaches ~0.27 macro F1 on this task,
      so its disagreements are mostly its own mistakes. Treat it as a nudge.
  name_hint
      the OSM name contains a keyword that points at a different class.

The queue is deterministic (fixed seed, stable sort), so rebuilding it gives
everyone the same order. Run:

    python -m curation.review.build_queue
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..paths import (
    ALPHAEARTH_EMBEDDINGS, CURATED_DIR, DEDUPED_DIR, REVIEW_DIR, REVIEW_QUEUE,
    SPLIT_ARTIFACT,
)

N_FOLDS = 5

# keyword in the OSM name -> the class it points at. deliberately short: a
# hint that fires on half a class is noise, not a hint.
NAME_HINTS = [
    # solar before power plant: "centrale solaire" is a solar farm.
    (r"(?i)\b(solar|photovolta)", "energy.generation.solar_farm"),
    (r"(?i)\b(power station|power plant|generating station|centrale)\b", "energy.generation.power_plant"),
    (r"(?i)\b(substation|switchyard|umspannwerk|subestaci[oó]n|poste source)\b", "energy.*.substation"),
    (r"(?i)(sewage|waste ?water|wwtp|effluent|kl[äa]ranlage|depuradora|[ée]puration|rioolwaterzuivering)", "water.wastewater.plant"),
    (r"(?i)\b(water ?works|drinking|potable|water treatment)\b", "water.treatment.plant"),
    (r"(?i)\b(data ?cent(er|re)|datacenter)\b", "telecom.data_center"),
    (r"(?i)\b(airport|aerodrome|airfield|aeropuerto|a[ée]roport)\b", "transport.airport"),
    # no train-station keyword: "station" names substations and pumping
    # stations as often as railway stations.
]


def _name_hint(name: str, label: str) -> str:
    """first keyword class that disagrees with the label, else ''."""
    if not name:
        return ""
    # "waste water treatment" matches both water patterns; the sewage one wins.
    for pattern, cls in NAME_HINTS:
        if re.search(pattern, name):
            if cls == "energy.*.substation":
                return "" if "substation" in label or label == "energy.distribution.other" else "a substation"
            return "" if cls == label else cls
    return ""


def _image_index() -> dict:
    """(asset_id, dataset folder) -> image path, from every v1_1k manifest."""
    index = {}
    for manifest in glob.glob(str(CURATED_DIR / "*_v1_1k" / "manifest.json")):
        folder = os.path.basename(os.path.dirname(manifest))
        with open(manifest, encoding="utf-8") as f:
            for r in json.load(f)["records"]:
                # forward slashes so the committed queue works on any OS.
                index[(r["asset_id"], folder)] = f"{folder}/images/{r['image_file']}"
    return index


def _out_of_fold_hints(split: pd.DataFrame) -> pd.DataFrame:
    emb = pd.read_parquet(ALPHAEARTH_EMBEDDINGS)
    cols = [c for c in emb.columns if re.fullmatch(r"A\d\d", c)]
    df = split[["asset_id", "asset_type", "block_id_x", "block_id_y"]].merge(
        emb[["asset_id"] + cols], on="asset_id", how="inner")
    # a tile with no embedding still gets reviewed, it just carries no model hint.
    df = df[df[cols].notna().all(axis=1)].reset_index(drop=True)

    X = df[cols].to_numpy()
    classes = np.array(sorted(df.asset_type.unique()))
    y = np.searchsorted(classes, df.asset_type.to_numpy())
    groups = df.block_id_x.astype(str) + "_" + df.block_id_y.astype(str)

    proba = np.zeros((len(df), len(classes)))
    for train, test in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
        model.fit(X[train], y[train])
        # a fold can miss a rare class entirely; place the columns it has.
        proba[np.ix_(test, model.classes_)] = model.predict_proba(X[test])

    top = proba.argmax(1)
    return pd.DataFrame({
        "asset_id":       df.asset_id,
        "model_suggests": classes[top],
        "model_p_top":    proba[np.arange(len(df)), top].round(3),
        "model_p_label":  proba[np.arange(len(df)), y].round(3),
    })


def build() -> pd.DataFrame:
    split = pd.read_parquet(SPLIT_ARTIFACT)

    names = pd.concat(
        pd.read_parquet(f, columns=["asset_id", "name"])
        for f in glob.glob(str(DEDUPED_DIR / "*.parquet"))
    ).drop_duplicates("asset_id")
    q = split.merge(names, on="asset_id", how="left")
    q["name"] = q["name"].fillna("").astype(str).str.strip()

    index = _image_index()
    q["image"] = [index.get((a, f"dataset_{r}_{s}_v1_1k"), "")
                  for a, r, s in zip(q.asset_id, q.region, q.sector)]
    missing = (q.image == "").sum()
    if missing:
        raise ValueError(f"{missing} split assets have no tile under {CURATED_DIR}")

    q = q.merge(_out_of_fold_hints(split), on="asset_id", how="left")
    q["model_suggests"] = q.model_suggests.fillna("")
    q["name_hint"] = [_name_hint(n, t) for n, t in zip(q.name, q.asset_type)]

    # secondary sources, when they have been fetched (see hifld_lines.py).
    q["evidence"], q["corroborated"] = "", False
    for path, text_col in [(REVIEW_DIR / "sources" / "hifld_lines.csv", "hifld_text")]:
        if path.exists():
            ev = pd.read_csv(path, keep_default_na=False)[["asset_id", text_col, "corroborated"]]
            q = q.merge(ev.rename(columns={text_col: "_text", "corroborated": "_corr"}),
                        on="asset_id", how="left")
            q["evidence"] = q.evidence.str.cat(q._text.fillna(""), sep=" | ").str.strip(" |")
            q["corroborated"] = q.corroborated | (q._corr.astype(str) == "True")
            q = q.drop(columns=["_text", "_corr"])

    # within a class: name hints first, then the tiles the model doubts most.
    doubt = (1 - q.model_p_label).fillna(0)
    q["priority"] = ((q.name_hint != "").astype(float) + doubt).round(3)
    q = q.sort_values(["asset_type", "priority", "asset_id"],
                      ascending=[True, False, True], kind="mergesort")
    q.insert(0, "order", range(len(q)))

    osm_type, osm_id = q.asset_id.str.extract(r"osm_(node|way|relation)_(\d+)").T.values
    q["osm_url"] = "https://www.openstreetmap.org/" + osm_type + "/" + osm_id

    return q[["order", "asset_id", "asset_type", "region", "sector", "split",
              "lat", "lon", "name", "name_hint", "model_suggests", "model_p_top",
              "model_p_label", "priority", "evidence", "corroborated", "osm_url", "image"]]


def main():
    q = build()
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    q.to_csv(REVIEW_QUEUE, index=False)
    print(f"wrote {len(q):,} tiles to {REVIEW_QUEUE}")
    print(f"  name hints:            {(q.name_hint != '').sum():,}")
    print(f"  model disagrees:       {(q.model_suggests != q.asset_type).sum():,}")
    print(f"  with source evidence:  {(q.evidence != '').sum():,}")
    print(f"  source-confirmed:      {q.corroborated.sum():,}")
    print(q.groupby("asset_type").size().to_string())


if __name__ == "__main__":
    main()
