"""
via.py
------
Moves review batches in and out of VIA (VGG Image Annotator 2.x), the same
tool Ed uses for the tower labels: one chip per tile, and the reviewer marks
whether the OSM label is right.

    # chips + VIA projects for everything this reviewer should look at next
    python -m curation.review.via export --reviewer justin

    # after saving the project from VIA, record the decisions
    python -m curation.review.via import --reviewer justin data/label_review/via/justin_first_*.json

Export writes one project per class per pass, in chunks of --batch tiles, to
data/label_review/via/. Each image carries file attributes:

    verdict        true / false / unconfirmed   (true = the OSM label is right)
    correct_class  the right class, only needed when verdict is false
    osm_label, name, hints, google_maps, street_view, note

Nothing is pre-selected, so every tile needs a deliberate click. Import turns
true into accept, false + correct_class into relabel, and anything else into
unsure, then appends to this reviewer's decisions log (see decisions.py).
Unsure tiles come back in a "revisit" project on the next export; with
--second-opinion they go to another reviewer instead.

Chips and VIA projects are regenerable and stay out of git; the decisions
logs are the record.
"""

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ..paths import CURATED_DIR, REVIEW_DIR, REVIEW_QUEUE
from . import decisions

CHIP_DIR = REVIEW_DIR / "chips"
VIA_DIR = REVIEW_DIR / "via"
CHIP_PX = 240
# 9-band stacks: B04 B03 B02 first, see SOURCE_BANDS in stac_imagery.py
RGB = [0, 1, 2]
NOT_INFRA = "not_infrastructure"
VERDICTS = ("true", "false", "unconfirmed")


def render_chip(npy: Path) -> bytes:
    a = np.load(npy)[RGB].astype(np.float32)
    lo, hi = np.nanpercentile(a, [2, 98])
    rgb = np.nan_to_num(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1))
    img = Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8))
    buf = io.BytesIO()
    # nearest-neighbour so a 10 m pixel stays a visible block, not a blur.
    img.resize((CHIP_PX, CHIP_PX), Image.NEAREST).save(buf, "PNG")
    return buf.getvalue()


def chip_for(row: dict) -> Path:
    path = CHIP_DIR / f"{row['asset_id']}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(render_chip(CURATED_DIR / row["image"]))
    return path


def _hints(row: dict) -> str:
    out = [row["evidence"]] if row.get("evidence") else []
    if row["name_hint"]:
        out.append(f"name suggests {row['name_hint']}")
    if row["model_suggests"] and row["model_suggests"] != row["asset_type"]:
        out.append(f"weak model suggests {row['model_suggests']} ({float(row['model_p_top']):.2f})")
    return "; ".join(out)


def _text(desc: str) -> dict:
    return {"type": "text", "description": desc, "default_value": ""}


def via_project(name: str, rows: list, classes: list, image_root: str) -> dict:
    attrs = {
        "verdict": {"type": "radio",
                    "description": "Is the OSM label right? true / false / unconfirmed",
                    "options": {v: "" for v in VERDICTS}, "default_options": {}},
        "correct_class": {"type": "dropdown",
                          "description": "Only when verdict is false",
                          "options": {c: "" for c in classes}, "default_options": {}},
        "osm_label": _text("OSM label under review"),
        "name": _text("OSM name"),
        "hints": _text("Weak hints, not evidence"),
        "google_maps": _text("Satellite view"),
        "street_view": _text("Street View at the asset point"),
        "lat": _text(""), "lon": _text(""),
        "note": _text("Optional"),
    }
    meta, ids = {}, []
    for row in rows:
        chip = chip_for(row)
        size = chip.stat().st_size
        key = f"{chip.name}{size}"
        lat, lon = float(row["lat"]), float(row["lon"])
        meta[key] = {
            "filename": chip.name, "size": size, "regions": [],
            "file_attributes": {
                "osm_label": row["asset_type"], "name": row["name"], "hints": _hints(row),
                "google_maps": f"https://www.google.com/maps/@?api=1&map_action=map&center={lat:.6f},{lon:.6f}&zoom=18&basemap=satellite",
                "street_view": f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat:.6f},{lon:.6f}",
                "lat": f"{lat:.6f}", "lon": f"{lon:.6f}",
            },
        }
        ids.append(key)
    return {
        "_via_settings": {
            "ui": {"annotation_editor_height": 25, "annotation_editor_fontsize": 0.8,
                   "leftsidebar_width": 18,
                   "image_grid": {"img_height": 80, "rshape_fill": "none",
                                  "rshape_fill_opacity": 0.3, "rshape_stroke": "yellow",
                                  "rshape_stroke_width": 2, "show_region_shape": True,
                                  "show_image_policy": "all"},
                   "image": {"region_label": "__via_region_id__",
                             "region_color": "__via_default_region_color__",
                             "region_label_font": "10px Sans",
                             "on_image_annotation_editor_placement": "NEAR_REGION"}},
            "core": {"buffer_size": 18, "filepath": {}, "default_filepath": image_root},
            "project": {"name": name},
        },
        "_via_img_metadata": meta,
        "_via_attributes": {"region": {}, "file": attrs},
        "_via_data_format_version": "2.0.10",
        "_via_image_id_list": ids,
    }


def export(reviewer: str, batch: int, image_root: str, classes_filter: list,
           second_opinion: bool = False) -> list:
    queue = pd.read_csv(REVIEW_QUEUE, keep_default_na=False)
    classes = sorted(queue.asset_type.unique()) + [NOT_INFRA]
    latest = decisions.load()
    queue["pass"] = [decisions.wants(reviewer, latest.get(a, {}), second_opinion)
                     for a in queue.asset_id]
    todo = queue[queue["pass"] != ""]
    if classes_filter:
        todo = todo[todo.asset_type.isin(classes_filter)]

    VIA_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for (pass_name, cls), group in todo.groupby(["pass", "asset_type"], sort=False):
        rows = group.sort_values("order").to_dict("records")
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            name = f"{reviewer}_{pass_name}_{cls}_{start // batch + 1:03d}"
            path = VIA_DIR / f"{name}.json"
            path.write_text(json.dumps(via_project(name, chunk, classes, image_root),
                                       ensure_ascii=False), encoding="utf-8")
            written.append((path, len(chunk)))
    return written


def to_decision(attrs: dict) -> tuple | None:
    """VIA file attributes -> (decision, label, note), or None if untouched."""
    verdict = attrs.get("verdict", "")
    note = attrs.get("note", "").strip()
    label = attrs.get("correct_class", "")
    if verdict == "true":
        return "accept", attrs.get("osm_label", ""), note
    if verdict == "false" and label:
        return "relabel", label, note
    if verdict == "false":
        return "unsure", "", ("marked false with no correct class. " + note).strip()
    if verdict == "unconfirmed":
        return "unsure", "", note
    return None


def import_(reviewer: str, paths: list) -> dict:
    queue = pd.read_csv(REVIEW_QUEUE, keep_default_na=False)
    labels = dict(zip(queue.asset_id, queue.asset_type))
    latest = decisions.load()
    counts = {"recorded": 0, "unchanged": 0, "untouched": 0}
    for path in paths:
        project = json.loads(Path(path).read_text(encoding="utf-8"))
        for meta in project["_via_img_metadata"].values():
            asset_id = Path(meta["filename"]).stem
            if asset_id not in labels:
                raise ValueError(f"{path}: {meta['filename']} is not in the queue")
            d = to_decision(meta.get("file_attributes", {}))
            if d is None:
                counts["untouched"] += 1
                continue
            decision, label, note = d
            if decision == "accept":
                label = labels[asset_id]  # never trust an edited osm_label field
            prev = latest.get(asset_id, {}).get(reviewer)
            if prev and (prev["decision"], prev["label"], prev["note"]) == (decision, label, note):
                counts["unchanged"] += 1
                continue
            latest.setdefault(asset_id, {})[reviewer] = decisions.append(
                reviewer, asset_id, decision, label, note)
            counts["recorded"] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description="VIA export/import for label review")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--reviewer", required=True)
    e.add_argument("--batch", type=int, default=500)
    e.add_argument("--class", dest="classes", action="append", default=[],
                   help="limit to one class; repeatable")
    e.add_argument("--image-root", default=str(CHIP_DIR) + os.sep,
                   help="folder VIA loads chips from, as seen on the reviewer's machine")
    e.add_argument("--second-opinion", action="store_true",
                   help="two-person rule: relabels and unsure tiles go to another reviewer")
    i = sub.add_parser("import")
    i.add_argument("--reviewer", required=True)
    i.add_argument("projects", nargs="+")
    args = p.parse_args()

    decisions.log_path(args.reviewer)  # validate the name up front
    if args.cmd == "export":
        written = export(args.reviewer, args.batch, args.image_root, args.classes,
                         args.second_opinion)
        for path, n in written:
            print(f"{n:>5}  {path}")
        print(f"{sum(n for _, n in written):,} tiles in {len(written)} projects")
    else:
        print(import_(args.reviewer, args.projects))


if __name__ == "__main__":
    main()
