"""
subsample_to_1k.py
------------------
produce a 1k subsampled view of each completed v1 cell, without refetching
any imagery. reads dataset_<region>_<sector>_v1/{manifest.json,summary.csv,images/},
applies proportional class-stratified sampling (target=1000, seed=42, same
algorithm as curation/sectors/sample_v1.py), and writes:

  data/curated_datasets/dataset_<region>_<sector>_v1_1k/
    images/           hardlinks to the chosen .npy files in ../dataset_*_v1/images/
    manifest.json     filtered to the chosen records
    summary.csv       filtered to the chosen records
    _SUCCESS          completion marker

cells with <=TARGET tiles pass through unchanged (still hardlinked into the
new folder so consumers have a single uniform layout to point at).

idempotent -- re-run safely after additional cells finish.

Usage
-----
python -m curation.sectors.subsample_to_1k                 # every cell under data/curated_datasets/
python -m curation.sectors.subsample_to_1k --only africa energy
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from ..paths import REPO_ROOT, CURATED_DIR

REPO          = REPO_ROOT
DATASETS_DIR  = CURATED_DIR

V1_FOLDER_RE  = re.compile(r"^dataset_([a-z-]+)_(energy|water|transport|telecom)_v1$")
TARGET        = 1_000
SEED          = 42


def stratified_subsample(records: list[dict], target: int, seed: int) -> list[dict]:
    """proportional class-stratified subsample matching curation/sectors/sample_v1.py."""
    if len(records) <= target:
        return records
    df = pd.DataFrame(records)
    parts: list[pd.DataFrame] = []
    total_n = len(df)
    # each class is allocated its own rounded share, so the parts need not sum
    # to `target`: three of the 28 shipped cells land at 1,001 or 1,002. do not
    # trim the overshoot -- the published dataset was built with this exact
    # arithmetic and rounding it down would no longer reproduce those cells.
    for cls, cls_df in df.groupby("asset_type", sort=False):
        cls_n = len(cls_df)
        allocation = int(round(cls_n / total_n * target))
        take = min(allocation, cls_n)
        if take <= 0:
            continue
        parts.append(cls_df.sample(n=take, random_state=seed))
    if not parts:
        return []
    out = pd.concat(parts, ignore_index=True)
    return out.to_dict(orient="records")


def hardlink_image(src: Path, dst: Path) -> None:
    """create a Windows hardlink; fall back to copy if hardlink isn't possible."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def process_cell(src_dir: Path, target: int = TARGET, seed: int = SEED) -> dict:
    """subsample one completed v1 cell into a parallel _v1_1k folder."""
    m = V1_FOLDER_RE.match(src_dir.name)
    if not m:
        return {"cell": src_dir.name, "status": "skip_not_v1"}
    region, sector = m.group(1), m.group(2)
    tag = f"{region}/{sector}"

    src_success  = src_dir / "_SUCCESS"
    src_manifest = src_dir / "manifest.json"
    src_summary  = src_dir / "summary.csv"
    src_images   = src_dir / "images"
    if not (src_success.exists() and src_manifest.exists() and src_images.exists()):
        return {"cell": tag, "status": "skip_incomplete_source"}

    dst_dir       = src_dir.parent / f"dataset_{region}_{sector}_v1_1k"
    dst_images    = dst_dir / "images"
    dst_manifest  = dst_dir / "manifest.json"
    dst_summary   = dst_dir / "summary.csv"
    dst_success   = dst_dir / "_SUCCESS"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_images.mkdir(parents=True, exist_ok=True)

    with src_manifest.open() as f:
        m_obj = json.load(f)
    records = m_obj.get("records", [])
    n_src   = len(records)

    chosen = stratified_subsample(records, target=target, seed=seed)
    chosen_ids = {r["asset_id"] for r in chosen}

    # hardlink the chosen image files (and any temporal_file if present)
    n_linked = 0
    n_missing = 0
    for r in chosen:
        for fkey in ("image_file", "temporal_file"):
            fname = r.get(fkey)
            if not fname:
                continue
            s = src_images / fname
            if not s.exists():
                n_missing += 1
                continue
            d = dst_images / fname
            hardlink_image(s, d)
            n_linked += 1

    # write filtered manifest with updated top-level counts
    new_manifest = dict(m_obj)
    new_manifest["records"]  = chosen
    new_manifest["n_tiles"]  = len(chosen)
    # recompute asset_types/sources/modalities counts
    at_counts = {}
    for r in chosen:
        at_counts[r["asset_type"]] = at_counts.get(r["asset_type"], 0) + 1
    new_manifest["asset_types"] = at_counts
    new_manifest["subsample"]   = {
        "from_cell"  : src_dir.name,
        "target"     : target,
        "seed"       : seed,
        "n_source"   : n_src,
        "n_kept"     : len(chosen),
        "n_linked"   : n_linked,
        "n_missing"  : n_missing,
    }
    with dst_manifest.open("w") as f:
        json.dump(new_manifest, f, indent=2, default=str)

    # filter summary.csv if present
    if src_summary.exists():
        with src_summary.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row.get("asset_id") in chosen_ids]
            fieldnames = reader.fieldnames or []
        with dst_summary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # write _SUCCESS
    dst_success.write_text(json.dumps({
        "subsample_of"   : src_dir.name,
        "target"         : target,
        "seed"           : seed,
        "n_source"       : n_src,
        "n_kept"         : len(chosen),
        "n_image_links"  : n_linked,
        "n_missing_imgs" : n_missing,
    }, indent=2))

    return {
        "cell"      : tag,
        "status"    : "ok",
        "n_source"  : n_src,
        "n_kept"    : len(chosen),
        "n_linked"  : n_linked,
        "n_missing" : n_missing,
        "dst"       : str(dst_dir.relative_to(REPO)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", nargs=2, metavar=("REGION", "SECTOR"),
                    help="Process only this one cell")
    args = ap.parse_args()

    candidates = sorted(d for d in DATASETS_DIR.glob("dataset_*_v1")
                        if d.is_dir() and V1_FOLDER_RE.match(d.name))
    if args.only:
        region, sector = args.only
        candidates = [DATASETS_DIR / f"dataset_{region}_{sector}_v1"]

    if not candidates:
        sys.exit("No dataset_*_v1 folders found.")

    print(f"subsample_to_1k: target={args.target} seed={args.seed}")
    print(f"candidates: {len(candidates)}")
    print()
    print(f"{'cell':<30} {'status':<22} {'source':>9} {'kept':>6} {'linked':>7} {'miss':>5}")
    print("-" * 92)
    for src in candidates:
        r = process_cell(src, target=args.target, seed=args.seed)
        if r["status"] == "ok":
            print(f"{r['cell']:<30} {r['status']:<22} {r['n_source']:>9,} "
                  f"{r['n_kept']:>6,} {r['n_linked']:>7,} {r['n_missing']:>5}")
        else:
            print(f"{r['cell']:<30} {r['status']:<22}")


if __name__ == "__main__":
    main()
