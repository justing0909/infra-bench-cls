"""
finish_from_checkpoint.py
-------------------------
Take a partially-fetched cell's checkpoint pickle and run the rest of the
pipeline (QC -> triage -> DatasetAssembler) on whatever's already there.
Useful when a fetch was paused/interrupted and we don't want to refetch
the remaining assets (e.g. a v1 cell that overshot the new 1k target).

Writes:
  data/curated_datasets/dataset_<region>_<sector>_v1/
    images/, manifest.json, summary.csv, _SUCCESS

Usage
-----
python finish_from_checkpoint.py --region africa --sector energy
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

REPO        = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "curation"))

from curation.qc      import QualityChecker
from curation.triage  import RuleBasedTriager
from curation.dataset import DatasetAssembler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True)
    ap.add_argument("--sector", required=True)
    ap.add_argument("--checkpoint", type=Path,
                    help="Override checkpoint path "
                         "(default: data/checkpoints/v1/<region>_<sector>_fetch.pkl)")
    ap.add_argument("--output-dir", type=Path,
                    help="Override output dir "
                         "(default: data/curated_datasets/dataset_<region>_<sector>_v1)")
    ap.add_argument("--qc-min-valid", type=float, default=0.80)
    ap.add_argument("--contradiction-threshold", type=int, default=3)
    ap.add_argument("--low-threshold", type=int, default=4)
    args = ap.parse_args()

    ckpt_path = args.checkpoint or (REPO / "data" / "checkpoints" / "v1" /
                                    f"{args.region}_{args.sector}_fetch.pkl")
    out_dir   = args.output_dir or (REPO / "data" / "curated_datasets" /
                                    f"dataset_{args.region}_{args.sector}_v1")

    if not ckpt_path.exists():
        sys.exit(f"ERROR: no checkpoint at {ckpt_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint: {ckpt_path}")
    with ckpt_path.open("rb") as f:
        data = pickle.load(f)
    results = data.get("results", [])
    n_total = len(results)
    n_ok    = sum(1 for r in results if getattr(r, "status", None) == "ok")
    print(f"  total TileResults: {n_total:,}")
    print(f"  status=ok:         {n_ok:,}")

    ok_tiles = [r for r in results if getattr(r, "status", None) == "ok"]
    if not ok_tiles:
        sys.exit("no ok tiles in checkpoint — nothing to assemble")

    t0 = time.time()

    print(f"[QC]     QualityChecker(min_valid_ratio={args.qc_min_valid})")
    checker    = QualityChecker(min_valid_ratio=args.qc_min_valid)
    qc_results = checker.check_all(ok_tiles, max_workers=4)
    clean      = checker.filter_ok(qc_results)
    print(f"         QC passed: {len(clean):,} / {len(ok_tiles):,}")

    print(f"[triage] contradiction_threshold={args.contradiction_threshold} "
          f"low_threshold={args.low_threshold}")
    triager        = RuleBasedTriager(
        contradiction_threshold=args.contradiction_threshold,
        low_threshold=args.low_threshold)
    triage_results = triager.triage_all(clean, max_workers=4)
    accepted       = triager.filter_accepted(triage_results)
    print(f"         accepted: {len(accepted):,}")

    if not accepted:
        sys.exit("no tiles accepted after triage")

    print(f"[assemble] -> {out_dir}")
    assembler = DatasetAssembler(output_dir=str(out_dir))
    summary   = assembler.assemble(accepted_tiles=accepted,
                                   triage_results=triage_results)
    n_tiles   = len(summary) if summary is not None else len(accepted)

    elapsed = round(time.time() - t0, 1)
    success = out_dir / "_SUCCESS"
    success.write_text(json.dumps({
        "completed_at"     : datetime.utcnow().isoformat() + "Z",
        "region"           : args.region,
        "sector"           : args.sector,
        "modalities"       : ["sentinel2_ms", "sentinel1"],
        "buffer_m"         : 300,
        "temporal_stack"   : False,
        "from_checkpoint"  : str(ckpt_path),
        "n_checkpoint_ok"  : n_ok,
        "n_qc_passed"      : len(clean),
        "n_accepted"       : len(accepted),
        "n_dataset_tiles"  : int(n_tiles),
        "elapsed_s"        : elapsed,
        "note"             : "assembled from existing checkpoint without further fetching",
    }, indent=2))

    print(f"[done] {args.region}/{args.sector} in {elapsed:.0f}s — "
          f"{n_tiles:,} tiles assembled")


if __name__ == "__main__":
    main()
