"""
align_checkpoint_to_parquet.py
------------------------------
Trim a fetcher checkpoint so it only contains entries for asset_ids that
are present in a target parquet — and drop any non-ok results so they
get retried by the fetcher.

Use case: a checkpoint was built against an old (10k) v1 sample parquet,
and we just regenerated the parquet at target=1000 with different
asset_ids. Without this trim, fetch_all would carry forward the entire
10k checkpoint and the assembler would process every checkpointed tile
instead of just the 1k we now want.

Writes a backup at <ckpt>.before_align-<ts>.pkl, then atomically replaces
the checkpoint with the aligned version.

Usage
-----
python align_checkpoint_to_parquet.py \
    --checkpoint data/checkpoints/v1/africa_energy_fetch.pkl \
    --parquet    data/PIPELINE/03-v1-samples/africa_energy_v1_sample.parquet
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir / "curation"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--parquet",    type=Path, required=True)
    ap.add_argument("--dry-run",    action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve()
    pq   = args.parquet.resolve()

    if not ckpt.exists():
        sys.exit(f"ERROR: no checkpoint at {ckpt}")
    if not pq.exists():
        sys.exit(f"ERROR: no parquet at {pq}")

    print(f"checkpoint: {ckpt}")
    print(f"parquet:    {pq}")

    parquet_ids = set(pd.read_parquet(pq, columns=["asset_id"])["asset_id"])
    print(f"  parquet asset_ids: {len(parquet_ids):,}")

    with ckpt.open("rb") as f:
        data = pickle.load(f)
    results       = data.get("results", [])
    completed_ids = data.get("completed_ids", [])
    if not isinstance(completed_ids, set):
        completed_ids = set(completed_ids)

    print(f"  checkpoint results: {len(results):,}")
    print(f"  checkpoint completed_ids: {len(completed_ids):,}")
    status_counts = Counter(getattr(r, "status", "?") for r in results)
    print(f"  status counts: {dict(status_counts)}")

    # Keep only ok results whose asset_id is in the parquet
    kept = [
        r for r in results
        if getattr(r, "status", None) == "ok"
        and getattr(r, "asset_id", None) in parquet_ids
    ]
    kept_ids = {r.asset_id for r in kept}

    # Drop: non-ok of any kind, plus ok-but-not-in-new-parquet
    n_drop_status   = sum(1 for r in results if getattr(r, "status", None) != "ok")
    n_drop_outside  = sum(1 for r in results
                          if getattr(r, "status", None) == "ok"
                          and getattr(r, "asset_id", None) not in parquet_ids)

    print()
    print(f"  KEEP (ok AND in new parquet): {len(kept):,}")
    print(f"  DROP (non-ok):                {n_drop_status:,}")
    print(f"  DROP (ok but not in parquet): {n_drop_outside:,}")
    print(f"  -> will need to fetch:        {len(parquet_ids) - len(kept):,}")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = ckpt.with_suffix(ckpt.suffix + f".before_align-{ts}")
    shutil.copy2(ckpt, backup)
    print(f"\nbackup: {backup}")

    new_data = {
        "results"      : kept,
        "completed_ids": list(kept_ids),
    }
    tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(ckpt)
    print(f"checkpoint aligned: {ckpt} ({ckpt.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
