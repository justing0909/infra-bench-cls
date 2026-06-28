"""
scrub_checkpoint_failures.py
----------------------------
Repair a STAC fetcher checkpoint that captured a network outage as
"completed but failed" entries. Drops every TileResult whose status
is not "ok" and removes their asset_ids from `completed_ids`, so the
next fetch_all() call re-attempts them instead of skipping them.

Reads:  the .pkl path you pass
Writes: a backup at <path>.before_scrub-<ts>.pkl, then atomically
        replaces <path> with the cleaned version.

Usage
-----
python scrub_checkpoint_failures.py data/checkpoints/v1/africa_energy_fetch.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

# TileResult is pickled with module path `helpers.tile_types` (from the
# curation/ root). Make that resolvable.
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir / "curation"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Report counts but don't write.")
    args = ap.parse_args()

    ckpt = args.checkpoint.resolve()
    if not ckpt.exists():
        sys.exit(f"ERROR: no such checkpoint: {ckpt}")

    print(f"loading: {ckpt}")
    with ckpt.open("rb") as f:
        data = pickle.load(f)

    results       = data.get("results", [])
    completed_ids = data.get("completed_ids", [])
    if not isinstance(completed_ids, set):
        completed_ids = set(completed_ids)

    print(f"  results        = {len(results):,}")
    print(f"  completed_ids  = {len(completed_ids):,}")

    status_counts = Counter(getattr(r, "status", "?") for r in results)
    print(f"  status counts  = {dict(status_counts)}")

    ok_results = [r for r in results if getattr(r, "status", None) == "ok"]
    bad_results = [r for r in results if getattr(r, "status", None) != "ok"]
    bad_ids = {getattr(r, "asset_id", None) for r in bad_results
               if getattr(r, "asset_id", None) is not None}

    new_completed_ids = completed_ids - bad_ids

    print()
    print(f"  KEEP (status='ok'):       {len(ok_results):,}")
    print(f"  DROP (status!='ok'):      {len(bad_results):,}")
    print(f"  ids removed from skip:    {len(bad_ids):,}")
    print(f"  new completed_ids:        {len(new_completed_ids):,}")

    # Sanity check: every kept result's asset_id should be in new_completed_ids
    kept_ids = {getattr(r, "asset_id", None) for r in ok_results}
    leaked = kept_ids - new_completed_ids
    if leaked:
        print(f"  WARNING: {len(leaked)} ok results have no matching completed_id; "
              "they will be re-fetched but that's harmless.")
    # And no dropped id should be in the new set
    overlap = bad_ids & new_completed_ids
    if overlap:
        sys.exit(f"  ERROR: {len(overlap)} dropped ids still in completed_ids — abort.")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = ckpt.with_suffix(ckpt.suffix + f".before_scrub-{ts}")
    shutil.copy2(ckpt, backup)
    print(f"\nbackup written: {backup}")

    new_data = {
        "results"      : ok_results,
        "completed_ids": list(new_completed_ids),
    }
    tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(ckpt)
    print(f"checkpoint scrubbed: {ckpt} ({ckpt.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
