"""
fetch_v1_overnight.py
---------------------
Drives STAC imagery fetch for the 28 Infra-Bench v1 sample parquets.

Per cell:
  load parquet -> STACImageryFetcher.fetch_all -> QC -> triage -> assemble
  -> write data/curated_datasets/dataset_<region>_<sector>_v1/_SUCCESS

Cells with an existing _SUCCESS are skipped, so the script is rerunnable.
Checkpoints land in data/checkpoints/v1/<region>_<sector>_fetch.pkl, written
every 200 tiles by the fetcher.

Usage
-----
# Sanity-check the smallest cell:
python fetch_v1_overnight.py --only central-america telecom

# All 28, smallest-first per region:
python fetch_v1_overnight.py

# All 28 minus the sanity-check cell already done:
python fetch_v1_overnight.py  # _SUCCESS skip is automatic

# Override defaults:
python fetch_v1_overnight.py --start-workers 8 --max-workers 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Force UTF-8 on stdout/stderr. When this script runs detached under
# `cmd /c start /b ... >> log`, Python's stdout defaults to the Windows
# ANSI codepage (cp1252), which can't encode the ↑/↓ arrows that
# stac_imagery's adaptive-concurrency controller prints — that crash
# took down the first overnight run after ~3.5h on cells 2 & 3.
# `errors="replace"` is belt-and-braces in case any other glyph slips in.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
sys.path.insert(0, str(_script_dir / "curation"))

from curation.stac_imagery import STACImageryFetcher
from curation.qc            import QualityChecker
from curation.triage        import RuleBasedTriager
from curation.dataset       import DatasetAssembler
from curation.utils.io_utils import load_asset_table


REGIONS = [
    "central-america",
    "australia-oceania",
    "south-america",
    "africa",
    "asia",
    "north-america",
    "europe",
]

SECTORS = ["energy", "water", "transport", "telecom"]

DEFAULT_MODALITIES = ["sentinel2_ms", "sentinel1"]
DEFAULT_BUFFER_M   = 300


def cell_size(samples_dir: Path, region: str, sector: str) -> int | None:
    """Return row count for a (region, sector) parquet, or None if missing."""
    p = samples_dir / f"{region}_{sector}_v1_sample.parquet"
    if not p.exists():
        return None
    try:
        return len(pd.read_parquet(p, columns=["asset_id"]))
    except Exception:
        return None


def build_run_order(
    samples_dir: Path,
    order: str = "smallest_per_region",
) -> list[tuple[str, str, int]]:
    """Return [(region, sector, n_rows), ...] in execution order."""
    cells = []
    for region in REGIONS:
        for sector in SECTORS:
            n = cell_size(samples_dir, region, sector)
            if n is not None:
                cells.append((region, sector, n))

    if order == "smallest_per_region":
        # Within each region, smallest sector first. Regions in REGIONS order
        # (which is roughly smallest-total -> largest-total).
        by_region: dict[str, list[tuple[str, str, int]]] = {}
        for c in cells:
            by_region.setdefault(c[0], []).append(c)
        out = []
        for region in REGIONS:
            out.extend(sorted(by_region.get(region, []), key=lambda x: x[2]))
        return out
    if order == "smallest_global":
        return sorted(cells, key=lambda x: x[2])
    if order == "sector_major":
        sector_order = ["telecom", "transport", "water", "energy"]
        return sorted(cells, key=lambda x: (sector_order.index(x[1]), x[2]))
    raise ValueError(f"unknown order: {order}")


def run_cell(
    region          : str,
    sector          : str,
    samples_dir     : Path,
    datasets_dir    : Path,
    checkpoints_dir : Path,
    modalities      : list[str],
    buffer_m        : float,
    start_workers   : int,
    max_workers     : int,
    qc_min_valid    : float,
    contradiction   : int,
    low_threshold   : int,
) -> dict:
    """Run the full pipeline for one v1 cell. Returns a result dict."""
    tag         = f"{region}/{sector}"
    parquet     = samples_dir / f"{region}_{sector}_v1_sample.parquet"
    out_dir     = datasets_dir / f"dataset_{region}_{sector}_v1"
    success     = out_dir / "_SUCCESS"
    checkpoint  = checkpoints_dir / f"{region}_{sector}_fetch.pkl"

    print(f"\n{'='*72}")
    print(f"CELL  {tag}   ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"{'='*72}", flush=True)

    if success.exists():
        print(f"  [skip] {success} already exists")
        return {"cell": tag, "status": "skip_done"}

    if not parquet.exists():
        print(f"  [error] missing parquet: {parquet}")
        return {"cell": tag, "status": "missing_parquet"}

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    df = load_asset_table(str(parquet))
    print(f"  loaded {len(df):,} assets from {parquet.name}", flush=True)

    t0 = time.time()

    # ---- 1. STAC fetch ----
    print(f"  [1/4] STAC fetch ...", flush=True)
    fetcher = STACImageryFetcher(
        buffer_m             = buffer_m,
        modalities           = modalities,
        temporal_stack       = False,
        checkpoint_path      = str(checkpoint),
        checkpoint_every     = 200,
        adaptive_concurrency = True,
        start_workers        = start_workers,
        max_workers          = max_workers,
    )
    tiles = fetcher.fetch_all(df)
    n_ok   = sum(1 for t in tiles if t.status == "ok")
    n_fail = len(tiles) - n_ok
    print(f"        fetched ok={n_ok} fail={n_fail}", flush=True)

    if n_ok == 0:
        print(f"  [error] no tiles fetched successfully for {tag}")
        return {
            "cell": tag, "status": "all_failed",
            "n_assets": len(df), "n_ok": 0, "n_fail": n_fail,
            "elapsed_s": round(time.time() - t0, 1),
        }

    # ---- 2. QC ----
    print(f"  [2/4] QC (min_valid_ratio={qc_min_valid}) ...", flush=True)
    checker    = QualityChecker(min_valid_ratio=qc_min_valid)
    qc_results = checker.check_all(tiles, max_workers=4)
    clean      = checker.filter_ok(qc_results)
    print(f"        QC passed {len(clean)} / {len(tiles)}", flush=True)

    # ---- 3. triage ----
    print(f"  [3/4] triage ...", flush=True)
    triager        = RuleBasedTriager(
        contradiction_threshold=contradiction, low_threshold=low_threshold)
    triage_results = triager.triage_all(clean, max_workers=4)
    accepted       = triager.filter_accepted(triage_results)
    print(f"        accepted {len(accepted)}", flush=True)

    if not accepted:
        print(f"  [error] zero accepted tiles after triage for {tag}")
        return {
            "cell": tag, "status": "no_accepted",
            "n_assets": len(df), "n_ok": n_ok, "n_clean": len(clean),
            "n_accepted": 0,
            "elapsed_s": round(time.time() - t0, 1),
        }

    # ---- 4. assemble ----
    print(f"  [4/4] assemble -> {out_dir}", flush=True)
    assembler = DatasetAssembler(output_dir=str(out_dir))
    summary   = assembler.assemble(accepted_tiles=accepted,
                                   triage_results=triage_results)
    n_tiles   = len(summary) if summary is not None else len(accepted)

    elapsed = round(time.time() - t0, 1)
    payload = {
        "completed_at"   : datetime.utcnow().isoformat() + "Z",
        "region"         : region,
        "sector"         : sector,
        "modalities"     : modalities,
        "buffer_m"       : buffer_m,
        "temporal_stack" : False,
        "n_assets"       : len(df),
        "n_fetched_ok"   : n_ok,
        "n_fetched_fail" : n_fail,
        "n_qc_passed"    : len(clean),
        "n_accepted"     : len(accepted),
        "n_dataset_tiles": int(n_tiles),
        "elapsed_s"      : elapsed,
    }
    success.write_text(json.dumps(payload, indent=2))
    print(f"  [done] {tag} in {elapsed:.0f}s — {n_tiles} tiles assembled",
          flush=True)
    return {"cell": tag, "status": "ok", **payload}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples-dir", type=Path,
                    default=_script_dir / "data" / "PIPELINE" / "03-v1-samples")
    ap.add_argument("--datasets-dir", type=Path,
                    default=_script_dir / "data" / "curated_datasets")
    ap.add_argument("--checkpoints-dir", type=Path,
                    default=_script_dir / "data" / "checkpoints" / "v1")
    ap.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    ap.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M)
    ap.add_argument("--start-workers", type=int, default=8)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--qc-min-valid", type=float, default=0.80)
    ap.add_argument("--contradiction-threshold", type=int, default=3)
    ap.add_argument("--low-threshold", type=int, default=4)
    ap.add_argument("--order", choices=["smallest_per_region",
                                        "smallest_global",
                                        "sector_major"],
                    default="smallest_per_region")
    ap.add_argument("--only", nargs=2, metavar=("REGION", "SECTOR"),
                    help="Process only one cell (for sanity-checks).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the run order and exit without fetching.")
    args = ap.parse_args()

    args.samples_dir     = args.samples_dir.resolve()
    args.datasets_dir    = args.datasets_dir.resolve()
    args.checkpoints_dir = args.checkpoints_dir.resolve()
    args.datasets_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        region, sector = args.only
        n = cell_size(args.samples_dir, region, sector)
        if n is None:
            print(f"ERROR: no parquet for {region}/{sector}", file=sys.stderr)
            sys.exit(2)
        plan = [(region, sector, n)]
    else:
        plan = build_run_order(args.samples_dir, order=args.order)

    print(f"v1 overnight fetch — {len(plan)} cell(s) planned")
    print(f"  samples_dir     = {args.samples_dir}")
    print(f"  datasets_dir    = {args.datasets_dir}")
    print(f"  checkpoints_dir = {args.checkpoints_dir}")
    print(f"  modalities      = {args.modalities}")
    print(f"  buffer_m        = {args.buffer_m}  (=> {int(args.buffer_m*2)}m tile)")
    print(f"  workers         = {args.start_workers}/{args.max_workers}")
    print(f"  order           = {args.order}")
    print()
    print(f"{'idx':>3}  {'region':<20} {'sector':<10} {'n_assets':>10}")
    for i, (r, s, n) in enumerate(plan):
        print(f"{i:>3}  {r:<20} {s:<10} {n:>10,}")
    print(f"     {'TOTAL':<31} {sum(n for _,_,n in plan):>10,}", flush=True)

    if args.dry_run:
        return

    results = []
    t_start = time.time()
    for i, (region, sector, n) in enumerate(plan):
        print(f"\n>>> [{i+1}/{len(plan)}] {region}/{sector} (n={n:,})", flush=True)
        try:
            r = run_cell(
                region          = region,
                sector          = sector,
                samples_dir     = args.samples_dir,
                datasets_dir    = args.datasets_dir,
                checkpoints_dir = args.checkpoints_dir,
                modalities      = args.modalities,
                buffer_m        = args.buffer_m,
                start_workers   = args.start_workers,
                max_workers     = args.max_workers,
                qc_min_valid    = args.qc_min_valid,
                contradiction   = args.contradiction_threshold,
                low_threshold   = args.low_threshold,
            )
        except Exception as exc:
            print(f"  [exception] {region}/{sector}: {exc}", flush=True)
            traceback.print_exc()
            r = {"cell": f"{region}/{sector}", "status": "exception",
                 "error": str(exc)}
        results.append(r)

        # Persist a rolling run summary after every cell.
        summary_path = args.datasets_dir / "_v1_overnight_run_summary.json"
        summary_path.write_text(json.dumps({
            "started_at"  : datetime.fromtimestamp(t_start).isoformat(),
            "updated_at"  : datetime.now().isoformat(),
            "n_planned"   : len(plan),
            "n_done"      : i + 1,
            "elapsed_s"   : round(time.time() - t_start, 1),
            "results"     : results,
        }, indent=2))

    total_elapsed = time.time() - t_start
    print(f"\n{'='*72}")
    print(f"ALL DONE in {total_elapsed/60:.1f} min "
          f"({total_elapsed/3600:.2f} h)")
    print(f"{'='*72}")
    status_counts = {}
    for r in results:
        status_counts[r.get("status", "?")] = status_counts.get(r.get("status", "?"), 0) + 1
    for s, c in sorted(status_counts.items()):
        print(f"  {s:<20} {c}")


if __name__ == "__main__":
    main()
