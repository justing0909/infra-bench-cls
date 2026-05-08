"""
diagnose_stac.py
----------------
Diagnostic for the Planetary Computer STAC fetcher.

Goal: figure out where the per-tile time is actually going.

Usage:
    python diagnose_stac.py --assets-table <path> --n 30 --seed 42

Flags:
    --gdal-tweaks      Apply GDAL HTTP/2 + VSI cache env vars
    --cache-catalog    Reuse pystac catalog per worker thread
    --workers N        Worker count (default 16). Try 8, 16, 32 to find ceiling.
    --stage-timing     Instrument every STAC search, every sign(), every
                       rasterio.open + windowed read, and report a breakdown.

The --stage-timing flag is the key diagnostic. It tells us whether per-tile
seconds are spent in:
    (a) catalog.search()  — STAC API call to find scenes
    (b) sign()            — generating signed asset URLs (mostly local)
    (c) rasterio.open()   — TLS handshake + COG header read
    (d) src.read(window=) — actual range-read of pixel data
"""

import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from statistics import median

import pandas as pd


# ---------------------------------------------------------------------------
# GDAL tweaks (must run before rasterio is imported)
# ---------------------------------------------------------------------------

def apply_gdal_tweaks() -> None:
    os.environ.setdefault("GDAL_HTTP_VERSION", "2")
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
    os.environ.setdefault("VSI_CACHE", "TRUE")
    os.environ.setdefault("VSI_CACHE_SIZE", "1000000000")
    os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    print("  [gdal] tweaks applied (HTTP/2, multiplex, VSI cache 1GB)")


# ---------------------------------------------------------------------------
# Network baseline
# ---------------------------------------------------------------------------

def network_baseline() -> None:
    import urllib.request
    url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    rtts = []
    for _ in range(5):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                r.read(1)
        except Exception as e:
            print(f"  [baseline] STAC ping failed: {e}")
            return
        rtts.append(time.perf_counter() - t0)
    print(f"  [baseline] STAC API RTT: median={median(rtts)*1000:.0f}ms "
          f"min={min(rtts)*1000:.0f}ms max={max(rtts)*1000:.0f}ms")


# ---------------------------------------------------------------------------
# Stage-timing instrumentation
# ---------------------------------------------------------------------------
# Approach: monkey-patch the methods that do network work so each call
# records its duration into a global accumulator.
# ---------------------------------------------------------------------------

class StageTimer:
    """Thread-safe accumulator for per-stage timings."""

    def __init__(self):
        self._lock = threading.Lock()
        self._totals = defaultdict(float)
        self._counts = defaultdict(int)

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._totals[stage] += seconds
            self._counts[stage] += 1

    def report(self, n_tiles: int, wall_seconds: float) -> None:
        print("\n=== Per-stage timing ===")
        print(f"  (totals across all {n_tiles} tiles, summed across all worker threads)")
        print(f"  {'stage':<28}{'calls':>8}{'total_s':>12}{'per_call_ms':>16}{'per_tile_s':>14}")
        ordered = sorted(self._totals.items(), key=lambda kv: -kv[1])
        for stage, total in ordered:
            n = self._counts[stage]
            per_call_ms = (total / n) * 1000 if n else 0
            per_tile_s = total / n_tiles if n_tiles else 0
            print(f"  {stage:<28}{n:>8}{total:>12.2f}{per_call_ms:>16.1f}{per_tile_s:>14.2f}")

        total_stage_s = sum(self._totals.values())
        print(f"\n  Sum of all stage time: {total_stage_s:.1f}s "
              f"(across all workers concurrently)")
        print(f"  Wall time:             {wall_seconds:.1f}s")
        if wall_seconds > 0:
            effective_workers = total_stage_s / wall_seconds
            print(f"  Effective parallelism: {effective_workers:.1f}x")
            print(f"  (If this is much less than your --workers count, you're")
            print(f"   not actually getting parallel work — likely because")
            print(f"   GIL contention, lock contention, or sequential structure")
            print(f"   inside fetch_tile is serializing the workers.)")


_TIMER: StageTimer = StageTimer()


def _wrap_method(cls, attr_name: str, stage_name: str, timer: StageTimer):
    original = getattr(cls, attr_name)

    def wrapped(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            timer.add(stage_name, time.perf_counter() - t0)

    setattr(cls, attr_name, wrapped)


def _wrap_function(module, attr_name: str, stage_name: str, timer: StageTimer):
    original = getattr(module, attr_name)

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timer.add(stage_name, time.perf_counter() - t0)

    setattr(module, attr_name, wrapped)


def install_stage_timing(timer: StageTimer) -> None:
    import pystac_client
    import planetary_computer
    import rasterio

    # 1. STAC API search
    _wrap_method(pystac_client.Client, "search", "stac_search", timer)

    # 2. items() iteration — actual network roundtrip happens here
    try:
        from pystac_client.item_search import ItemSearch
        _wrap_method(ItemSearch, "items", "stac_items_iter", timer)
    except Exception as e:
        print(f"  [stage-timing] couldn't patch ItemSearch.items: {e}")

    # 3. Signing
    _wrap_function(planetary_computer, "sign", "pc_sign", timer)
    if hasattr(planetary_computer, "sign_inplace"):
        _wrap_function(planetary_computer, "sign_inplace", "pc_sign_inplace", timer)

    # 4. rasterio.open — opens remote COG
    _wrap_function(rasterio, "open", "rio_open", timer)

    # 5. rasterio reads
    try:
        from rasterio.io import DatasetReader
        _wrap_method(DatasetReader, "read", "rio_read", timer)
    except Exception as e:
        print(f"  [stage-timing] couldn't patch DatasetReader.read: {e}")

    print("  [stage-timing] instrumented STAC search, sign, rasterio.open, rasterio.read")


# ---------------------------------------------------------------------------
# Cached-catalog wrapper
# ---------------------------------------------------------------------------

class CachedCatalogFetcher:
    def __init__(self, base_fetcher, use_sign_modifier: bool = True):
        self._base = base_fetcher
        self._local = threading.local()
        self._use_sign_modifier = use_sign_modifier

    def _catalog(self):
        cat = getattr(self._local, "catalog", None)
        if cat is None:
            import pystac_client
            import planetary_computer
            if self._use_sign_modifier:
                cat = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=planetary_computer.sign_inplace,
                )
            else:
                cat = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                )
            self._local.catalog = cat
        return cat

    def fetch_tile(self, row):
        original = self._base._get_catalog
        self._base._get_catalog = self._catalog
        try:
            return self._base.fetch_tile(row)
        finally:
            self._base._get_catalog = original


class NoSignModifierFetcher:
    """
    Wraps a fetcher so its _get_catalog returns a client WITHOUT the
    sign_inplace modifier. Used when --no-sign-modifier is passed alone
    (without --cache-catalog).
    """

    def __init__(self, base_fetcher):
        self._base = base_fetcher
        self._local = threading.local()

    def _catalog(self):
        cat = getattr(self._local, "catalog", None)
        if cat is None:
            import pystac_client
            cat = pystac_client.Client.open(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
            )
            self._local.catalog = cat
        return cat

    def fetch_tile(self, row):
        original = self._base._get_catalog
        self._base._get_catalog = self._catalog
        try:
            return self._base.fetch_tile(row)
        finally:
            self._base._get_catalog = original


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_diagnostic(args) -> None:
    if args.gdal_tweaks:
        apply_gdal_tweaks()

    print("\n=== Network baseline ===")
    network_baseline()

    print("\n=== Loading sample ===")
    if args.assets_table.endswith(".parquet"):
        df = pd.read_parquet(args.assets_table)
    else:
        df = pd.read_csv(args.assets_table)
    if args.seed is not None:
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    else:
        df = df.head(args.n)
    df = df.reset_index(drop=True)
    print(f"  loaded {len(df)} assets from {args.assets_table}")

    sys.path.insert(0, os.path.dirname(os.path.abspath(args.stac_module_dir)))

    # Install stage timing BEFORE importing stac_imagery, so its references
    # to rasterio.open / planetary_computer.sign pick up the wrapped versions.
    if args.stage_timing:
        # The patches need to happen on the actual module objects. Import the
        # libraries here, patch them, then import stac_imagery — its internal
        # `import rasterio` etc. will get the same module objects (Python
        # caches modules), so the patches propagate.
        import rasterio  # noqa
        import pystac_client  # noqa
        import planetary_computer  # noqa
        install_stage_timing(_TIMER)

    from stac_imagery import STACImageryFetcher

    print("\n=== Building fetcher ===")
    fetcher = STACImageryFetcher(
        buffer_m             = 300,
        modalities           = ["sentinel2_ms", "sentinel1"],
        temporal_stack       = False,
        adaptive_concurrency = False,
        start_workers        = args.workers,
        max_workers          = args.workers,
        checkpoint_path      = None,
    )

    if args.cache_catalog and args.no_sign_modifier:
        print("  [patch] catalog caching enabled (per-thread reuse)")
        print("  [patch] sign_inplace modifier DISABLED (relying on explicit sign() calls)")
        fetcher = CachedCatalogFetcher(fetcher, use_sign_modifier=False)
        per_tile_runner = fetcher.fetch_tile
    elif args.cache_catalog:
        print("  [patch] catalog caching enabled (per-thread reuse)")
        fetcher = CachedCatalogFetcher(fetcher, use_sign_modifier=True)
        per_tile_runner = fetcher.fetch_tile
    elif args.no_sign_modifier:
        print("  [patch] sign_inplace modifier DISABLED (relying on explicit sign() calls)")
        fetcher = NoSignModifierFetcher(fetcher)
        per_tile_runner = fetcher.fetch_tile
    else:
        per_tile_runner = fetcher.fetch_tile

    print("\n=== Running fetch ===")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = [row for _, row in df.iterrows()]
    per_tile_seconds = []
    n_ok = n_fail = 0
    t_start = time.perf_counter()

    def _timed(row):
        t0 = time.perf_counter()
        result = per_tile_runner(row)
        return result, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_timed, r) for r in rows]
        for fut in as_completed(futures):
            result, dt = fut.result()
            per_tile_seconds.append(dt)
            if result.status == "ok":
                n_ok += 1
            else:
                n_fail += 1

    t_total = time.perf_counter() - t_start

    print("\n=== Results ===")
    print(f"  total wall time:   {t_total:.1f}s")
    print(f"  tiles ok / fail:   {n_ok} / {n_fail}")
    print(f"  tiles/sec:         {n_ok / t_total:.2f}")
    if per_tile_seconds:
        s = sorted(per_tile_seconds)
        p50 = s[len(s) // 2]
        p95 = s[int(len(s) * 0.95)]
        print(f"  per-tile median:   {p50:.2f}s")
        print(f"  per-tile p95:      {p95:.2f}s")
        print(f"  per-tile min/max:  {min(s):.2f}s / {max(s):.2f}s")
    print(f"  workers:           {args.workers}")
    print(f"  gdal tweaks:       {args.gdal_tweaks}")
    print(f"  catalog cached:    {args.cache_catalog}")
    print(f"  sign modifier:     {'OFF (explicit only)' if args.no_sign_modifier else 'ON (sign_inplace)'}")
    print(f"  stage timing:      {args.stage_timing}")

    if args.stage_timing:
        _TIMER.report(n_tiles=n_ok, wall_seconds=t_total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets-table", required=True)
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=16,
                   help="Worker count (default 16). Sweep 8, 16, 32, 64 to find ceiling.")
    p.add_argument("--gdal-tweaks", action="store_true")
    p.add_argument("--cache-catalog", action="store_true")
    p.add_argument("--stage-timing", action="store_true",
                   help="Instrument STAC API, sign, rasterio.open, rasterio.read")
    p.add_argument("--no-sign-modifier", action="store_true",
                   help="Skip the sign_inplace catalog modifier; rely on per-modality "
                        "explicit sign() calls. Tests whether bulk signing of unused "
                        "items is the bottleneck.")
    p.add_argument("--stac-module-dir", default=".")
    args = p.parse_args()
    run_diagnostic(args)


if __name__ == "__main__":
    main()