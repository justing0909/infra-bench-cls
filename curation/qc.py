"""
qc.py
-----
Basic imagery quality control for infrastructure asset tiles.
Filters out tiles that are unsuitable for training before they
enter the deduplication and label triage steps.

Three checks are applied in order:
  1. Valid pixel ratio  — rejects tiles that are mostly nodata/black
  2. Edge artifact      — rejects tiles clipped at the boundary of a scene
  3. Cloud/brightness   — rejects tiles that are too bright (likely cloud)
                          or too dark (shadow, missing data)

Each check is configurable via thresholds. Defaults are conservative —
when in doubt, keep the tile and let triage handle edge cases.

Usage:
    from qc import QualityChecker
    checker = QualityChecker()
    qc_results = checker.check_all(tile_results)   # list of TileResult
    clean      = checker.filter_ok(qc_results)     # only passing tiles
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from legacy.imagery import TileResult


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------
# These are starting points — tune based on your imagery and region.

DEFAULT_MIN_VALID_RATIO  = 0.80   # at least 80% of pixels must be non-zero
DEFAULT_EDGE_MARGIN_PX   = 5      # flag if any border strip is >90% zero
DEFAULT_MAX_BRIGHTNESS   = 220    # mean brightness above this = likely cloud
DEFAULT_MIN_BRIGHTNESS   = 15     # mean brightness below this = likely nodata


# ---------------------------------------------------------------------------
# QC result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    """
    Holds the quality control outcome for one TileResult.

    Attributes
    ----------
    asset_id      : str
    asset_type    : str
    source        : str   — "naip" or "sentinel2"
    status        : str   — "pass", "fail_valid_pixels", "fail_edge",
                            "fail_brightness", "fail_no_image"
    valid_ratio   : float — fraction of non-zero pixels
    mean_brightness : float
    checks_passed : list  — which checks passed
    checks_failed : list  — which checks failed
    tile          : TileResult — reference to original tile
    """
    asset_id        : str
    asset_type      : str
    source          : str
    status          : str
    valid_ratio     : float
    mean_brightness : float
    checks_passed   : List[str] = field(default_factory=list)
    checks_failed   : List[str] = field(default_factory=list)
    tile            : Optional[TileResult] = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_valid_pixels(image: np.ndarray,
                        min_ratio: float) -> tuple:
    """
    Checks that enough pixels contain real data (non-zero across all bands).
    Returns (passed: bool, valid_ratio: float).
    """
    # pixel is valid if at least one band is non-zero
    valid_mask = np.any(image > 0, axis=0)
    valid_ratio = valid_mask.mean()
    return valid_ratio >= min_ratio, float(valid_ratio)


def _check_edge_artifacts(image: np.ndarray,
                          margin_px: int,
                          zero_threshold: float = 0.90) -> bool:
    """
    Checks for edge clipping — a border strip that is mostly zero suggests
    the tile was cut at the edge of imagery coverage.
    Returns True if the tile passes (no significant edge artifact detected).
    """
    _, h, w = image.shape
    if h < margin_px * 2 or w < margin_px * 2:
        return False   # tile too small to evaluate

    # check each of the four border strips
    strips = [
        image[:, :margin_px, :],          # top
        image[:, -margin_px:, :],         # bottom
        image[:, :, :margin_px],          # left
        image[:, :, -margin_px:],         # right
    ]

    for strip in strips:
        zero_frac = (strip == 0).all(axis=0).mean()
        if zero_frac > zero_threshold:
            return False   # this border is mostly empty

    return True


def _check_brightness(image: np.ndarray,
                      min_brightness: float,
                      max_brightness: float) -> tuple:
    """
    Checks mean pixel brightness across all bands.
    Too bright = cloud/overexposure. Too dark = shadow/nodata.
    Returns (passed: bool, mean_brightness: float).
    """
    mean_brightness = float(image[image > 0].mean()) if (image > 0).any() else 0.0
    passed = min_brightness <= mean_brightness <= max_brightness
    return passed, mean_brightness


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QualityChecker:
    """
    Runs quality control checks on imagery tiles from imagery.py.

    Parameters
    ----------
    min_valid_ratio  : float — minimum fraction of valid (non-zero) pixels
    edge_margin_px   : int   — width of border strip to check for clipping
    max_brightness   : float — upper brightness threshold (cloud rejection)
    min_brightness   : float — lower brightness threshold (nodata rejection)
    """

    def __init__(
        self,
        min_valid_ratio : float = DEFAULT_MIN_VALID_RATIO,
        edge_margin_px  : int   = DEFAULT_EDGE_MARGIN_PX,
        max_brightness  : float = DEFAULT_MAX_BRIGHTNESS,
        min_brightness  : float = DEFAULT_MIN_BRIGHTNESS,
    ):
        self.min_valid_ratio = min_valid_ratio
        self.edge_margin_px  = edge_margin_px
        self.max_brightness  = max_brightness
        self.min_brightness  = min_brightness

    def check_tile(self, tile: TileResult) -> QCResult:
        """
        Runs all QC checks on a single TileResult.
        Returns a QCResult with pass/fail status and diagnostic values.
        """
        # Handle tiles that failed at the imagery fetch stage
        if tile.image is None:
            return QCResult(
                asset_id=tile.asset_id,
                asset_type=tile.asset_type,
                source=tile.source,
                status="fail_no_image",
                valid_ratio=0.0,
                mean_brightness=0.0,
                checks_failed=["no_image"],
                tile=tile,
            )

        image = tile.image
        passed_checks = []
        failed_checks = []

        # --- Check 1: valid pixels ---
        valid_ok, valid_ratio = _check_valid_pixels(image, self.min_valid_ratio)
        if valid_ok:
            passed_checks.append("valid_pixels")
        else:
            failed_checks.append("valid_pixels")

        # --- Check 2: edge artifacts ---
        edge_ok = _check_edge_artifacts(image, self.edge_margin_px)
        if edge_ok:
            passed_checks.append("edge_artifacts")
        else:
            failed_checks.append("edge_artifacts")

        # --- Check 3: brightness ---
        brightness_ok, mean_brightness = _check_brightness(
            image, self.min_brightness, self.max_brightness
        )
        if brightness_ok:
            passed_checks.append("brightness")
        else:
            failed_checks.append("brightness")

        # Overall status
        if failed_checks:
            status = f"fail_{failed_checks[0]}"
        else:
            status = "pass"

        return QCResult(
            asset_id=tile.asset_id,
            asset_type=tile.asset_type,
            source=tile.source,
            status=status,
            valid_ratio=valid_ratio,
            mean_brightness=mean_brightness,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            tile=tile,
        )

    def check_all(self, tiles: List[TileResult],
                  max_workers: int = 8) -> List[QCResult]:
        """
        Runs QC on a list of TileResults in parallel.
        Returns a list of QCResult in the same order.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        results   = [None] * len(tiles)
        lock      = threading.Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.check_tile, tile): i
                       for i, tile in enumerate(tiles)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                with lock:
                    completed += 1
                    if completed % 1000 == 0 or completed == len(tiles):
                        passed = sum(1 for r in results if r and r.passed)
                        print(f"  QC [{completed}/{len(tiles)}] "
                              f"passed={passed}")

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        print(f"QC complete: {passed} passed, {failed} failed "
              f"({len(results)} total)")
        return results

    def filter_ok(self, qc_results: List[QCResult]) -> List[TileResult]:
        """
        Returns only the TileResults that passed all QC checks.
        Use this to get clean tiles ready for deduplication.
        """
        return [r.tile for r in qc_results if r.passed and r.tile is not None]

    def summarize(self, qc_results: List[QCResult]) -> pd.DataFrame:
        """
        Converts QC results to a summary DataFrame for inspection.
        """
        rows = []
        for r in qc_results:
            rows.append({
                "asset_id":        r.asset_id,
                "asset_type":      r.asset_type,
                "source":          r.source,
                "status":          r.status,
                "passed":          r.passed,
                "valid_ratio":     round(r.valid_ratio, 3),
                "mean_brightness": round(r.mean_brightness, 1),
                "checks_failed":   ", ".join(r.checks_failed) or "none",
            })
        return pd.DataFrame(rows)

    def failure_analysis(self, qc_results: List[QCResult]) -> None:
        """
        Prints a breakdown of failure reasons to help tune thresholds.
        """
        df = self.summarize(qc_results)
        print("=== QC Failure Analysis ===")
        print(f"Total tiles:  {len(df)}")
        print(f"Passed:       {df['passed'].sum()}")
        print(f"Failed:       {(~df['passed']).sum()}")
        print()
        print("Failures by reason:")
        fail_df = df[~df["passed"]]
        if fail_df.empty:
            print("  None — all tiles passed!")
        else:
            print(fail_df["status"].value_counts().to_string())
        print()
        print("Failures by asset type:")
        if fail_df.empty:
            print("  None")
        else:
            print(fail_df["asset_type"].value_counts().to_string())
        print()
        print(f"Valid ratio range:     "
              f"{df['valid_ratio'].min():.3f} – {df['valid_ratio'].max():.3f}")
        print(f"Brightness range:      "
              f"{df['mean_brightness'].min():.1f} – "
              f"{df['mean_brightness'].max():.1f}")


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pandas as pd
    from curation.sources import GeoFabrikSource
    from curation.legacy.imagery import ImageryFetcher

    PBF_PATH  = "data/pbf/us-northeast-260407.osm_power_only.osm.pbf"
    INPUT_CSV = "data/us-northeast_all_assets.csv"

    # Load assets
    if os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} assets from {INPUT_CSV}")
    else:
        src = GeoFabrikSource(PBF_PATH, min_confidence="medium")
        df  = src.extract_all()

    # Sample a few per type
    df_sample = (
        df.groupby("asset_type", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), 2), random_state=42))
          .reset_index(drop=True)
    )

    print(f"\nFetching tiles for {len(df_sample)} sampled assets...")
    fetcher = ImageryFetcher(buffer_m=150, sources=["sentinel2"])
    tiles   = fetcher.fetch_all(df_sample)

    print("\nRunning QC...")
    checker = QualityChecker()
    qc_results = checker.check_all(tiles)

    print("\nQC summary:")
    print(checker.summarize(qc_results).to_string(index=False))

    print()
    checker.failure_analysis(qc_results)

    clean_tiles = checker.filter_ok(qc_results)
    print(f"\n{len(clean_tiles)} clean tiles ready for deduplication.")