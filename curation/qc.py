"""
qc.py
-----
imagery quality control for infrastructure asset tiles.

handles multimodal tiles (sentinel2_ms, sentinel1, landsat_thermal, naip)
with per-modality thresholds.

QC is always applied to TileResult.image — the single best composite.
temporal stacks (image_stack) are not QC'd individually; if the best
composite passes, the stack is accepted.

four checks in order:
  1. valid pixel ratio  — rejects tiles that are mostly nodata/zero.
                          checks optical bands only — SAR and thermal can
                          have non-zero values even when optical data is
                          empty, which would falsely inflate the ratio.
  2. edge artifact      — rejects tiles clipped at scene boundary
  3. minimum tile size  — rejects tiles too small to be meaningful
                          (catches edge cases from windowed reads)
  4. value range        — per-modality range check on optical bands only

Usage:
    from curation.qc import QualityChecker
    checker = QualityChecker()
    qc_results = checker.check_all(tile_results)
    clean      = checker.filter_ok(qc_results)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from .helpers.tile_types import TileResult, MODALITY_REGISTRY


# ---------------------------------------------------------------------------
# default thresholds
# ---------------------------------------------------------------------------

DEFAULT_MIN_VALID_RATIO = 0.80
DEFAULT_EDGE_MARGIN_PX  = 5
DEFAULT_MIN_TILE_DIM    = 20   # pixels — reject tiles smaller than this

# per-modality value range thresholds (applied to optical bands only)
MODALITY_RANGE_THRESHOLDS = {
    "sentinel2_ms":    (15, 220),
    "sentinel2_rgb":   (15, 220),
    "naip":            (15, 220),
    "sentinel1":       (-28.0, 5.0),
    "landsat_thermal": (220.0, 340.0),
}

DEFAULT_RANGE_THRESHOLD = (15, 220)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _optical_bands(image: np.ndarray, modalities: List[str]) -> np.ndarray:
    """
    returns the optical-only slice of a multimodal (C, H, W) array.
    optical modalities are those with dtype='uint8' in MODALITY_REGISTRY.
    falls back to full image if no modality info available.
    """
    if not modalities:
        return image

    n_optical = sum(
        MODALITY_REGISTRY[m]["n_bands"]
        for m in modalities
        if m in MODALITY_REGISTRY
        and MODALITY_REGISTRY[m]["dtype"] == "uint8"
        and m != "naip"  # NAIP is optical but CONUS-only, don't use for primary check
    )

    if n_optical == 0 or n_optical > image.shape[0]:
        return image

    return image[:n_optical]


# ---------------------------------------------------------------------------
# check functions
# ---------------------------------------------------------------------------

def _check_valid_pixels(image: np.ndarray, min_ratio: float,
                         modalities: List[str]) -> tuple:
    """
    checks that enough pixels contain real optical data.
    uses optical bands only — SAR/thermal can be non-zero even when
    optical data is empty (clouds, no-data fill), inflating the ratio.
    """
    optical     = _optical_bands(image, modalities)
    valid_mask  = np.any(optical != 0, axis=0)
    valid_ratio = float(valid_mask.mean())
    return valid_ratio >= min_ratio, valid_ratio


def _check_edge_artifacts(image: np.ndarray, margin_px: int,
                           zero_threshold: float = 0.90) -> bool:
    """
    checks for edge clipping using optical bands only.
    """
    _, h, w = image.shape
    if h < margin_px * 2 or w < margin_px * 2:
        return False

    strips = [
        image[:, :margin_px, :],
        image[:, -margin_px:, :],
        image[:, :, :margin_px],
        image[:, :, -margin_px:],
    ]
    for strip in strips:
        zero_frac = (strip == 0).all(axis=0).mean()
        if zero_frac > zero_threshold:
            return False
    return True


def _check_min_size(image: np.ndarray,
                    min_dim: int = DEFAULT_MIN_TILE_DIM) -> bool:
    """
    rejects tiles where either spatial dimension is below min_dim pixels.
    catches edge cases from windowed reads near scene boundaries.
    """
    _, h, w = image.shape
    return h >= min_dim and w >= min_dim


def _check_value_range(image: np.ndarray,
                        modalities: List[str]) -> tuple:
    """
    checks mean value of the optical bands against expected range.
    uses the primary optical modality's thresholds.
    """
    primary = modalities[0] if modalities else "sentinel2_rgb"
    vmin, vmax = MODALITY_RANGE_THRESHOLDS.get(primary, DEFAULT_RANGE_THRESHOLD)

    optical  = _optical_bands(image, modalities)
    nonzero  = optical[optical != 0]
    if nonzero.size == 0:
        return False, 0.0

    mean_val = float(nonzero.mean())
    passed   = vmin <= mean_val <= vmax
    return passed, mean_val


# ---------------------------------------------------------------------------
# QC result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    """
    holds the quality control outcome for one TileResult.

    status values:
        "pass"
        "fail_valid_pixels"
        "fail_edge_artifacts"
        "fail_min_size"
        "fail_value_range"
        "fail_no_image"
    """
    asset_id        : str
    asset_type      : str
    source          : str
    status          : str
    valid_ratio     : float
    mean_value      : float
    modalities      : List[str]
    checks_passed   : List[str] = field(default_factory=list)
    checks_failed   : List[str] = field(default_factory=list)
    tile            : Optional[TileResult] = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------

class QualityChecker:
    """
    runs QC checks on imagery tiles.
    modality-aware: valid pixel and value range checks use optical bands only.

    Parameters
    ----------
    min_valid_ratio : float — minimum fraction of valid optical pixels
    edge_margin_px  : int   — width of border strip to check for clipping
    min_tile_dim    : int   — minimum spatial dimension in pixels
    """

    def __init__(
        self,
        min_valid_ratio : float = DEFAULT_MIN_VALID_RATIO,
        edge_margin_px  : int   = DEFAULT_EDGE_MARGIN_PX,
        min_tile_dim    : int   = DEFAULT_MIN_TILE_DIM,
    ):
        self.min_valid_ratio = min_valid_ratio
        self.edge_margin_px  = edge_margin_px
        self.min_tile_dim    = min_tile_dim

    def check_tile(self, tile: TileResult) -> QCResult:
        modalities = getattr(tile, "modalities", ["sentinel2_rgb"])

        if tile.image is None:
            return QCResult(
                asset_id      = tile.asset_id,
                asset_type    = tile.asset_type,
                source        = tile.source,
                status        = "fail_no_image",
                valid_ratio   = 0.0,
                mean_value    = 0.0,
                modalities    = modalities,
                checks_failed = ["no_image"],
                tile          = tile,
            )

        image         = tile.image
        passed_checks = []
        failed_checks = []

        # check 1: valid pixels (optical bands only)
        valid_ok, valid_ratio = _check_valid_pixels(
            image, self.min_valid_ratio, modalities
        )
        (passed_checks if valid_ok else failed_checks).append("valid_pixels")

        # check 2: edge artifacts
        edge_ok = _check_edge_artifacts(image, self.edge_margin_px)
        (passed_checks if edge_ok else failed_checks).append("edge_artifacts")

        # check 3: minimum tile size
        size_ok = _check_min_size(image, self.min_tile_dim)
        (passed_checks if size_ok else failed_checks).append("min_size")

        # check 4: value range (optical bands only)
        range_ok, mean_val = _check_value_range(image, modalities)
        (passed_checks if range_ok else failed_checks).append("value_range")

        status = f"fail_{failed_checks[0]}" if failed_checks else "pass"

        return QCResult(
            asset_id      = tile.asset_id,
            asset_type    = tile.asset_type,
            source        = tile.source,
            status        = status,
            valid_ratio   = valid_ratio,
            mean_value    = mean_val,
            modalities    = modalities,
            checks_passed = passed_checks,
            checks_failed = failed_checks,
            tile          = tile,
        )

    def check_all(self, tiles: List[TileResult],
                  max_workers: int = 8) -> List["QCResult"]:
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
                        print(f"  QC [{completed}/{len(tiles)}] passed={passed}")

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        print(f"QC complete: {passed} passed, {failed} failed "
              f"({len(results)} total)")
        return results

    def filter_ok(self, qc_results: List["QCResult"]) -> List[TileResult]:
        return [r.tile for r in qc_results if r.passed and r.tile is not None]

    def summarize(self, qc_results: List["QCResult"]) -> pd.DataFrame:
        rows = []
        for r in qc_results:
            rows.append({
                "asset_id":      r.asset_id,
                "asset_type":    r.asset_type,
                "source":        r.source,
                "modalities":    "+".join(r.modalities),
                "status":        r.status,
                "passed":        r.passed,
                "valid_ratio":   round(r.valid_ratio, 3),
                "mean_value":    round(r.mean_value, 3),
                "checks_failed": ", ".join(r.checks_failed) or "none",
            })
        return pd.DataFrame(rows)

    def failure_analysis(self, qc_results: List["QCResult"]) -> None:
        df = self.summarize(qc_results)
        print("=== QC Failure Analysis ===")
        print(f"Total:   {len(df)}")
        print(f"Passed:  {df['passed'].sum()}")
        print(f"Failed:  {(~df['passed']).sum()}")
        print("\nFailures by reason:")
        fail_df = df[~df["passed"]]
        if fail_df.empty:
            print("  None — all tiles passed!")
        else:
            print(fail_df["status"].value_counts().to_string())
        print("\nFailures by asset type:")
        if not fail_df.empty:
            print(fail_df["asset_type"].value_counts().to_string())
        print(f"\nValid ratio range: "
              f"{df['valid_ratio'].min():.3f} – {df['valid_ratio'].max():.3f}")
        print(f"Mean value range:  "
              f"{df['mean_value'].min():.3f} – {df['mean_value'].max():.3f}")