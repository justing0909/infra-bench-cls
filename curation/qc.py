"""
qc.py
-----
Imagery quality control for infrastructure asset tiles.

Extended to support multimodal tiles (sentinel2_ms, sentinel1,
landsat_thermal, naip) with per-modality thresholds.

QC is always applied to TileResult.image — the single best composite.
Temporal stacks (image_stack) are not QC'd individually; if the best
composite passes, the stack is accepted.

Three checks in order:
  1. Valid pixel ratio  — rejects tiles that are mostly nodata/zero
  2. Edge artifact      — rejects tiles clipped at scene boundary
  3. Value range        — replaces the old brightness check with a
                          per-modality range check (uint8 optical uses
                          the same 15-220 defaults; SAR and thermal use
                          modality-appropriate ranges from MODALITY_REGISTRY)

Usage:
    from qc import QualityChecker
    checker = QualityChecker()
    qc_results = checker.check_all(tile_results)
    clean      = checker.filter_ok(qc_results)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

from helpers.tile_types import TileResult, MODALITY_REGISTRY


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_MIN_VALID_RATIO = 0.80
DEFAULT_EDGE_MARGIN_PX  = 5

# Per-modality value range thresholds
# For optical (uint8): mean pixel value must be in [min, max]
# For SAR (float32 dB): mean must be in [min, max]
# For thermal (float32 K): mean must be in [min, max]
# These are intentionally loose — triage handles edge cases.

MODALITY_RANGE_THRESHOLDS = {
    "sentinel2_ms":    (15, 220),     # uint8
    "sentinel2_rgb":   (15, 220),     # uint8
    "naip":            (15, 220),     # uint8
    "sentinel1":       (-28.0, 5.0),  # dB — below -28 is noise, above 5 unusual
    "landsat_thermal": (220.0, 340.0),# Kelvin — sanity range for Earth surfaces
}

# Fallback if modality not in table
DEFAULT_RANGE_THRESHOLD = (15, 220)


# ---------------------------------------------------------------------------
# QC result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    """
    Holds the quality control outcome for one TileResult.

    status values:
        "pass"
        "fail_valid_pixels"
        "fail_edge"
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
# Check functions
# ---------------------------------------------------------------------------

def _check_valid_pixels(image: np.ndarray, min_ratio: float) -> tuple:
    """
    Checks that enough pixels contain real data across all bands.
    A pixel is valid if at least one band is non-zero.
    """
    valid_mask  = np.any(image != 0, axis=0)
    valid_ratio = float(valid_mask.mean())
    return valid_ratio >= min_ratio, valid_ratio


def _check_edge_artifacts(image: np.ndarray, margin_px: int,
                           zero_threshold: float = 0.90) -> bool:
    """
    Checks for edge clipping — a border strip mostly zero/NaN
    suggests the tile was cut at imagery coverage boundary.
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


def _check_value_range(image: np.ndarray,
                        modalities: List[str]) -> tuple:
    """
    Checks mean value of the primary modality's bands against its expected range.
    Uses per-modality thresholds from MODALITY_RANGE_THRESHOLDS.
    Falls back to optical defaults for unknown modalities.

    Returns (passed: bool, mean_value: float)
    """
    # Use the first modality to determine thresholds
    primary = modalities[0] if modalities else "sentinel2_rgb"
    vmin, vmax = MODALITY_RANGE_THRESHOLDS.get(primary, DEFAULT_RANGE_THRESHOLD)

    # Compute mean over non-zero values only
    nonzero = image[image != 0]
    if nonzero.size == 0:
        return False, 0.0

    mean_val = float(nonzero.mean())
    passed   = vmin <= mean_val <= vmax
    return passed, mean_val


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class QualityChecker:
    """
    Runs QC checks on imagery tiles.
    Modality-aware: threshold logic adapts to sentinel1, landsat_thermal, etc.

    Parameters
    ----------
    min_valid_ratio : float — minimum fraction of valid (non-zero) pixels
    edge_margin_px  : int   — width of border strip to check for clipping
    """

    def __init__(
        self,
        min_valid_ratio : float = DEFAULT_MIN_VALID_RATIO,
        edge_margin_px  : int   = DEFAULT_EDGE_MARGIN_PX,
    ):
        self.min_valid_ratio = min_valid_ratio
        self.edge_margin_px  = edge_margin_px

    def check_tile(self, tile: TileResult) -> QCResult:
        modalities = getattr(tile, "modalities", ["sentinel2_rgb"])

        if tile.image is None:
            return QCResult(
                asset_id=tile.asset_id,
                asset_type=tile.asset_type,
                source=tile.source,
                status="fail_no_image",
                valid_ratio=0.0,
                mean_value=0.0,
                modalities=modalities,
                checks_failed=["no_image"],
                tile=tile,
            )

        image = tile.image
        passed_checks = []
        failed_checks = []

        # Check 1: valid pixels
        valid_ok, valid_ratio = _check_valid_pixels(image, self.min_valid_ratio)
        (passed_checks if valid_ok else failed_checks).append("valid_pixels")

        # Check 2: edge artifacts
        edge_ok = _check_edge_artifacts(image, self.edge_margin_px)
        (passed_checks if edge_ok else failed_checks).append("edge_artifacts")

        # Check 3: value range (modality-aware)
        range_ok, mean_val = _check_value_range(image, modalities)
        (passed_checks if range_ok else failed_checks).append("value_range")

        status = f"fail_{failed_checks[0]}" if failed_checks else "pass"

        return QCResult(
            asset_id=tile.asset_id,
            asset_type=tile.asset_type,
            source=tile.source,
            status=status,
            valid_ratio=valid_ratio,
            mean_value=mean_val,
            modalities=modalities,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            tile=tile,
        )

    def check_all(self, tiles: List[TileResult],
                  max_workers: int = 8) -> List["QCResult"]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results   = [None] * len(tiles)
        lock      = __import__("threading").Lock()
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