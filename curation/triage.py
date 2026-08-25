"""
triage.py
---------
confidence triage for infrastructure asset tiles.

evaluates each tile against its weak label and assigns a confidence level:
  - "high"          → accept automatically, send to the training corpus
  - "low"           → flag for human review
  - "contradiction" → reject or send for re-labeling

RuleBasedTriager is deterministic and needs no network: it derives image
statistics (brightness, texture, spectral ratios, centering) and counts how
many fall outside the per-asset-type expected ranges. an LLM tool-calling
variant lived here during development but was never used for a shipped
dataset; see git history if it is ever needed again.

Usage:
    from curation.triage import RuleBasedTriager, TriageResult

    triager = RuleBasedTriager()
    results = triager.triage_all(tiles, df)   # tiles from qc.py, df from sources
    accepted = triager.filter_accepted(results)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .helpers.tile_types import TileResult


# ---------------------------------------------------------------------------
# triage result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    """
    holds the triage outcome for one tile.

    Attributes
    ----------
    asset_id         : str
    asset_type       : str
    source           : str
    confidence       : str   — "high", "low", "contradiction"
    flag_for_human   : bool
    reason           : str   — explanation of the decision
    signals          : dict  — diagnostic values used in the decision
    tile             : TileResult
    """
    asset_id       : str
    asset_type     : str
    source         : str
    confidence     : str
    flag_for_human : bool
    reason         : str
    signals        : Dict    = field(default_factory=dict)
    tile           : Optional[TileResult] = None

    @property
    def accepted(self) -> bool:
        return self.confidence == "high"

    @property
    def rejected(self) -> bool:
        return self.confidence == "contradiction"


# ---------------------------------------------------------------------------
# shared signal extraction
# ---------------------------------------------------------------------------

def _extract_signals(tile: TileResult) -> dict:
    """
    extracts diagnostic signals from a tile image for use in triage decisions.
    returns a dict of named numeric values.
    """
    img = tile.image
    if img is None:
        return {}

    # brightness statistics per band
    mean_brightness  = float(img.mean())
    std_brightness   = float(img.std())

    # spectral ratio — R/G ratio can hint at vegetation vs industrial surface
    r_band = img[0].astype(float)
    g_band = img[1].astype(float)
    b_band = img[2].astype(float)

    # avoid division by zero
    rg_ratio = float(np.mean(r_band / np.where(g_band > 0, g_band, 1)))
    rb_ratio = float(np.mean(r_band / np.where(b_band > 0, b_band, 1)))

    # texture — standard deviation of local differences (proxy for structure)
    # high texture = structured/built environment, low = homogeneous (field/water)
    dx = np.diff(img[0].astype(float), axis=1)
    dy = np.diff(img[0].astype(float), axis=0)
    texture = float(np.sqrt(dx ** 2).mean() + np.sqrt(dy ** 2).mean())

    # valid pixel coverage
    valid_ratio = float(np.any(img > 0, axis=0).mean())

    # center crop brightness vs full image — asset should be near center
    h, w = img.shape[1], img.shape[2]
    cy, cx = h // 2, w // 2
    margin = min(h, w) // 4
    center = img[:, cy - margin:cy + margin, cx - margin:cx + margin]
    center_brightness = float(center.mean()) if center.size > 0 else 0.0

    return {
        "mean_brightness":   mean_brightness,
        "std_brightness":    std_brightness,
        "rg_ratio":          rg_ratio,
        "rb_ratio":          rb_ratio,
        "texture":           texture,
        "valid_ratio":       valid_ratio,
        "center_brightness": center_brightness,
        "center_vs_full":    center_brightness / max(mean_brightness, 1),
    }


# ---------------------------------------------------------------------------
# rule-based triage
# ---------------------------------------------------------------------------

# per-asset-type rules — thresholds informed by visual characteristics.
# these are starting points; tune based on observed signal distributions.
# format: dict of signal_name -> (min, max) acceptable range.
# a signal outside its range contributes to low/contradiction confidence.

ASSET_TYPE_RULES = {
    "energy.transmission.substation": {
        "texture":        (8, 9999),    # substations are structured
        "mean_brightness": (20, 210),
    },
    "energy.distribution.substation": {
        "texture":        (6, 9999),
        "mean_brightness": (20, 210),
    },
    "energy.distribution.substation_untyped": {
        "texture":        (5, 9999),
        "mean_brightness": (20, 210),
    },
    "energy.generation.power_plant": {
        "texture":        (5, 9999),
        "mean_brightness": (20, 210),
    },
    "energy.generation.solar_farm": {
        "mean_brightness": (10, 180),   # solar panels are dark-ish
        "rg_ratio":        (0.5, 1.5),  # not strongly red or green
    },
    "energy.generation.wind_farm": {
        "texture":        (2, 9999),    # some structure expected
        "mean_brightness": (20, 220),
    },
    "energy.generation.generator": {
        "texture":        (3, 9999),
        "mean_brightness": (20, 210),
    },
}

# default rules for asset types not explicitly listed
DEFAULT_RULES = {
    "mean_brightness": (15, 220),
    "valid_ratio":     (0.8, 1.0),
}


class RuleBasedTriager:
    """
    rule-based confidence triage using image statistics.

    for each tile, extracts diagnostic signals and checks them against
    per-asset-type thresholds. the number of signals outside acceptable
    ranges determines the confidence level.

    Parameters
    ----------
    contradiction_threshold : int
        number of failed signal checks before classifying as contradiction.
        default 3 — requires multiple signals to be off before rejecting.
    low_threshold : int
        number of failed checks before classifying as low confidence.
        default 1.
    """

    def __init__(
        self,
        contradiction_threshold: int = 3,
        low_threshold: int = 1,
    ):
        self.contradiction_threshold = contradiction_threshold
        self.low_threshold           = low_threshold

    def triage_tile(self, tile: TileResult) -> TriageResult:
        """triages a single tile."""
        if tile.image is None:
            return TriageResult(
                asset_id=tile.asset_id, asset_type=tile.asset_type,
                source=tile.source, confidence="contradiction",
                flag_for_human=False,
                reason="No image available",
                tile=tile,
            )

        signals  = _extract_signals(tile)
        rules    = ASSET_TYPE_RULES.get(tile.asset_type, DEFAULT_RULES)
        failures = []

        for signal_name, (lo, hi) in rules.items():
            val = signals.get(signal_name)
            if val is not None and not (lo <= val <= hi):
                failures.append(
                    f"{signal_name}={val:.2f} outside [{lo}, {hi}]"
                )

        n_failures = len(failures)

        if n_failures >= self.contradiction_threshold:
            confidence     = "contradiction"
            flag_for_human = False
            reason = f"{n_failures} signals outside expected range: " \
                     f"{'; '.join(failures)}"
        elif n_failures >= self.low_threshold:
            confidence     = "low"
            flag_for_human = True
            reason = f"{n_failures} signal(s) borderline: " \
                     f"{'; '.join(failures)}"
        else:
            confidence     = "high"
            flag_for_human = False
            reason         = "All signals within expected range"

        return TriageResult(
            asset_id=tile.asset_id, asset_type=tile.asset_type,
            source=tile.source, confidence=confidence,
            flag_for_human=flag_for_human,
            reason=reason, signals=signals, tile=tile,
        )

    def triage_all(self, tiles: List[TileResult],
                   max_workers: int = 8) -> List[TriageResult]:
        """triages a list of tiles in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        results   = [None] * len(tiles)
        lock      = threading.Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.triage_tile, tile): i
                       for i, tile in enumerate(tiles)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                with lock:
                    completed += 1
                    if completed % 1000 == 0 or completed == len(tiles):
                        high  = sum(1 for r in results if r and r.confidence == "high")
                        low   = sum(1 for r in results if r and r.confidence == "low")
                        contr = sum(1 for r in results if r and r.confidence == "contradiction")
                        print(f"  Triage [{completed}/{len(tiles)}] "
                              f"high={high} low={low} contradiction={contr}")

        high  = sum(1 for r in results if r.confidence == "high")
        low   = sum(1 for r in results if r.confidence == "low")
        contr = sum(1 for r in results if r.confidence == "contradiction")
        print(f"Triage complete: {high} high / {low} low / {contr} contradiction "
              f"({len(results)} total)")
        return results

    def filter_accepted(self, results: List[TriageResult]) -> List[TileResult]:
        """returns tiles with high confidence — ready for dataset assembly."""
        return [r.tile for r in results if r.accepted and r.tile is not None]

    def filter_review(self, results: List[TriageResult]) -> List[TriageResult]:
        """returns tiles flagged for human review."""
        return [r for r in results if r.flag_for_human]

    def summarize(self, results: List[TriageResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append({
                "asset_id":       r.asset_id,
                "asset_type":     r.asset_type,
                "source":         r.source,
                "confidence":     r.confidence,
                "flag_for_human": r.flag_for_human,
                "reason":         r.reason,
                **{f"sig_{k}": round(v, 3)
                   for k, v in r.signals.items()},
            })
        return pd.DataFrame(rows)

