"""
triage.py
---------
Confidence triage for infrastructure asset tiles.

Evaluates each tile against its weak label and assigns a confidence level:
  - "high"         → accept automatically, send to training corpus
  - "low"          → flag for human review
  - "contradiction" → reject or send for re-labeling

Two implementations are provided:
  1. RuleBasedTriager   — fast, deterministic, no API needed
                          uses image statistics + spatial checks
  2. AgentTriager       — uses an LLM with tool-calling to make
                          more nuanced decisions (requires Anthropic API)

Both implement the same interface so they are interchangeable.
Start with RuleBasedTriager, swap to AgentTriager when ready.

Usage:
    from triage import RuleBasedTriager, TriageResult

    triager = RuleBasedTriager()
    results = triager.triage_all(tiles, df)   # tiles from qc.py, df from sources
    accepted = triager.filter_accepted(results)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from imagery import TileResult


# ---------------------------------------------------------------------------
# Triage result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TriageResult:
    """
    Holds the triage outcome for one tile.

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
# Shared signal extraction
# ---------------------------------------------------------------------------

def _extract_signals(tile: TileResult) -> dict:
    """
    Extracts diagnostic signals from a tile image for use in triage decisions.
    Returns a dict of named numeric values.
    """
    img = tile.image
    if img is None:
        return {}

    # Brightness statistics per band
    mean_brightness  = float(img.mean())
    std_brightness   = float(img.std())

    # Spectral ratio — R/G ratio can hint at vegetation vs industrial surface
    r_band = img[0].astype(float)
    g_band = img[1].astype(float)
    b_band = img[2].astype(float)

    # avoid division by zero
    rg_ratio = float(np.mean(r_band / np.where(g_band > 0, g_band, 1)))
    rb_ratio = float(np.mean(r_band / np.where(b_band > 0, b_band, 1)))

    # Texture — standard deviation of local differences (proxy for structure)
    # High texture = structured/built environment, low = homogeneous (field/water)
    dx = np.diff(img[0].astype(float), axis=1)
    dy = np.diff(img[0].astype(float), axis=0)
    texture = float(np.sqrt(dx ** 2).mean() + np.sqrt(dy ** 2).mean())

    # Valid pixel coverage
    valid_ratio = float(np.any(img > 0, axis=0).mean())

    # Center crop brightness vs full image — asset should be near center
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
# Rule-based triage
# ---------------------------------------------------------------------------

# Per-asset-type rules — thresholds informed by visual characteristics.
# These are starting points; tune based on observed signal distributions.
# Format: dict of signal_name -> (min, max) acceptable range.
# A signal outside its range contributes to low/contradiction confidence.

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

# Default rules for asset types not explicitly listed
DEFAULT_RULES = {
    "mean_brightness": (15, 220),
    "valid_ratio":     (0.8, 1.0),
}


class RuleBasedTriager:
    """
    Rule-based confidence triage using image statistics.

    For each tile, extracts diagnostic signals and checks them against
    per-asset-type thresholds. The number of signals outside acceptable
    ranges determines the confidence level.

    Parameters
    ----------
    contradiction_threshold : int
        Number of failed signal checks before classifying as contradiction.
        Default 3 — requires multiple signals to be off before rejecting.
    low_threshold : int
        Number of failed checks before classifying as low confidence.
        Default 1.
    """

    def __init__(
        self,
        contradiction_threshold: int = 3,
        low_threshold: int = 1,
    ):
        self.contradiction_threshold = contradiction_threshold
        self.low_threshold           = low_threshold

    def triage_tile(self, tile: TileResult) -> TriageResult:
        """Triages a single tile."""
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

    def triage_all(self, tiles: List[TileResult]) -> List[TriageResult]:
        """Triages a list of tiles. Accepts output of qc.filter_ok()."""
        results = [self.triage_tile(t) for t in tiles]
        high  = sum(1 for r in results if r.confidence == "high")
        low   = sum(1 for r in results if r.confidence == "low")
        contr = sum(1 for r in results if r.confidence == "contradiction")
        print(f"Triage complete: {high} high / {low} low / {contr} contradiction "
              f"({len(results)} total)")
        return results

    def filter_accepted(self, results: List[TriageResult]) -> List[TileResult]:
        """Returns tiles with high confidence — ready for dataset assembly."""
        return [r.tile for r in results if r.accepted and r.tile is not None]

    def filter_review(self, results: List[TriageResult]) -> List[TriageResult]:
        """Returns tiles flagged for human review."""
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


# ---------------------------------------------------------------------------
# Agentic triage (stub — wire up Anthropic API when ready)
# ---------------------------------------------------------------------------

class AgentTriager:
    """
    Agentic confidence triage using an LLM with tool-calling.

    The agent receives a tile's image statistics and claimed label, then
    calls tools sequentially to build up evidence before routing:

      check_asset_visibility(signals)
          → is the image consistent with a real infrastructure asset?
          → uses texture, brightness, valid_ratio
          → if not visible: contradiction

      compare_to_reference(signals, asset_type)
          → does the image match expected characteristics for this asset type?
          → uses asset-type-specific signal expectations
          → if strong match: high confidence
          → if moderate: check geometry

      check_geometry_alignment(signals)
          → is the asset centered in the tile?
          → uses center_vs_full brightness ratio
          → if off-center: low confidence, flag for human

    The agent decides which tools to call based on what it finds —
    it does not run all three on every tile. This is the agentic behavior.

    Backend options (set via `backend` parameter):
      "ollama"    — local Ollama instance (free, good for dev)
                    requires: ollama running + a tool-capable model pulled
                    e.g. `ollama pull llama3.1` or `ollama pull qwen2.5`
      "anthropic" — Anthropic API via OpenAI-compatible endpoint
                    requires: Anthropic API key
      "openai"    — OpenAI API
                    requires: OpenAI API key

    The backend swap is just base_url + api_key + model_name —
    the tool-calling loop is identical across all three.

    Parameters
    ----------
    backend    : str  — "ollama", "anthropic", or "openai"
    api_key    : str  — API key (not needed for ollama)
    model      : str  — model name (defaults per backend below)
    base_url   : str  — override API endpoint (useful for custom deployments)
    """

    BACKEND_DEFAULTS = {
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key":  "ollama",           # ollama ignores this but openai client requires it
            "model":    "llama3.1",         # or "qwen2.5", "mistral" — any tool-capable model
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com/v1",
            "api_key":  None,               # must be provided
            "model":    "claude-sonnet-4-20250514",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key":  None,               # must be provided
            "model":    "gpt-4o-mini",
        },
    }

    # Tool definitions in OpenAI format — identical across all backends
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "check_asset_visibility",
                "description": (
                    "Check whether the image tile contains a visible, "
                    "recognizable infrastructure asset. Uses texture and "
                    "brightness signals. Returns a visibility assessment."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "texture": {
                            "type": "number",
                            "description": "Texture score from the tile (higher = more structure)"
                        },
                        "mean_brightness": {
                            "type": "number",
                            "description": "Mean pixel brightness 0-255"
                        },
                        "valid_ratio": {
                            "type": "number",
                            "description": "Fraction of non-zero pixels 0-1"
                        },
                    },
                    "required": ["texture", "mean_brightness", "valid_ratio"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_to_reference",
                "description": (
                    "Compare tile signals against expected characteristics "
                    "for this asset type. Returns similarity assessment: "
                    "'strong', 'moderate', or 'weak'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asset_type": {
                            "type": "string",
                            "description": "The claimed asset type label"
                        },
                        "texture": {"type": "number"},
                        "mean_brightness": {"type": "number"},
                        "rg_ratio": {
                            "type": "number",
                            "description": "Red/green ratio — low for vegetation, ~1 for built"
                        },
                        "std_brightness": {
                            "type": "number",
                            "description": "Brightness standard deviation — higher for varied surfaces"
                        },
                    },
                    "required": ["asset_type", "texture", "mean_brightness",
                                 "rg_ratio", "std_brightness"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_geometry_alignment",
                "description": (
                    "Check whether the asset appears centered in the tile "
                    "or is clipped to one edge (suggesting a coordinate offset). "
                    "Returns 'centered' or 'off_center'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "center_vs_full": {
                            "type": "number",
                            "description": (
                                "Ratio of center crop brightness to full image brightness. "
                                "Values near 1.0 suggest the asset is centered."
                            )
                        },
                    },
                    "required": ["center_vs_full"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_triage_decision",
                "description": (
                    "Submit the final triage decision for this tile. "
                    "Call this once you have gathered enough evidence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "low", "contradiction"],
                            "description": (
                                "high: accept for training corpus. "
                                "low: flag for human review. "
                                "contradiction: reject or re-label."
                            )
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief explanation of the decision"
                        },
                    },
                    "required": ["confidence", "reason"],
                },
            },
        },
    ]

    def __init__(
        self,
        backend: str = "ollama",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tool_calls: int = 6,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "AgentTriager requires the openai package. "
                "Install with: pip install openai"
            )

        if backend not in self.BACKEND_DEFAULTS:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from: {list(self.BACKEND_DEFAULTS.keys())}"
            )

        defaults = self.BACKEND_DEFAULTS[backend]
        self.backend        = backend
        self.model          = model    or defaults["model"]
        self.base_url       = base_url or defaults["base_url"]
        self.api_key        = api_key  or defaults["api_key"]
        self.max_tool_calls = max_tool_calls

        if self.api_key is None:
            raise ValueError(
                f"api_key is required for backend '{backend}'. "
                f"Pass it as AgentTriager(backend='{backend}', api_key='...')"
            )

        from openai import OpenAI
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        print(f"AgentTriager initialized: backend={backend}, model={self.model}")

    # ------------------------------------------------------------------
    # Tool implementations — these run locally, not in the LLM
    # ------------------------------------------------------------------

    def _tool_check_visibility(self, args: dict) -> str:
        texture    = args.get("texture", 0)
        brightness = args.get("mean_brightness", 0)
        valid      = args.get("valid_ratio", 0)

        if valid < 0.7:
            return "NOT_VISIBLE: too many missing pixels (valid_ratio={:.2f})".format(valid)
        if brightness < 10:
            return "NOT_VISIBLE: image too dark (mean_brightness={:.1f})".format(brightness)
        if brightness > 230:
            return "NOT_VISIBLE: image too bright, possible cloud (mean_brightness={:.1f})".format(brightness)
        if texture < 2:
            return "UNCERTAIN: very low texture — may be empty field or water (texture={:.1f})".format(texture)
        return "VISIBLE: asset appears present (texture={:.1f}, brightness={:.1f})".format(
            texture, brightness
        )

    def _tool_compare_reference(self, args: dict) -> str:
        asset_type = args.get("asset_type", "")
        texture    = args.get("texture", 0)
        brightness = args.get("mean_brightness", 0)
        rg_ratio   = args.get("rg_ratio", 1.0)
        std_bright = args.get("std_brightness", 0)

        rules = ASSET_TYPE_RULES.get(asset_type, DEFAULT_RULES)
        failures = []

        if "texture" in rules:
            lo, hi = rules["texture"]
            if not (lo <= texture <= hi):
                failures.append(f"texture={texture:.1f} (expected {lo}-{hi})")
        if "mean_brightness" in rules:
            lo, hi = rules["mean_brightness"]
            if not (lo <= brightness <= hi):
                failures.append(f"brightness={brightness:.1f} (expected {lo}-{hi})")
        if "rg_ratio" in rules:
            lo, hi = rules["rg_ratio"]
            if not (lo <= rg_ratio <= hi):
                failures.append(f"rg_ratio={rg_ratio:.2f} (expected {lo}-{hi})")

        if not failures:
            return "STRONG_MATCH: all signals consistent with {}".format(asset_type)
        elif len(failures) == 1:
            return "MODERATE_MATCH: one signal borderline — {}".format(failures[0])
        else:
            return "WEAK_MATCH: multiple signals inconsistent — {}".format("; ".join(failures))

    def _tool_check_geometry(self, args: dict) -> str:
        ratio = args.get("center_vs_full", 1.0)
        if 0.7 <= ratio <= 1.4:
            return "CENTERED: asset appears centered in tile (center_vs_full={:.2f})".format(ratio)
        return "OFF_CENTER: asset may be clipped to edge (center_vs_full={:.2f})".format(ratio)

    def _dispatch_tool(self, tool_name: str, args: dict) -> str:
        """Routes a tool call to its local implementation."""
        if tool_name == "check_asset_visibility":
            return self._tool_check_visibility(args)
        elif tool_name == "compare_to_reference":
            return self._tool_compare_reference(args)
        elif tool_name == "check_geometry_alignment":
            return self._tool_check_geometry(args)
        elif tool_name == "submit_triage_decision":
            return "DECISION_SUBMITTED"
        else:
            return f"ERROR: unknown tool {tool_name}"

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    def triage_tile(self, tile: TileResult) -> TriageResult:
        """
        Runs the agentic triage loop for a single tile.

        The LLM decides which tools to call and in what order.
        The loop continues until the model calls submit_triage_decision
        or until max_tool_calls is reached.
        """
        import json

        if tile.image is None:
            return TriageResult(
                asset_id=tile.asset_id, asset_type=tile.asset_type,
                source=tile.source, confidence="contradiction",
                flag_for_human=False,
                reason="No image available",
                tile=tile,
            )

        signals = _extract_signals(tile)

        system_prompt = (
            "You are a satellite imagery quality control agent for infrastructure assets. "
            "You will assess whether an imagery tile correctly represents the labeled "
            "infrastructure asset type. Use the available tools to gather evidence, "
            "then call submit_triage_decision with your final verdict. "
            "Be efficient — only call the tools you need to make a confident decision."
        )

        user_message = (
            f"Assess this imagery tile:\n"
            f"  Asset type (claimed label): {tile.asset_type}\n"
            f"  Imagery source: {tile.source}\n"
            f"  Image signals: {json.dumps({k: round(v, 3) for k, v in signals.items()}, indent=2)}\n\n"
            f"Use the tools to assess visibility, label match, and geometry alignment, "
            f"then submit your triage decision."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        final_confidence = "low"
        final_reason     = "max tool calls reached without decision"

        for _ in range(self.max_tool_calls):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.TOOLS,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            messages.append(msg)

            # No tool call — model gave a text response instead of deciding
            if not msg.tool_calls:
                final_reason = msg.content or "no decision reached"
                break

            # Process each tool call
            decision_submitted = False
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except Exception:
                    args = {}

                result = self._dispatch_tool(name, args)

                # Capture the decision if submitted
                if name == "submit_triage_decision":
                    final_confidence = args.get("confidence", "low")
                    final_reason     = args.get("reason", "")
                    decision_submitted = True

                # Feed tool result back to the model
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result,
                })

            if decision_submitted:
                break

        flag_for_human = final_confidence == "low"

        return TriageResult(
            asset_id=tile.asset_id, asset_type=tile.asset_type,
            source=tile.source, confidence=final_confidence,
            flag_for_human=flag_for_human,
            reason=final_reason, signals=signals, tile=tile,
        )

    def triage_all(self, tiles: List[TileResult]) -> List[TriageResult]:
        """Triages a list of tiles using the agentic loop."""
        results = []
        total   = len(tiles)
        for i, tile in enumerate(tiles):
            result = self.triage_tile(tile)
            results.append(result)
            if (i + 1) % 5 == 0 or (i + 1) == total:
                high  = sum(1 for r in results if r.confidence == "high")
                low   = sum(1 for r in results if r.confidence == "low")
                contr = sum(1 for r in results if r.confidence == "contradiction")
                print(f"  [{i+1}/{total}] high={high} low={low} contradiction={contr}")
        return results

    def filter_accepted(self, results: List[TriageResult]) -> List[TileResult]:
        return [r.tile for r in results if r.accepted and r.tile is not None]

    def filter_review(self, results: List[TriageResult]) -> List[TriageResult]:
        return [r for r in results if r.flag_for_human]

    def summarize(self, results: List[TriageResult]) -> pd.DataFrame:
        """Same interface as RuleBasedTriager.summarize()."""
        rows = []
        for r in results:
            rows.append({
                "asset_id":       r.asset_id,
                "asset_type":     r.asset_type,
                "source":         r.source,
                "confidence":     r.confidence,
                "flag_for_human": r.flag_for_human,
                "reason":         r.reason,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from sources import GeoFabrikSource
    from imagery import ImageryFetcher
    from qc import QualityChecker

    PBF_PATH  = "data/pbf/maine-latest.osm.pbf"
    INPUT_CSV = "data/maine_all_assets.csv"

    # Load assets
    if os.path.exists(INPUT_CSV):
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} assets")
    else:
        src = GeoFabrikSource(PBF_PATH, min_confidence="medium")
        df  = src.extract_all()

    # Sample
    df_sample = (
        df.groupby("asset_type", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), 2), random_state=42))
          .reset_index(drop=True)
    )

    # Fetch
    print("\nFetching tiles...")
    fetcher = ImageryFetcher(buffer_m=150, sources=["sentinel2"])
    tiles   = fetcher.fetch_all(df_sample)

    # QC
    print("\nRunning QC...")
    checker    = QualityChecker()
    qc_results = checker.check_all(tiles)
    clean      = checker.filter_ok(qc_results)

    # Triage
    print("\nRunning triage...")
    triager        = RuleBasedTriager()
    triage_results = triager.triage_all(clean)

    print("\nTriage summary:")
    summary = triager.summarize(triage_results)
    print(summary[["asset_type", "source", "confidence",
                   "flag_for_human", "reason"]].to_string(index=False))

    accepted = triager.filter_accepted(triage_results)
    review   = triager.filter_review(triage_results)
    print(f"\n{len(accepted)} tiles accepted for training corpus")
    print(f"{len(review)} tiles flagged for human review")