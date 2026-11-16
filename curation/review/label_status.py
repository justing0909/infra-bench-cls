"""
label_status.py
---------------
One row per benchmark tile saying what its label is after review and who
checked it, so every tile has a status and OSM is a fallback, not ground
truth:

  original_label   the OSM-derived label in the released benchmark
  screened_label   the label after review (may be not_infrastructure)
  checked_by       the most trustworthy check the tile has:
                     human            the person's own decision (review log)
                     human_unsure     the person could not tell; OSM kept
                     official_source  a strict match to an official register
                                      auto-accepted it (auto_accept.py)
                     agent            the calibrated investigator agent,
                                      confidence at or above THRESHOLD
                     agent_unsure     the agent below THRESHOLD; OSM kept
                     unscreened       nothing yet; OSM kept
  confidence       the agent's self-reported confidence (agent rows only)
  detail           the person's note, the matching source, or the agent's reason

THRESHOLD is the investigator's confidence threshold chosen on the
calibration half of the spatially blocked split (calib_s2_v1, after
adjudication): about 45% held-out coverage at ~1-2% error. Change it only
with a new calibration.

    python -m curation.review.label_status
"""

import json

import pandas as pd

from ..paths import REVIEW_DIR, REVIEW_QUEUE
from . import decisions
from .two_agent import RUNS_DIR, _role_files

THRESHOLD = 0.65
SCREEN_RUN = "screen_s2_v1"
REVIEWER = "justin"
OUT = REVIEW_DIR / "label_status_v1.csv"
COLUMNS = ["asset_id", "region", "split", "original_label", "screened_label", "checked_by",
           "confidence", "detail"]


def _agent_rows(run: str) -> dict:
    """the investigator's decision per tile across all finished batches of a run;
    batches not yet run are skipped, so the file can be built mid-run."""
    out = {}
    folder = RUNS_DIR / run / "investigator"
    for d in sorted(folder.glob("batch_*")) if folder.exists() else []:
        if (d / "decisions.json").exists():
            for r in json.loads((d / "decisions.json").read_text(encoding="utf-8")):
                out[r["id"]] = r
    return out


def build(run: str = SCREEN_RUN, threshold: float = THRESHOLD) -> pd.DataFrame:
    q = pd.read_csv(REVIEW_QUEUE, keep_default_na=False)
    latest = decisions.load()
    agent = _agent_rows(run)
    rows = []
    for t in q.itertuples():
        osm, entries = t.asset_type, latest.get(t.asset_id, {})
        row = {"asset_id": t.asset_id, "region": t.region, "split": t.split, "original_label": osm,
               "screened_label": osm, "checked_by": "unscreened", "confidence": "", "detail": ""}
        if REVIEWER in entries:
            e = entries[REVIEWER]
            if e["decision"] == "unsure":
                row.update(checked_by="human_unsure", detail=e.get("note", ""))
            else:
                row.update(checked_by="human", detail=e.get("note", ""),
                           screened_label=osm if e["decision"] == "accept" else e["label"])
        elif "source" in entries:
            row.update(checked_by="official_source", detail=entries["source"].get("note", ""))
        elif t.asset_id in agent:
            a = agent[t.asset_id]
            confident = a["action"] != "request_review" and a["confidence"] >= threshold
            label = osm if a["action"] == "confirm" else a.get("label") or osm
            row.update(checked_by="agent" if confident else "agent_unsure", confidence=a["confidence"],
                       detail=a.get("reason", ""), screened_label=label if confident else osm)
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def main():
    df = build()
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df):,} tiles to {OUT}")
    print(df.checked_by.value_counts().to_string())
    changed = df[df.screened_label != df.original_label]
    print(f"screened label differs from OSM on {len(changed):,} tiles "
          f"({changed.checked_by.value_counts().to_dict()})")


if __name__ == "__main__":
    main()
