"""
decisions.py
------------
Reads the per-reviewer decision logs and works out what each reviewer should
see next.

Each reviewer appends JSON lines to data/label_review/decisions_<reviewer>.jsonl,
so two people reviewing in parallel never touch the same file and their logs
merge in git without conflicts. A line is

    {"asset_id", "reviewer", "decision", "label", "note", "at"}

where decision is "accept", "relabel" or "unsure", and label is the class the
reviewer settled on (the OSM label for an accept, "" for unsure). A later line
for the same reviewer and asset replaces an earlier one, which is how undo and
second thoughts work without rewriting the file.

With one reviewer (the default), any accept or relabel settles a tile, and a
tile marked unsure comes back to that reviewer as a revisit.

With second_opinion=True, the two-person rule applies instead: a tile one
reviewer accepted is done, a tile someone relabelled or marked unsure goes to
a second reviewer, and it is done once two reviewers agree on a label.
Anything else stays open for adjudication.
"""

import glob
import json
import os
import re
from datetime import datetime, timezone

from ..paths import REVIEW_DIR

DECISIONS = ("accept", "relabel", "unsure")


def log_path(reviewer: str) -> str:
    if not re.fullmatch(r"[a-z0-9_-]+", reviewer):
        raise ValueError("reviewer names are lowercase letters, digits, - and _")
    return str(REVIEW_DIR / f"decisions_{reviewer}.jsonl")


def append(reviewer: str, asset_id: str, decision: str, label: str, note: str = "") -> dict:
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    entry = {
        "asset_id": asset_id, "reviewer": reviewer, "decision": decision,
        "label": label if decision != "unsure" else "", "note": note,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(REVIEW_DIR, exist_ok=True)
    with open(log_path(reviewer), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load() -> dict:
    """asset_id -> {reviewer: latest entry}, across every reviewer's log."""
    latest = {}
    for path in sorted(glob.glob(str(REVIEW_DIR / "decisions_*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    latest.setdefault(e["asset_id"], {})[e["reviewer"]] = e
    return latest


def status(entries: dict, second_opinion: bool = False) -> str:
    """'open', 'done', 'unsure', 'needs_second' or 'disputed' for one asset."""
    if not entries:
        return "open"
    labels = [e["label"] for e in entries.values() if e["decision"] != "unsure"]
    if not second_opinion:
        if not labels:
            return "unsure"
        return "done" if len(set(labels)) == 1 else "disputed"
    if any(e["decision"] == "accept" for e in entries.values()) and len(entries) == 1:
        return "done"
    if len(entries) == 1:
        return "needs_second"
    # two or more reviewers: done once at least two agree and none disagree.
    if len(labels) >= 2 and len(set(labels)) == 1:
        return "done"
    return "disputed"


def final_label(entries: dict, second_opinion: bool = False) -> str:
    """the agreed label for a done asset, else ''."""
    if status(entries, second_opinion) != "done":
        return ""
    return next(e["label"] for e in entries.values() if e["decision"] != "unsure")


def wants(reviewer: str, entries: dict, second_opinion: bool = False) -> str:
    """which pass this asset belongs to for this reviewer, or '' if none."""
    s = status(entries, second_opinion)
    if s == "open":
        return "first"
    if not second_opinion:
        # the reviewer's own unsure tiles come back to them.
        return "revisit" if s == "unsure" and reviewer in entries else ""
    if reviewer in entries:
        return ""
    if s in ("needs_second", "disputed"):
        return "second"
    return ""
