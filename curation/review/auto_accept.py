"""
auto_accept.py
--------------
Passes the most obvious tiles without a human look, openly and with an audit.

A tile is eligible when a secondary source independently confirms its OSM
label under that source's strict rule (the `auto_ok` column each source
writes):

  hifld_lines   OSM transmission substation, HIFLD lines of 100 kV or more
                not traced from OSM end there, and a line-end name matches
  eia_plants    an EIA plant of the matching kind within 300 m whose name
                matches, with no conflicting plant nearby
  registers     a NARN or NaPTAN station within 300 m, or an FAA airport within
                2 km, whose name matches

A fixed random AUDIT_SHARE of eligible tiles (at least one per source) is held
back and left for human review, so the paper can report how often a strict
source match agrees with a person. The rest are accepted into their own log,
decisions_source.jsonl, under the reviewer "source", with the rule and the
evidence in the note. Tiles a person has already decided are never touched,
and a later human decision on an auto-accepted tile shows up as a
disagreement rather than being overwritten.

    python -m curation.review.auto_accept            # apply
    python -m curation.review.auto_accept --report   # audit agreement so far
"""

import argparse
import zlib

import numpy as np
import pandas as pd

from ..paths import REVIEW_DIR, REVIEW_QUEUE
from . import decisions

REVIEWER = "source"
AUDIT_SHARE = 0.10
SEED = 0
AUDIT = REVIEW_DIR / "auto_accept_audit.csv"

# source csv -> (evidence text column, rule shown in the decision note)
SOURCES = {
    "hifld_lines.csv": ("hifld_text", "HIFLD lines >=100 kV end here and a line-end name matches"),
    "eia_plants.csv": ("eia_text", "EIA plant of the matching kind within 300 m, name matches"),
    "registers.csv": ("register_text", "official register entry nearby, name matches"),
}


def eligible() -> pd.DataFrame:
    rows = []
    for name, (text_col, rule) in SOURCES.items():
        path = REVIEW_DIR / "sources" / name
        if not path.exists():
            continue
        ev = pd.read_csv(path, keep_default_na=False)
        ev = ev[ev.auto_ok.astype(str) == "True"]
        source = ev["register"] if "register" in ev else name.removesuffix(".csv")
        rows.append(pd.DataFrame({"asset_id": ev.asset_id, "source": source,
                                  "rule": rule, "evidence": ev[text_col]}))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["asset_id", "source", "rule", "evidence"])
    # a tile confirmed by two sources counts once, under the first
    return out.drop_duplicates("asset_id").reset_index(drop=True)


def choose_audit(cands: pd.DataFrame, latest: dict) -> pd.Series:
    """the held-back sample, AUDIT_SHARE per source and at least one.

    It is persistent: tiles held back on an earlier run stay held back, and a
    source only gets new picks when it has grown (or is new), drawn from tiles
    nobody has decided yet. So adding a source never reshuffles the audit or
    pulls in a tile that was already accepted."""
    held = set()
    if AUDIT.exists():
        prev = pd.read_csv(AUDIT, keep_default_na=False)
        held = set(prev.loc[prev.audit_kind == "held_back", "asset_id"])
    keep = set(held & set(cands.asset_id))
    for source, g in cands.sort_values("asset_id").groupby("source"):
        want = max(1, round(AUDIT_SHARE * len(g))) - len(keep & set(g.asset_id))
        free = g[~g.asset_id.isin(keep) & g.asset_id.map(lambda a: not latest.get(a))].asset_id.to_numpy()
        if want > 0 and len(free):
            rng = np.random.default_rng([SEED, zlib.crc32(source.encode())])
            keep |= set(rng.choice(free, size=min(want, len(free)), replace=False))
    return cands.asset_id.isin(keep)


def apply() -> dict:
    queue = pd.read_csv(REVIEW_QUEUE, keep_default_na=False)
    labels = dict(zip(queue.asset_id, queue.asset_type))
    cands = eligible()
    cands = cands[cands.asset_id.isin(labels)]
    latest = decisions.load()
    cands["audit"] = choose_audit(cands, latest)
    # the audit is the held-back sample plus any eligible tile a person had
    # already decided: both are a human check on a strict source match.
    human = cands.asset_id.map(lambda a: any(r != REVIEWER for r in latest.get(a, {})))
    cands["audit_kind"] = np.where(cands.audit, "held_back", np.where(human, "already_reviewed", ""))
    cands.loc[cands.audit_kind != "", ["asset_id", "source", "audit_kind", "rule", "evidence"]].to_csv(
        AUDIT, index=False)

    counts = {"eligible": len(cands), "audit_held_back": int(cands.audit.sum()),
              "accepted": 0, "already_accepted": 0, "human_decided": 0}
    for c in cands[~cands.audit].itertuples():
        entries = latest.get(c.asset_id, {})
        if REVIEWER in entries:
            counts["already_accepted"] += 1
        elif entries:
            counts["human_decided"] += 1
        else:
            first = c.evidence.split(" | ")[0]
            decisions.append(REVIEWER, c.asset_id, "accept", labels[c.asset_id],
                             f"auto-accepted ({c.source}): {c.rule}. {first}")
            counts["accepted"] += 1
    return counts


def report() -> pd.DataFrame:
    """how people decided the held-back audit tiles."""
    audit = pd.read_csv(AUDIT, keep_default_na=False)
    queue = pd.read_csv(REVIEW_QUEUE, keep_default_na=False)
    labels = dict(zip(queue.asset_id, queue.asset_type))
    latest = decisions.load()
    rows = []
    for a in audit.itertuples():
        human = {r: e for r, e in latest.get(a.asset_id, {}).items() if r != REVIEWER}
        for e in human.values():
            rows.append({"source": a.source, "agrees": e["decision"] == "accept"
                         or e.get("label") == labels[a.asset_id], "decision": e["decision"]})
    df = pd.DataFrame(rows, columns=["source", "agrees", "decision"])
    summary = audit.groupby("source").size().rename("audit_tiles").to_frame()
    summary["reviewed"] = df.groupby("source").size()
    summary["agree"] = df.groupby("source").agrees.sum()
    summary = summary.apply(pd.to_numeric).fillna(0).astype(int)
    summary["agreement_%"] = (100 * summary.agree / summary.reviewed.where(summary.reviewed > 0)).round(1)
    return summary


def main():
    p = argparse.ArgumentParser(description="auto-accept strict source matches, with an audit")
    p.add_argument("--report", action="store_true", help="show audit agreement instead of applying")
    args = p.parse_args()
    if args.report:
        print(report().to_string())
    else:
        print(apply())
        print(f"audit tiles left for review: {AUDIT}")


if __name__ == "__main__":
    main()
