"""
agent_report.py
---------------
Scores the investigator-only agent run against the human review: raw accuracy
against trusting OSM, errors fixed and introduced, the effect of authoritative
source lines, a confidence threshold chosen on the calibration half of the
spatially blocked split and reported on the held-out half (by class and
region), and the NAIP ablation on the US tiles. Writes
investigator_scored.csv into the run.

    python -m curation.review.agent_report
"""
import json, glob, numpy as np, pandas as pd
from scipy.stats import beta
from curation.review import two_agent as t
R = t.RUNS_DIR
q = pd.read_csv("data/label_review/queue.csv", keep_default_na=False).set_index("asset_id")
d = pd.DataFrame([json.loads(l) for l in open("data/label_review/decisions_justin.jsonl", encoding="utf-8") if l.strip()]).sort_values("at").drop_duplicates("asset_id", keep="last").set_index("asset_id")
split = pd.read_csv(R / "calib_split_v1.csv").set_index("asset_id")
def load(run, role="investigator"):
    pk, rows = t._role_files(R / run / role)
    ev = {p["id"]: p.get("evidence", {}) for p in pk}
    return {r["id"]: r for r in rows}, ev
inv, ev = load("calib_s2_v1")
def truth(i):
    e = d.loc[i]; return q.loc[i, "asset_type"] if e.decision == "accept" else e.label
rows = []
unsure_now = {i for i in inv if d.loc[i, "decision"] == "unsure"}   # a re-check can turn a decision into unsure
for i, a in inv.items():
    if i in unsure_now:
        continue
    osm = q.loc[i, "asset_type"]
    said = osm if a["action"] == "confirm" else a.get("label", "")
    rows.append({"id": i, "osm": osm, "truth": truth(i), "said": said, "action": a["action"], "conf": a["confidence"],
                 "sources": ev[i]["sources"]["available"], "half": split.loc[i, "half"], "cls": split.loc[i, "cls"], "region": split.loc[i, "region"]})
r = pd.DataFrame(rows); r["ok"] = r.said == r.truth; r["osm_ok"] = r.osm == r.truth
def ci(k, n):
    return f"{100*(beta.ppf(.025,k,n-k+1) if k else 0):.0f}-{100*(beta.ppf(.975,k+1,n-k) if k<n else 1):.0f}%"
print(f"Sentinel investigator, all 705: accuracy {r.ok.mean():.1%} | trusting OSM: {r.osm_ok.mean():.1%}")
w = r[~r.osm_ok]
print(f"  OSM wrong on {len(w)} tiles: agent fixed {int((w.action.eq('relabel') & w.ok).sum())}, relabelled wrongly {int((w.action.eq('relabel') & ~w.ok).sum())}, confirmed the error {int(w.action.eq('confirm').sum())}")
print(f"  OSM right on {int(r.osm_ok.sum())} tiles: agent wrongly changed {int((r.osm_ok & r.action.eq('relabel')).sum())}")
print(f"  with authoritative source lines: {r[r.sources].ok.mean():.0%} of {int(r.sources.sum())} (OSM {r[r.sources].osm_ok.mean():.0%}) | without: {r[~r.sources].ok.mean():.0%} of {int((~r.sources).sum())} (OSM {r[~r.sources].osm_ok.mean():.0%})")
# single-agent threshold on the calibration half
cal, held = r[r.half == "calibration"], r[r.half == "heldout"]
print("\ncalibration sweep (auto = confidence >= t and not request_review):")
best = None
for th in np.arange(0.5, 1.0001, 0.05):
    s = cal[(cal.conf >= th) & cal.action.ne("request_review")]
    err = 1 - s.ok.mean() if len(s) else None
    print(f"  {th:.2f}: coverage {len(s)/len(cal):.0%} ({len(s)}), error {err:.1%}" if len(s) else f"  {th:.2f}: none")
    if best is None and len(s) and err <= 0.05: best = round(th, 2)
print("chosen threshold:", best)
h = held[(held.conf >= best) & held.action.ne("request_review")]
k = int((~h.ok).sum())
print(f"HELD-OUT at {best}: coverage {len(h)/len(held):.0%} ({len(h)} of {len(held)}), error {k/len(h):.1%} ({k} wrong, 95% CI {ci(k, len(h))})")
print(f"  for comparison, trusting OSM on the held-out half: error {1-held.osm_ok.mean():.1%}")
print("\nheld-out by class at the chosen threshold:")
for c, g in held.groupby("cls"):
    a = g[(g.conf >= best) & g.action.ne("request_review")]
    print(f"  {c.split('.')[-1]:<14} {c.split('.')[0]:<9} auto {len(a):>2}/{len(g):<3} wrong {int((~a.ok).sum())}")
print("\nheld-out by region:")
for c, g in held.groupby("region"):
    a = g[(g.conf >= best) & g.action.ne("request_review")]
    print(f"  {c:<18} auto {len(a):>2}/{len(g):<3} wrong {int((~a.ok).sum())}")
# NAIP ablation, investigator: same US tiles, Sentinel-only vs with NAIP
invn, _ = load("calib_naip_v1")
us = [i for i in invn if i in inv and i not in unsure_now]
acc = lambda dct, ids: np.mean([(q.loc[i, "asset_type"] if dct[i]["action"] == "confirm" else dct[i].get("label")) == truth(i) for i in ids])
print(f"\nNAIP ablation on {len(us)} US tiles, investigator accuracy: Sentinel-only {acc(inv, us):.0%} vs with NAIP {acc(invn, us):.0%}; trusting OSM {np.mean([q.loc[i,'asset_type']==truth(i) for i in us]):.0%}")
r.to_csv(R / "calib_s2_v1" / "investigator_scored.csv", index=False)
