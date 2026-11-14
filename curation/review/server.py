"""
server.py
---------
Local label-review page, an alternative to the VIA projects that writes the
same decisions log. Each tile shows the Sentinel-2 chip next to a
high-resolution satellite map, with Google Maps and Street View one key away,
and takes one keypress per decision. Serves on 127.0.0.1 only.

    python -m curation.review.server --reviewer justin
    # then open http://127.0.0.1:8765

Needs data/label_review/queue.csv (python -m curation.review.build_queue) and
the v1_1k tiles under data/curated_datasets/.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ..paths import DATA_DIR, REVIEW_DIR, REVIEW_QUEUE
from . import decisions
from .via import NOT_INFRA, chip_for

PAGE = Path(__file__).with_name("index.html")


class Review:
    def __init__(self, reviewer: str, second_opinion: bool):
        self.reviewer = reviewer
        self.second_opinion = second_opinion
        self.queue = pd.read_csv(REVIEW_QUEUE, keep_default_na=False)
        self.by_id = {r["asset_id"]: r for r in self.queue.to_dict("records")}
        self.classes = sorted(self.queue.asset_type.unique()) + [NOT_INFRA]
        audit = REVIEW_DIR / "auto_accept_audit.csv"
        self.audit = (set(pd.read_csv(audit).query("audit_kind == 'held_back'").asset_id)
                      if audit.exists() else set())
        sample = REVIEW_DIR / "class_sample.csv"
        draw = pd.read_csv(sample, keep_default_na=False) if sample.exists() else pd.DataFrame()
        self.sample = dict(draw[["asset_id", "hint"]].values) if len(draw) else {}
        # the sample pass walks each class in its random draw order, so any
        # prefix you review is itself a random sample and you can stop early.
        self.sample_pos = ({a: i for i, a in enumerate(draw.sort_values(["asset_type", "draw"]).asset_id)}
                           if len(draw) else {})

    def _view(self, mode: str, cls: str) -> pd.DataFrame:
        q = self.queue if not cls else self.queue[self.queue.asset_type == cls]
        if mode == "sample":
            q = q[q.asset_id.isin(self.sample_pos)]
            q = q.assign(order=q.asset_id.map(self.sample_pos)).sort_values("order")
        return q

    def _item(self, row: dict, entries: dict) -> dict:
        hint = self.sample.get(row["asset_id"], "")
        return {
            **row,
            "evidence": " | ".join(x for x in (hint, row.get("evidence", "")) if x),
            "status": decisions.status(entries, self.second_opinion),
            "mine": entries.get(self.reviewer),
            "others": [e for r, e in entries.items() if r != self.reviewer],
        }

    def footprint(self, asset_id: str) -> list:
        """the asset's mapped OSM outline as lat/lon rings, from the Overpass
        responses cached by evaluation/analysis/asset_footprint_share.py, so the
        page makes no live OSM requests (Overpass rate-limits per address)."""
        if not hasattr(self, "_footprints"):
            self._footprints = {}
            for path in sorted((DATA_DIR / "asset_footprints" / "overpass").glob("*.json")):
                for el in json.loads(path.read_text(encoding="utf-8"))["elements"]:
                    ways = [el] if el["type"] == "way" else [m for m in el.get("members", []) if m.get("type") == "way"]
                    rings = [[[p["lat"], p["lon"]] for p in w.get("geometry", [])] for w in ways]
                    self._footprints[f"osm_{el['type']}_{el['id']}"] = [r for r in rings if r]
        return self._footprints.get(asset_id, [])

    @staticmethod
    def arcgis_key() -> str:
        """an ArcGIS Location Platform API key, from ARCGIS_API_KEY or a local
        data/label_review/arcgis_key.txt (git-ignored). Esri World Imagery is only
        shown with one: using it inside a custom app needs an ArcGIS account."""
        key = os.environ.get("ARCGIS_API_KEY", "").strip()
        path = REVIEW_DIR / "arcgis_key.txt"
        if not key and path.exists():
            key = path.read_text(encoding="utf-8").strip()
        return key

    def naip_png(self, asset_id: str) -> bytes | None:
        """NAIP aerial imagery (USDA, public domain) for a US tile, cached."""
        from .hifld_lines import in_us
        from .two_agent import _naip
        row = self.by_id.get(asset_id)
        if row is None:
            return None
        lat, lon = float(row["lat"]), float(row["lon"])
        if not in_us(lat, lon):
            return None
        cache = REVIEW_DIR / "naip_cache" / f"{asset_id}.png"
        missing = cache.with_suffix(".none")
        if missing.exists():
            return None
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            if _naip(lat, lon, cache) is None:
                missing.touch()
                return None
        return cache.read_bytes()

    def sample_progress(self) -> dict:
        """per class: how far the random sample has got and whether it can stop."""
        if not self.sample:
            return {}
        from . import class_sample
        summary, _ = class_sample.report(self.reviewer)
        # a class with no decisions has no accuracy: send null, never NaN,
        # which is not valid JSON and stops the page loading.
        acc = lambda v: None if v is None or v != v else float(v)
        return {r["class"]: {"reviewed": int(r["reviewed"]), "sampled": int(r["sampled"]),
                             "accuracy": acc(r["accuracy_%"]), "ci": r["ci95_%"], "next": r["next"]}
                for r in summary.to_dict("records")}

    def state(self) -> dict:
        latest = decisions.load()
        counts = {}
        for a in self.by_id:
            s = decisions.status(latest.get(a, {}), self.second_opinion)
            counts[s] = counts.get(s, 0) + 1
        modes = ["first", "revisit"] if not self.second_opinion else ["first", "second", "adjudicate"]
        if self.queue.corroborated.astype(str).eq("True").any():
            modes.insert(0, "confirmed")
        if self.audit:
            modes.insert(0, "audit")
        if self.sample:
            modes.insert(0, "sample")
        return {"reviewer": self.reviewer, "classes": self.classes, "modes": modes,
                "total": len(self.by_id), "counts": counts,
                "mine": sum(1 for e in latest.values() if self.reviewer in e),
                "sample_progress": self.sample_progress(),
                "arcgis_key": self.arcgis_key()}

    def _in_pass(self, mode: str, entries: dict, row: dict) -> bool:
        if mode == "adjudicate":
            return decisions.status(entries, True) == "disputed"
        if mode == "sample":
            # the random per-class sample (class_sample.py): every tile in it
            # needs this reviewer's own call, even one a source auto-accepted.
            return row["asset_id"] in self.sample and self.reviewer not in entries
        if mode == "audit":
            # the held-back sample of strict source matches (auto_accept.py)
            return (row["asset_id"] in self.audit
                    and decisions.wants(self.reviewer, entries, self.second_opinion) == "first")
        if mode == "confirmed":
            # first looks a secondary source already agrees with: still a
            # human decision each, just a fast one.
            return (str(row.get("corroborated")) == "True"
                    and decisions.wants(self.reviewer, entries, self.second_opinion) == "first")
        return decisions.wants(self.reviewer, entries, self.second_opinion) == mode

    def next(self, mode: str, cls: str, after: int) -> dict | None:
        latest = decisions.load()
        q = self._view(mode, cls)
        if mode == "sample" and not cls:
            # across all classes, skip the ones whose estimate is already tight
            # enough; picking a class by name still lets you keep going in it.
            enough = {c for c, p in self.sample_progress().items() if p["next"] == "enough"}
            q = q[~q.asset_type.isin(enough)]
        for row in q[q.order > after].to_dict("records"):
            entries = latest.get(row["asset_id"], {})
            if self._in_pass(mode, entries, row):
                return self._item(row, entries)
        return None

    def listing(self, mode: str, cls: str, around: int, limit: int = 60) -> list:
        """the file list: a few tiles before `around` (any status) and the
        next ones still in this pass, like VIA's project panel."""
        latest = decisions.load()
        q = self._view(mode, cls)
        out = []
        for row in q[q.order >= around].to_dict("records"):
            entries = latest.get(row["asset_id"], {})
            mine = entries.get(self.reviewer)
            if mine or self._in_pass(mode, entries, row):
                out.append({"asset_id": row["asset_id"], "order": row["order"],
                            "name": row["name"], "mine": mine["decision"] if mine else ""})
            if len(out) >= limit:
                break
        return out

    def item(self, asset_id: str, mode: str = "") -> dict | None:
        row = self.by_id.get(asset_id)
        if row is None:
            return None
        if mode == "sample" and asset_id in self.sample_pos:
            # keep "next" moving through the draw order after going back
            row = {**row, "order": self.sample_pos[asset_id]}
        return self._item(row, decisions.load().get(asset_id, {}))

    def decide(self, body: dict) -> dict:
        asset_id = body["asset_id"]
        if asset_id not in self.by_id:
            raise ValueError(f"unknown asset {asset_id}")
        decision, label = body["decision"], body.get("label", "")
        if decision == "accept":
            label = self.by_id[asset_id]["asset_type"]
        elif decision == "relabel" and label not in self.classes:
            raise ValueError(f"unknown class {label}")
        return decisions.append(self.reviewer, asset_id, decision, label,
                                body.get("note", ""))


def make_handler(review: Review):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._send(200, review.state())
                elif url.path == "/api/next":
                    self._send(200, {"item": review.next(qs.get("mode", "first"),
                                                         qs.get("cls", ""),
                                                         int(qs.get("after", -1)))})
                elif url.path == "/api/list":
                    self._send(200, {"items": review.listing(qs.get("mode", "first"),
                                                             qs.get("cls", ""),
                                                             int(qs.get("around", 0)))})
                elif url.path == "/api/item":
                    self._send(200, {"item": review.item(qs.get("asset_id", ""), qs.get("mode", ""))})
                elif url.path.startswith("/footprint/") and url.path.endswith(".json"):
                    self._send(200, {"rings": review.footprint(url.path[len("/footprint/"):-len(".json")])})
                elif url.path.startswith("/naip/") and url.path.endswith(".png"):
                    png = review.naip_png(url.path[len("/naip/"):-len(".png")])
                    if png is None:
                        self._send(404, {"error": "no NAIP for this tile"})
                    else:
                        self._send(200, png, "image/png")
                elif url.path.startswith("/tile/") and url.path.endswith(".png"):
                    # look the asset up rather than trusting the path.
                    row = review.by_id.get(url.path[len("/tile/"):-len(".png")])
                    if row is None:
                        self._send(404, {"error": "unknown asset"})
                    else:
                        self._send(200, chip_for(row).read_bytes(), "image/png")
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:
                self._send(500, {"error": str(e)})

        def do_POST(self):
            try:
                if urlparse(self.path).path != "/api/decide":
                    return self._send(404, {"error": "not found"})
                length = int(self.headers.get("Content-Length", 0))
                self._send(200, review.decide(json.loads(self.rfile.read(length))))
            except Exception as e:
                self._send(400, {"error": str(e)})

    return Handler


def main():
    p = argparse.ArgumentParser(description="local label-review page")
    p.add_argument("--reviewer", required=True,
                   help="your short name, e.g. justin; names your decisions file")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--second-opinion", action="store_true",
                   help="two-person rule: relabels and unsure tiles go to another reviewer")
    args = p.parse_args()

    decisions.log_path(args.reviewer)  # validate the name before serving
    review = Review(args.reviewer, args.second_opinion)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(review))
    print(f"reviewing as {args.reviewer}: http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
