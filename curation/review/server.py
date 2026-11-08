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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ..paths import REVIEW_QUEUE
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

    def _item(self, row: dict, entries: dict) -> dict:
        return {
            **row,
            "status": decisions.status(entries, self.second_opinion),
            "mine": entries.get(self.reviewer),
            "others": [e for r, e in entries.items() if r != self.reviewer],
        }

    def state(self) -> dict:
        latest = decisions.load()
        counts = {}
        for a in self.by_id:
            s = decisions.status(latest.get(a, {}), self.second_opinion)
            counts[s] = counts.get(s, 0) + 1
        modes = ["first", "revisit"] if not self.second_opinion else ["first", "second", "adjudicate"]
        if self.queue.corroborated.astype(str).eq("True").any():
            modes.insert(0, "confirmed")
        return {"reviewer": self.reviewer, "classes": self.classes, "modes": modes,
                "total": len(self.by_id), "counts": counts,
                "mine": sum(1 for e in latest.values() if self.reviewer in e)}

    def _in_pass(self, mode: str, entries: dict, row: dict) -> bool:
        if mode == "adjudicate":
            return decisions.status(entries, True) == "disputed"
        if mode == "confirmed":
            # first looks a secondary source already agrees with: still a
            # human decision each, just a fast one.
            return (str(row.get("corroborated")) == "True"
                    and decisions.wants(self.reviewer, entries, self.second_opinion) == "first")
        return decisions.wants(self.reviewer, entries, self.second_opinion) == mode

    def next(self, mode: str, cls: str, after: int) -> dict | None:
        latest = decisions.load()
        q = self.queue if not cls else self.queue[self.queue.asset_type == cls]
        for row in q[q.order > after].to_dict("records"):
            entries = latest.get(row["asset_id"], {})
            if self._in_pass(mode, entries, row):
                return self._item(row, entries)
        return None

    def listing(self, mode: str, cls: str, around: int, limit: int = 60) -> list:
        """the file list: a few tiles before `around` (any status) and the
        next ones still in this pass, like VIA's project panel."""
        latest = decisions.load()
        q = self.queue if not cls else self.queue[self.queue.asset_type == cls]
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

    def item(self, asset_id: str) -> dict | None:
        row = self.by_id.get(asset_id)
        if row is None:
            return None
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
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
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
                    self._send(200, {"item": review.item(qs.get("asset_id", ""))})
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
